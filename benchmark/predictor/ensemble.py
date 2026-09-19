"""구간 게이트 앙상블과 비교 기준선 — 설계는 paper/70_ensemble_design.md.

모든 클래스는 sklearn 호환(BaseEstimator, RegressorMixin)이며 입출력은 v4j/v4k와 같은
로그 공간 z = log1p(y_seconds)다. 그래서 eval_fill_v4j.nested_oof()·GridSearchCV를
수정 없이 쓸 수 있고, joblib 번들로 저장·로드된다.

  LogTargetXGBRegressor      (F) 타깃 변환만 바꾼 단일 XGB. log1p(초)는 0.1초 아래에서 사실상
                                 선형이라 소형 셀의 상대 오차가 손실에 거의 안 잡힌다 → log(y) 로 학습
  WeightedXGBRegressor       (D) 단일 XGB + 소형 셀 sample_weight
  ResidualCorrectedRegressor (E) base XGB + 소형 구간 잔차 모델(내부 K-겹 OOF 잔차로 학습)
  RegimeEnsembleRegressor    (B) base XGB + 소형 전문 XGB + 게이트(hard / soft / clf)
  AlgoAverageRegressor       (A) XGB·GB·RF 로그 예측 평균
  PerBackendRegressor        (C) 백엔드 열 기준 모델 4개 (대조군)

공통 규칙: fit()은 학습 폴드의 z만 본다. predict()는 X와 base 예측만 쓴다(라우팅에 실제 z 사용 금지).
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold
from xgboost import XGBClassifier, XGBRegressor

_EPS = 1e-6


def _xgb(n_estimators, max_depth, learning_rate, random_state, **kw):
    return XGBRegressor(n_estimators=n_estimators, max_depth=max_depth,
                        learning_rate=learning_rate, random_state=random_state,
                        n_jobs=1, verbosity=0, **kw)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def z_to_seconds(z):
    return np.expm1(z)


def seconds_to_z(y):
    return np.log1p(y)


# ----------------------------------------------------------------------------- (F)
class LogTargetXGBRegressor(BaseEstimator, RegressorMixin):
    """단일 XGB, 학습 타깃 변환만 선택. 입력 z=log1p(초)를 내부에서 바꿔 학습하고 예측은 z로 되돌린다.

    transform:
      'log1p'    z 그대로 (현재 논문 모델과 동일 → 대조용)
      'log'      log(y + eps): 전 구간 상대 오차 손실
      'log1p_ms' log1p(y·1000): 1 ms 아래만 선형, 그 위는 로그
    """

    def __init__(self, transform="log", n_estimators=400, max_depth=3, learning_rate=0.1,
                 random_state=42):
        self.transform = transform
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

    def _fwd(self, z):
        y = z_to_seconds(z)
        if self.transform == "log1p":
            return z
        if self.transform == "log":
            return np.log(y + _EPS)
        if self.transform == "log1p_ms":
            return np.log1p(y * 1000.0)
        raise ValueError(self.transform)

    def _inv(self, t):
        if self.transform == "log1p":
            return t
        if self.transform == "log":
            return seconds_to_z(np.clip(np.exp(t) - _EPS, 0, None))
        if self.transform == "log1p_ms":
            return seconds_to_z(np.expm1(t) / 1000.0)
        raise ValueError(self.transform)

    def fit(self, X, z, sample_weight=None):
        X, z = np.asarray(X, float), np.asarray(z, float)
        self.model_ = _xgb(self.n_estimators, self.max_depth, self.learning_rate, self.random_state)
        self.model_.fit(X, self._fwd(z), sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self._inv(self.model_.predict(np.asarray(X, float)))


# ----------------------------------------------------------------------------- (D)
class WeightedXGBRegressor(BaseEstimator, RegressorMixin):
    """단일 XGB(z 타깃) + 소형 셀 가중. weight = 1 + alpha·s(z), s는 scheme에 따라
    'step'   s = 1[z < tau_s]
    'linear' s = clip((tau_s − z)/tau_s, 0, 1)
    alpha=0이면 현재 논문 모델과 동일."""

    def __init__(self, tau_s=np.log1p(0.05), alpha=5.0, scheme="step",
                 n_estimators=400, max_depth=3, learning_rate=0.1, random_state=42):
        self.tau_s = tau_s
        self.alpha = alpha
        self.scheme = scheme
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

    def _weights(self, z):
        if self.scheme == "step":
            s = (z < self.tau_s).astype(float)
        elif self.scheme == "linear":
            s = np.clip((self.tau_s - z) / max(self.tau_s, _EPS), 0, 1)
        else:
            raise ValueError(self.scheme)
        return 1.0 + self.alpha * s

    def fit(self, X, z):
        X, z = np.asarray(X, float), np.asarray(z, float)
        self.model_ = _xgb(self.n_estimators, self.max_depth, self.learning_rate, self.random_state)
        self.model_.fit(X, z, sample_weight=self._weights(z))
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X, float))


# ----------------------------------------------------------------------------- (E)
class ResidualCorrectedRegressor(BaseEstimator, RegressorMixin):
    """base XGB + 소형 구간 잔차 모델.
    fit: base를 전체에 학습. 학습 폴드 안에서 K-겹 OOF 잔차 r = z − ẑ_b 를 구하고,
         z < tau_s 인 셀의 잔차로 두 번째 XGB를 학습(과적합된 in-sample 잔차는 쓰지 않음).
    predict: ẑ = ẑ_b + w·r̂,  w = σ((tau − ẑ_b)/beta)  (beta→0이면 hard).
    주의: 내부 K-겹은 그룹 정보가 없어 KFold — 같은 구성의 다른 백엔드 셀이 잔차 추정에 섞일 수 있다."""

    def __init__(self, tau=np.log1p(0.03), tau_s=np.log1p(0.05), beta=0.01, inner_folds=4,
                 n_estimators=400, max_depth=3, learning_rate=0.1,
                 res_n_estimators=200, res_max_depth=3, res_learning_rate=0.1,
                 min_cells=20, random_state=42):
        self.tau = tau
        self.tau_s = tau_s
        self.beta = beta
        self.inner_folds = inner_folds
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.res_n_estimators = res_n_estimators
        self.res_max_depth = res_max_depth
        self.res_learning_rate = res_learning_rate
        self.min_cells = min_cells
        self.random_state = random_state

    def fit(self, X, z):
        X, z = np.asarray(X, float), np.asarray(z, float)
        mk = lambda: _xgb(self.n_estimators, self.max_depth, self.learning_rate, self.random_state)
        self.base_ = mk().fit(X, z)
        oof = np.zeros(len(z))
        for tr, te in KFold(self.inner_folds, shuffle=True, random_state=self.random_state).split(X):
            oof[te] = mk().fit(X[tr], z[tr]).predict(X[te])
        small = z < self.tau_s
        self.n_small_ = int(small.sum())
        if self.n_small_ >= self.min_cells:
            self.residual_ = _xgb(self.res_n_estimators, self.res_max_depth,
                                  self.res_learning_rate, self.random_state)
            self.residual_.fit(X[small], (z - oof)[small])
        else:
            self.residual_ = None
        return self

    def predict_components(self, X):
        X = np.asarray(X, float)
        zb = self.base_.predict(X)
        if self.residual_ is None:
            return zb, np.zeros_like(zb), np.zeros_like(zb)
        r = self.residual_.predict(X)
        w = (zb < self.tau).astype(float) if self.beta <= 0 else _sigmoid((self.tau - zb) / self.beta)
        return zb, r, w

    def predict(self, X):
        zb, r, w = self.predict_components(X)
        return zb + w * r


# ----------------------------------------------------------------------------- (B)
class RegimeEnsembleRegressor(BaseEstimator, RegressorMixin):
    """base XGB + 소형 전문 XGB + 게이트.
    gate: 'hard' w = 1[ẑ_b < tau] · 'soft' w = σ((tau − ẑ_b)/beta) · 'clf' w = P(z < tau | X)
    전문 모델은 z < tau_s 인 학습 셀로만 학습(tau_s ≥ tau 권장). 셀이 min_cells 미만이면 base만 쓴다."""

    def __init__(self, tau=np.log1p(0.03), tau_s=np.log1p(0.05), gate="soft", beta=0.01,
                 n_estimators=400, max_depth=3, learning_rate=0.1,
                 spec_n_estimators=200, spec_max_depth=3, spec_learning_rate=0.1,
                 clf_n_estimators=200, clf_max_depth=3, min_cells=20, random_state=42):
        self.tau = tau
        self.tau_s = tau_s
        self.gate = gate
        self.beta = beta
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.spec_n_estimators = spec_n_estimators
        self.spec_max_depth = spec_max_depth
        self.spec_learning_rate = spec_learning_rate
        self.clf_n_estimators = clf_n_estimators
        self.clf_max_depth = clf_max_depth
        self.min_cells = min_cells
        self.random_state = random_state

    def fit(self, X, z):
        X, z = np.asarray(X, float), np.asarray(z, float)
        self.base_ = _xgb(self.n_estimators, self.max_depth, self.learning_rate, self.random_state).fit(X, z)
        small = z < self.tau_s
        self.n_small_ = int(small.sum())
        self.specialist_ = None
        self.clf_ = None
        if self.n_small_ >= self.min_cells:
            self.specialist_ = _xgb(self.spec_n_estimators, self.spec_max_depth,
                                    self.spec_learning_rate, self.random_state).fit(X[small], z[small])
            if self.gate == "clf":
                lab = (z < self.tau).astype(int)
                if 0 < lab.sum() < len(lab):
                    self.clf_ = XGBClassifier(n_estimators=self.clf_n_estimators,
                                              max_depth=self.clf_max_depth, learning_rate=0.1,
                                              random_state=self.random_state, n_jobs=1,
                                              verbosity=0).fit(X, lab)
                else:
                    self.specialist_ = None
        return self

    def gate_weights(self, X, zb=None):
        X = np.asarray(X, float)
        if self.specialist_ is None:
            return np.zeros(len(X))
        if zb is None:
            zb = self.base_.predict(X)
        if self.gate == "hard":
            return (zb < self.tau).astype(float)
        if self.gate == "soft":
            return _sigmoid((self.tau - zb) / max(self.beta, _EPS))
        if self.gate == "clf":
            return self.clf_.predict_proba(X)[:, 1]
        raise ValueError(self.gate)

    def predict_components(self, X):
        X = np.asarray(X, float)
        zb = self.base_.predict(X)
        if self.specialist_ is None:
            return zb, zb.copy(), np.zeros_like(zb)
        return zb, self.specialist_.predict(X), self.gate_weights(X, zb)

    def predict(self, X):
        zb, zs, w = self.predict_components(X)
        return (1.0 - w) * zb + w * zs


# ----------------------------------------------------------------------------- (A)
class AlgoAverageRegressor(BaseEstimator, RegressorMixin):
    """XGB·GB·RF의 로그 예측 평균. 각 구성원은 v4k tuned_comparison에서 동률이었던 설정을 기본값으로.
    members: 'xgb,gb,rf' 부분집합 문자열."""

    def __init__(self, members="xgb,gb,rf", n_estimators=400, max_depth=3, learning_rate=0.1,
                 rf_n_estimators=200, rf_max_depth=None, random_state=42):
        self.members = members
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_depth = rf_max_depth
        self.random_state = random_state

    def fit(self, X, z):
        X, z = np.asarray(X, float), np.asarray(z, float)
        self.models_ = []
        for m in self.members.split(","):
            m = m.strip()
            if m == "xgb":
                est = _xgb(self.n_estimators, self.max_depth, self.learning_rate, self.random_state)
            elif m == "gb":
                est = GradientBoostingRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                                learning_rate=self.learning_rate,
                                                random_state=self.random_state)
            elif m == "rf":
                est = RandomForestRegressor(n_estimators=self.rf_n_estimators, max_depth=self.rf_max_depth,
                                            random_state=self.random_state, n_jobs=1)
            else:
                raise ValueError(m)
            self.models_.append(est.fit(X, z))
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        return np.mean([m.predict(X) for m in self.models_], axis=0)


# ----------------------------------------------------------------------------- (C)
class PerBackendRegressor(BaseEstimator, RegressorMixin):
    """X의 backend_col 열 값마다 XGB 하나. 통합 모델과의 대조군.
    학습에 없던 백엔드 값이 predict에 오면 전체 학습 모델(fallback_)로 예측."""

    def __init__(self, backend_col=-1, transform="log1p", n_estimators=400, max_depth=3,
                 learning_rate=0.1, random_state=42):
        self.backend_col = backend_col
        self.transform = transform          # 구성원의 타깃 변환 (LogTargetXGBRegressor와 동일 의미)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

    def _split(self, X):
        X = np.asarray(X, float)
        b = X[:, self.backend_col]
        Xf = np.delete(X, self.backend_col, axis=1)
        return b, Xf

    def _mk(self):
        return LogTargetXGBRegressor(transform=self.transform, n_estimators=self.n_estimators,
                                     max_depth=self.max_depth, learning_rate=self.learning_rate,
                                     random_state=self.random_state)

    def fit(self, X, z):
        z = np.asarray(z, float)
        b, Xf = self._split(X)
        self.models_ = {}
        for v in np.unique(b):
            m = b == v
            self.models_[float(v)] = self._mk().fit(Xf[m], z[m])
        self.fallback_ = self._mk().fit(Xf, z)
        return self

    def predict(self, X):
        b, Xf = self._split(X)
        out = np.zeros(len(b))
        for v in np.unique(b):
            m = b == v
            model = self.models_.get(float(v), self.fallback_)
            out[m] = model.predict(Xf[m])
        return out


def bundle_versions():
    """joblib 번들에 같이 저장할 라이브러리 버전."""
    import sklearn
    import xgboost
    return {"sklearn": sklearn.__version__, "xgboost": xgboost.__version__, "numpy": np.__version__}
