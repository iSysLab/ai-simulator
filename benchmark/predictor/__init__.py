"""예측기 패키지 — 논문 파이프라인(merged_schema 120열, 로그 공간 타깃)용 추정기.

eval_fill_* 실험 스크립트와 배포 스크립트(predict_v4.py)가 같은 구현을 쓰도록
sklearn 호환 클래스를 여기에 둔다. joblib 번들 역직렬화 시 import 경로가 고정된다.
"""
from .ensemble import (
    RegimeEnsembleRegressor,
    WeightedXGBRegressor,
    ResidualCorrectedRegressor,
    LogTargetXGBRegressor,
    AlgoAverageRegressor,
    PerBackendRegressor,
    bundle_versions,
)

__all__ = [
    "bundle_versions",
    "RegimeEnsembleRegressor",
    "WeightedXGBRegressor",
    "ResidualCorrectedRegressor",
    "LogTargetXGBRegressor",
    "AlgoAverageRegressor",
    "PerBackendRegressor",
]
