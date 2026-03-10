# Windows 환경 이전 가이드

**작성일**: 2026-03-09  
**대상**: Mac에서 작업한 DNN 프로젝트를 Windows PC로 옮겨 작업하고 싶을 때

---

## 1. 이전할 항목 요약

### 1.1 반드시 옮겨야 할 것

| 항목 | 경로 | 설명 |
|------|------|------|
| **실험 데이터 (CSV)** | `data/stage1/`, `data/stage2/`, `data/stage3/`, `data/stage4/` | ANN, CNN, Transformer, GAN 측정 결과 |
| **학습된 예측 모델** | `models/trained/*.pkl` | XGBoost, Random Forest, feature_columns |
| **ONNX 샘플** | `data/stage5/onnx_samples/*.onnx` | ONNX 예측 데모용 |
| **소스 코드** | `experiments/`, `models/`, `utils/` | 전체 Python 코드 |
| **보고서** | `reports/*.md` | final_report, progress_report 등 |

### 1.2 옮기지 않아도 되는 것 (자동 생성 가능)

| 항목 | 이유 |
|------|------|
| `data/raw/`, `data/MNIST/`, `data/cifar-10-*` | `.gitignore` 대상, 실행 시 자동 다운로드 |
| `.venv/`, `venv/` | Windows에서 새로 `python -m venv .venv` |
| `__pycache__/` | 실행 시 자동 생성 |

### 1.3 용량 대략

- CSV + pkl + onnx + 코드: **수십 MB** 수준
- MNIST/CIFAR-10 원본 데이터: **수백 MB** (필요 시 Windows에서 다시 다운로드)

---

## 2. 이전 방법

### 방법 A: Git 사용 (권장)

Windows PC에서:

```powershell
# Git이 설치되어 있다면
git clone https://github.com/iSysLab/ai-simulator.git dnn
cd dnn
git checkout khg9859/present   # 또는 해당 브랜치
```

- **장점**: 코드·데이터 동기화, 버전 관리
- **주의**: `models/trained/*.pkl`, `data/stage*/*.csv` 등이 Git에 커밋되어 있어야 함.  
  `.gitignore`에 있으면 수동 복사 필요.

### 방법 B: ZIP/압축 파일로 복사

1. Mac에서 프로젝트 폴더를 ZIP으로 압축
2. USB, 클라우드(Google Drive, OneDrive 등), 이메일로 Windows로 전송
3. Windows에서 압축 해제

**압축 시 포함할 폴더**:
```
dnn/
├── experiments/
├── models/          # trained/*.pkl 포함 확인
├── utils/
├── data/            # stage1~5 CSV, onnx_samples
├── reports/
├── .cursor/
└── (requirements.txt 등 있으면 포함)
```

### 방법 C: 클라우드 동기화

- OneDrive, Google Drive 등에 `dnn` 폴더를 넣고 동기화
- Mac과 Windows 양쪽에서 같은 폴더를 사용

---

## 3. Windows 환경 설정

### 3.1 Python 설치

1. [python.org](https://www.python.org/downloads/)에서 Python 3.10 이상 다운로드
2. 설치 시 **"Add Python to PATH"** 체크
3. PowerShell에서 확인: `python --version`

### 3.2 가상환경 생성

```powershell
cd C:\Users\본인계정\Desktop\dnn   # 프로젝트 경로
python -m venv .venv
.venv\Scripts\activate
```

### 3.3 패키지 설치

```powershell
pip install torch torchvision
pip install scikit-learn xgboost matplotlib pandas numpy joblib
pip install onnx onnxscript
pip install psutil   # hardware_info.py 사용 시
```

**NVIDIA GPU 사용 시** (CUDA):
- [PyTorch CUDA 설치](https://pytorch.org/get-started/locally/) 에서 CUDA 버전에 맞는 명령 확인
- 예: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

### 3.4 XGBoost (Windows)

- Mac의 `brew install libomp`는 Windows에 해당 없음
- Windows에서는 보통 `pip install xgboost`만으로 동작
- 오류 시: [XGBoost Windows 설치 가이드](https://xgboost.readthedocs.io/en/stable/install.html) 참고

---

## 4. Mac vs Windows 차이 (디바이스)

| 환경 | GPU | device 값 | 비고 |
|------|-----|-----------|------|
| **Mac (M1)** | MPS (Apple Silicon) | `mps` | 현재 학습 데이터 기준 |
| **Windows (NVIDIA)** | CUDA | `cuda` | MPS 대신 사용 |
| **Windows (GPU 없음)** | — | `cpu` | CPU만 사용 |

### 4.1 예측 모델(device_encoded) 호환

학습 데이터는 `cpu`(0)와 `mps`(1)로 수집됨.  
Windows에서:

- **CPU만 사용**: `--device cpu` → `device_encoded=0` → 그대로 사용 가능
- **CUDA 사용**: `--device cuda` → `device_encoded=1`로 처리 (GPU로 간주)

즉, **예측 시** `cuda`를 `mps`와 동일하게 "GPU"로 취급하면 됨.

### 4.2 실험 스크립트 (새 데이터 수집 시)

Windows에서 **새로** ANN/CNN/Transformer/GAN 시간을 측정하려면:

- `devices = ['cpu', 'mps']` → `devices = ['cpu', 'cuda']` 로 변경
- `device == 'mps'` 조건 → `device in ('mps', 'cuda')` 로 변경
- `torch.mps.synchronize()` → `torch.cuda.synchronize()` (cuda일 때)

`utils/timer.py`는 이미 `cuda` 지원.  
실험 스크립트에서 `mps`만 쓰는 부분을 `cuda`로 확장하면 됨.

---

## 5. 실행 순서 (Windows에서)

```powershell
# 1. 가상환경 활성화
.venv\Scripts\activate

# 2. 예측 모델이 있다면 (이미 학습된 .pkl 사용)
python experiments/train_predictor.py   # 재학습 불필요 시 생략

# 3. ONNX 샘플로 예측 테스트
python experiments/predict_from_onnx.py data/stage5/onnx_samples/ann_l2_w256.onnx --device cpu

# 4. (선택) Op 프로파일 데모
python experiments/op_profile_demo.py --device cpu
```

---

## 6. 트러블슈팅 (Windows)

| 문제 | 대응 |
|------|------|
| `ModuleNotFoundError` | `pip install` 로 해당 패키지 설치 |
| `XGBoostError: libomp` | Windows에서는 libomp 불필요. XGBoost 재설치: `pip install --upgrade xgboost` |
| `torch.cuda.is_available() == False` | NVIDIA 드라이버, CUDA 툴킷 설치 확인. CPU만 쓰려면 `--device cpu` |
| 경로 오류 (`\` vs `/`) | Python은 `\`와 `/` 모두 처리. `pathlib.Path` 또는 `os.path.join` 사용 권장 |
| 한글 경로 깨짐 | 프로젝트를 영문 경로에 두는 것을 권장 (예: `C:\Projects\dnn`) |

---

## 7. 체크리스트

이전 전 Mac에서:

- [ ] `models/trained/*.pkl` 존재 확인
- [ ] `data/stage1/ann_mnist_results.csv` 등 CSV 존재 확인
- [ ] `data/stage5/onnx_samples/*.onnx` 존재 확인
- [ ] Git push로 원격에 최신 상태 반영 (Git 사용 시)

Windows에서:

- [ ] Python 3.10+ 설치
- [ ] 가상환경 생성 및 패키지 설치
- [ ] `predict_from_onnx.py`로 예측 테스트
- [ ] (선택) `train_predictor.py` 실행 가능 여부 확인

---

## 8. 추가 유틸 (Windows 호환)

프로젝트에 `utils/device_utils.py`가 포함되어 있습니다.

- **get_best_device()**: CUDA(Windows) > MPS(Mac) > CPU 순으로 자동 선택
- **device_to_encoded()**: 예측 모델용 device_encoded 변환 (cuda → 1)
- **synchronize_device()**: CUDA/MPS 동기화 (시간 측정 정확도용)

`predict_from_onnx.py`는 `--device cuda`를 지원합니다.

---

*이 가이드는 `final_report.md` 14번 "개발 환경 및 실행 방법"의 Windows 보완용입니다.*
