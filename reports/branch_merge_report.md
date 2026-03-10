# 브랜치 병합 보고서 — ijunsoo / dal-merge

**작성자**: 김홍근  
**작성일**: 2026-03-09  
**대상 브랜치**: `origin/ijunsoo`, `origin/dal-merge`  
**병합 전략**: 유용한 코드만 포팅 (현재 `experiments/`, `models/`, `utils/` 구조 유지)

---

## 목차

1. [병합 개요](#1-병합-개요)
2. [브랜치별 기여 내용](#2-브랜치별-기여-내용)
3. [포팅한 내용 상세](#3-포팅한-내용-상세)
4. [용어 설명 (쉽게)](#4-용어-설명-쉽게)
5. [실행 방법](#5-실행-방법)
6. [파일 구조 변경](#6-파일-구조-변경)

---

## 1. 병합 개요

### 1.1 왜 병합했나?

팀원(ijunsoo, 달현)이 각자 개발한 코드 중 **현재 프로젝트에 유용한 부분**을 가져와서 통합했습니다.  
전체 구조를 바꾸지 않고, **필요한 모듈만 추가**하는 방식으로 진행했습니다.

### 1.2 병합 방식

| 방식 | 설명 |
|------|------|
| **포팅 (Port)** | 다른 브랜치의 코드를 복사해서 현재 구조에 맞게 수정 |
| **전체 병합 (Merge)** | git merge로 두 브랜치를 합침 — 이번에는 사용하지 않음 |

이번에는 **포팅**만 수행했습니다.  
ijunsoo 브랜치는 `benchmark/` 패키지 구조를 쓰고, dal-merge는 `collect/`, `features/` 구조를 쓰기 때문에,  
**현재 프로젝트 구조(experiments/, models/, utils/)를 유지**하면서 필요한 파일만 `utils/` 등에 추가했습니다.

### 1.3 가져온 브랜치

| 브랜치 | 출처 | 가져온 내용 |
|--------|------|-------------|
| **ijunsoo** | https://github.com/iSysLab/ai-simulator (ijunsoo 브랜치) | Op-Level 프로파일러 |
| **dal-merge** | https://github.com/iSysLab/ai-simulator (dal-merge 브랜치) | 하드웨어 동적 감지 (psutil) |

---

## 2. 브랜치별 기여 내용

### 2.1 ijunsoo 브랜치

**역할**: 벤치마크 프레임워크 설계, Op-Level 프로파일링

- **Op-Level 프로파일러** (`benchmark/features/op_profiler.py`)
  - 모델을 Conv, Linear, ReLU 등 **연산(op) 단위로 분해**
  - 각 op의 실행 시간을 **개별 측정** 후 합산
  - 보고서 "방법 2" (모델을 작은 연산 단위로 분해 → 각 단위 시간 측정 → 합산)에 해당

- **기타**: benchmark/ 패키지, Transformer, GAN, ONNX 파이프라인 등  
  → 현재 프로젝트에 이미 동일/유사 기능이 있어 **op_profiler만 포팅**

### 2.2 dal-merge 브랜치 (달현)

**역할**: XGBoost, 하드웨어 피처, 데이터 수집

- **하드웨어 동적 감지** (`features/extractor.py` 내 `get_hardware_info()`)
  - `psutil` 라이브러리로 CPU 코어 수, 클럭, RAM, GPU 메모리를 **실행 시점에 자동 감지**
  - 현재 프로젝트는 M1 고정값(`HARDWARE_INFO`)을 쓰지만, 다른 PC에서 실험할 때 유용

- **기타**: collect/, predictor/, 27개 feature 추출 등  
  → 현재 프로젝트는 33개 Feature와 다른 구조를 사용하므로 **get_hardware_info만 포팅**

---

## 3. 포팅한 내용 상세

### 3.1 `utils/op_profiler.py` (ijunsoo → 포팅)

**역할**: 모델을 연산(op) 단위로 분해하고, 각 op의 실행 시간을 측정합니다.

**주요 함수**:

| 함수 | 설명 |
|------|------|
| `decompose_model()` | 모델을 Conv, Linear, ReLU 등 op 리스트로 분해 |
| `measure_op_times()` | 각 op의 실행 시간을 워밍업 후 반복 측정 |
| `simulate_total_time()` | op별 시간을 합산하여 전체 시간 시뮬레이션 |
| `print_op_profile()` | 결과를 콘솔에 출력 |

**수정 사항** (원본 대비):

- **MPS 지원**: Apple Silicon GPU 사용 시 `torch.mps.synchronize()` 호출 추가
- **한국어 주석**: 각 함수·변수에 설명 추가
- **독립 모듈화**: `benchmark.` import 제거, `utils/`에서 단독 사용 가능

### 3.2 `utils/hardware_info.py` (dal-merge → 포팅)

**역할**: 실행 환경의 CPU, RAM, GPU 정보를 동적으로 수집합니다.

**주요 함수**:

| 함수 | 설명 |
|------|------|
| `get_hardware_info(device_str)` | CPU 코어, 클럭, RAM, GPU 메모리 등 반환 |
| `_get_l2_cache_mb()` | L2 캐시 크기 추정 (Windows/macOS/Linux 지원) |

**수정 사항**:

- **MPS 지원**: `device_str='mps'`일 때 M1 통합 메모리를 GPU 메모리로 간주
- **macOS L2 캐시**: `sysctl hw.l2cachesize` 등으로 감지 (원본은 Windows wmic만 지원)
- **psutil 미설치 시**: M1 고정값으로 폴백

### 3.3 `experiments/op_profile_demo.py` (신규)

**역할**: Op-Level 프로파일러를 ANN, CNN 샘플 모델에 적용하는 데모 스크립트입니다.

**실행 예**:

```bash
python experiments/op_profile_demo.py --device cpu
python experiments/op_profile_demo.py --device mps --model ann
```

---

## 4. 용어 설명 (쉽게)

### 4.1 Op-Level 프로파일링

| 용어 | 설명 |
|------|------|
| **Op** | Operation, 연산 단위. Conv, Linear, ReLU 등 하나의 레이어/연산 |
| **프로파일링** | "어디서 시간이 얼마나 걸리는지" 측정하는 것 |
| **Op-Level** | 모델 전체가 아니라, **각 연산(op)마다** 시간을 재는 것 |

**비유**: 요리 시간을 "전체 30분"이 아니라 "썰기 5분, 볶기 15분, 끓이기 10분"처럼 단계별로 재는 것.

### 4.2 포팅 (Port)

다른 프로젝트/브랜치의 코드를 **가져와서 현재 프로젝트 구조에 맞게 수정**하는 것.  
원본을 그대로 복사하는 게 아니라, import 경로, 플랫폼 지원 등을 조정합니다.

### 4.3 psutil

Python 라이브러리. CPU 사용률, 메모리, 디스크 등 **시스템 정보**를 읽어옵니다.  
`pip install psutil`로 설치합니다.

### 4.4 MPS (Metal Performance Shaders)

Apple Silicon(M1, M2 등)의 **GPU 가속** 인터페이스.  
PyTorch에서 `device='mps'`로 지정하면 M1 GPU를 사용합니다.

### 4.5 방법 1 vs 방법 2 (보고서 기준)

| 구분 | 방법 1 | 방법 2 |
|------|--------|--------|
| **접근** | 33개 Feature → ML → 전체 시간 예측 | op 분해 → 각 op 시간 측정 → 합산 |
| **구현** | `train_predictor.py`, XGBoost/RF | `op_profiler.py` |
| **상태** | 완료 (R²_log ≈ 0.93) | 포팅 완료 (데모 가능) |

---

## 5. 실행 방법

### 5.1 Op-Level 프로파일 데모

```bash
# CPU에서 ANN + CNN 프로파일
python experiments/op_profile_demo.py --device cpu

# MPS(Apple Silicon GPU)에서 ANN만
python experiments/op_profile_demo.py --device mps --model ann

# 배치 크기 32로 CNN만
python experiments/op_profile_demo.py --device cpu --model cnn --batch-size 32
```

### 5.2 하드웨어 정보 확인 (Python)

```python
from utils.hardware_info import get_hardware_info

hw = get_hardware_info('mps')
print(hw)
# {'cpu_cores': 8, 'cpu_freq_ghz': 3.2, 'ram_total_gb': 8.0, ...}
```

### 5.3 의존성

- **op_profiler**: `torch`, `numpy` (기존 환경에 있음)
- **hardware_info**: `psutil` — 없으면 `pip install psutil`

---

## 6. 파일 구조 변경

### 6.1 추가된 파일

```
dnn/
├── utils/
│   ├── op_profiler.py      # [신규] ijunsoo op_profiler 포팅
│   └── hardware_info.py    # [신규] dal-merge get_hardware_info 포팅
├── experiments/
│   └── op_profile_demo.py  # [신규] Op 프로파일 데모
└── reports/
    └── branch_merge_report.md  # [신규] 이 보고서
```

### 6.2 기존 파일과의 관계

| 기존 파일 | 포팅 모듈 | 관계 |
|----------|----------|------|
| `utils/onnx_feature_extractor.py` | op_profiler | 별도 경로. op_profiler는 PyTorch 모델 직접 분해, onnx_extractor는 ONNX 파일 파싱 |
| `utils/timer.py` | op_profiler | op_profiler 내부에서 `time.perf_counter` 사용. timer.py는 전체 모델 측정용 |
| `experiments/train_predictor.py` | hardware_info | train_predictor는 `HARDWARE_INFO` 고정값 사용. hardware_info는 동적 감지용 (선택) |

---

## 7. 요약

| 항목 | 내용 |
|------|------|
| **병합 브랜치** | ijunsoo, dal-merge |
| **포팅 방식** | 유용한 코드만 복사·수정, 현재 구조 유지 |
| **추가 파일** | `utils/op_profiler.py`, `utils/hardware_info.py`, `experiments/op_profile_demo.py` |
| **보고서** | `reports/branch_merge_report.md` (본 문서) |

---

*이 보고서는 `final_report.md` 18번 섹션의 상세 참고용 문서입니다. 통합 보고서는 `final_report.md`를 참고하세요.*
