# final 브랜치 통합 노트 (팀 공유용)

`hong-0311`, `dal-merge`, `ijunsoo` 세 브랜치를 **공통 조상 없이** 파일 단위로 합친 `final` 브랜치입니다.

## 무엇을 어디서 가져왔나

### ijunsoo
- **`benchmark/` 패키지**: 설정 생성기(160개 조합), 모델 레지스트리, `DeviceManager` / `ExperimentRunner`, `ResultsManager`(JSON 증분 저장).
- **루트 스크립트**: `run_benchmark.py`, `train_predictor.py`(장치별 분리·GridSearchCV), `visualize_results.py`(9종 그림), `export_onnx.py`, `predict_from_onnx.py`.
- **`results/`**: 기존 벤치마크 JSON·CSV·학습된 모델·figure.
- **`CLAUDE.md`**: AI 어시스턴트용 요약.

### dal-merge
- **`features/` + `utils/`**: **111차원** 피처 추출(`features/extractor.py`), op-level 통계(`features/op_profiler.py`), 33개 하드웨어 필드(`utils/hardware_info.py`).
- **`collect/`**: ANN/CNN/Transformer/GAN 독립 수집기 + 재개(resume) 로직.
- **`models/`**(루트): collect용 `SimpleANN`, `SimpleCNN`, ViT, GAN 정의.
- **`scripts/`**: ONNX 내보내기·`onnxruntime` 실측 vs 예측(`scripts/predict_from_onnx.py`).
- **`predictor/train.py`**: CSV 기반 111피처 학습(참고용으로 유지).
- **`reports/`**(구 `report/`): `branch_integration.md` 등.

### hong-0311
- **`reports/final_report.md`**: Stage 1~5 통합 연구 본문(교수님 보고용).
- **`experiments/`**: Stage별 실험·분석·ONNX 데모(`experiments/predict_from_onnx.py`는 92피처·앙상블 경로).
- **`data/stage1~5/`**, **`models/trained/*.pkl`**: 단계별 CSV·ONNX 샘플·학습된 메타모델.
- **`utils/onnx_feature_extractor.py`**: ONNX → 피처(Stage 5 호환).
- **루트 `README.md`**: 상세 서술형 보고서.

## 디렉터리 역할 요약

| 경로 | 역할 |
|------|------|
| `benchmark/` | 통합 벤치마크(권장 진입점) |
| `features/` | 111차원 PyTorch 피처(단일 소스) |
| `collect/` | dal 스타일 그리드 수집 |
| `scripts/` | ONNX 파이프라인·Stage 합본 시각화 복사본 |
| `experiments/` | Hong Stage 1~5 스크립트 |
| `reports/` | 최종 보고서 + dal 통합 문서 |

## 피처 차원

- **111차원**: `train_predictor.py`의 `FEATURE_COLUMNS`와 루트 `features/extractor.py` 출력이 일치하도록 맞춤.
- **`benchmark/features/extractor.py`**: ijunsoo `model_type` → dal `ann/cnn/...` 매핑 후 루트 추출기 호출, `model_family_encoded`는 6종(0~5)으로 덮어씀.

## 팀원에게 전달할 한 줄

- **Hong**: Stage 보고·ONNX 샘플·92피처 시절 메타모델 경로는 `experiments/` + `models/trained/`.
- **Dal**: 111피처·수집기·ONNX 검증은 `features/` + `collect/` + `scripts/`.
- **Junsoo**: 모듈형 벤치마크·시각화·JSON 파이프라인은 `benchmark/` + 루트 `run_benchmark.py` / `visualize_results.py`.
