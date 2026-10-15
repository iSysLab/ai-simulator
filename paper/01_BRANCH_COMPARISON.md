# 브랜치 비교 분석 & 통합 전략 (main v2.0 ↔ ijunsoo v3.0)

> 두 브랜치는 공통 조상 `5282298 [fix] Linux 한글 폰트`에서 갈라짐.
> - **main**: `integration` 병합 → 111피처 v2.0, `dal/`+`support/`, past/·docs/images 자산
> - **ijunsoo**: 독자 발전 → 96피처 **v3.0 Zero-Config 크로스플랫폼**, `platform/` 모듈
> - **데이터는 둘 다 안전** (Mac json은 ijunsoo가 git 추적 중).

---

## 1. 정면 비교

| 측면 | **main v2.0** | **ijunsoo v3.0** | 논문 채택 |
|---|---|---|---|
| 피처 스키마 | 111차원 (5그룹 A~E) | **96차원 (17카테고리)** | v3.0 뼈대 + v2.0 Op-level |
| 하드웨어 감지 | `support/hardware_info.py` | **`platform/` (detector·cpu·gpu·memory_info·compatibility) — Zero-Config** | ✅ v3.0 |
| 벤치 데이터 | 378행 (Win: CUDA 160 + CPU 218) | **698행 (Win 378 + Mac 320)** | ✅ v3.0 |
| Mac 하드웨어 기재 | 통합 언급만 | **Apple M4, 24GB 통합메모리, MPS** (명확) | ✅ v3.0 (※M1 아님, M4) |
| 크로스플랫폼 발견 | 약함 | **MobileNet depthwise: ARM CPU 14x 느림 / MPS 35x 가속** | ✅ v3.0 (핵심) |
| 예측 성능 (R²_log) | CPU+CUDA 0.92~0.998 | CPU+CUDA 0.966~0.998 (약간↑) | v3.0 표 |
| 메모리바운드 분석 | **강함**: `total_op_memory_write` 0.64, 산점도 SVG | fig6 중요도 (수치 미상) | ✅ v2.0 흡수 |
| ONNX 예측 경로 | `scripts/` 정리 | 루트 `export_onnx.py`/`predict_from_onnx.py` | 둘 중 동작본 |
| 문서 자산 | `docs/images/*.svg`, `past/` 이력 | fig1~9 PNG | 둘 다 활용 |

---

## 2. 통합 스토리 (논문 관점에서 두 브랜치의 상보성)

논문의 크로스플랫폼 주장은 **두 층위**로 구성되며, 각 브랜치가 한 층씩 채운다.

### 층위 A — 관찰(Observation): "같은 모델도 플랫폼마다 실행 특성이 다르다"
- **출처: ijunsoo v3.0**
- 근거: 698샘플 2플랫폼(Win RTX 4060 Ti + Mac M4 MPS) 벤치.
- 킬러 발견: **MobileNet의 depthwise separable conv가 Mac ARM CPU에서 14배 느림**(PyTorch ARM 빌드에 MKLDNN/oneDNN 미포함), 반면 MPS에서 **35배 가속**되어 정상화.
- → "왜 하드웨어를 피처로 넣어야 하는가"에 대한 강력한 동기.

### 층위 B — 예측(Prediction): "하드웨어를 피처로 넣은 단일 메타모델로 통합 예측"
- **출처: main v2.0 + ijunsoo v3.0 공통**
- 근거: 하드웨어 피처(33/그룹D 또는 32/HW카테고리)를 입력에 포함 → 디바이스별 예측기 불필요.
- 분석 깊이: **v2.0의 메모리바운드 피처 중요도**(`total_op_memory_write` 0.64 ≫ `flops` 0.03) — 층위 A의 "왜 다른가"를 정량적으로 설명.

### 통합 논지 (개정판)
> 딥러닝 실행 특성은 플랫폼(CPU/CUDA/Apple MPS)에 따라 최대 수십 배까지 달라진다(관찰). 이를 FLOPs만으로는 설명할 수 없고 **메모리 트래픽·하드웨어 특성**이 지배한다(분석). 따라서 하드웨어를 입력 피처로 인코딩한 **단일 크로스플랫폼 메타모델**로 학습·추론·메모리를 예측한다(제안). ← v3.0 관찰 + v2.0 분석 + 공통 예측.

---

## 3. 통합 시 해결할 항목

- [ ] **피처 수 확정**: 논문엔 "약 100차원 구조+하드웨어 피처"로 서술하고, 부록/표에 카테고리 요약. 96 vs 111 혼선 방지 — **v3.0 96 기준 + Op-level 언급**.
- [ ] **Mac 사양 정정**: research-plan.mdc는 "M1"이라 적혀 있으나 v3.0 README는 **M4, 24GB 통합메모리**. → 실제 측정 기기 확인 후 확정.
- [ ] **MPS 예측 성능**: 두 브랜치 모두 예측표가 CPU+CUDA. 크로스플랫폼 "예측"을 강화하려면 **MPS 단독/통합 메타모델 CV** 추가 권장(있으면 표에 MPS 행 추가).
- [ ] **메모리바운드 수치 재확인**: `total_op_memory_write` 0.64는 v2.0 산출. v3.0 96피처에도 동일 op-level 피처 있는지 확인(`memory_bytes`, op 비율).
- [ ] **동작 기준 코드**: 실제로 돌릴 땐 한 브랜치를 골라야 함(구조가 상이). 논문 수치 재생성은 **ijunsoo(v3.0)** 에서 하는 게 데이터·크로스플랫폼과 일치.

---

## ★ 최종 결정 (구현 완료) — 병합 스키마 131차원

"96 vs 111" 논쟁의 결론: **개수가 아니라 어떤 피처냐**가 핵심. 111에만 있던 op-level 메모리 피처(`total_op_memory_write` 등)가 논문의 메모리바운드 근거이고, 96에만 있던 크로스플랫폼 하드웨어 피처(`gpu_tensor_core_count`, `is_unified_memory` 등)도 필요 → **둘을 병합**.

**병합 규칙**: v2.0(111) 전량 유지 + v3.0 고유 29개 중 이름중복 9개 제외 + 신규 20개 추가 = **131차원**.

| 구현물 | 위치 |
|---|---|
| 병합 스키마 정의 (131) | `benchmark/features/merged_schema.py` (`MERGED_FEATURE_COLUMNS`) |
| op-level 피처 보강 스크립트 | `enrich_oplevel.py` → `results/*_enriched.json` |
| 학습 옵션 추가 | `train_predictor.py --features merged` |

**제외한 중복 9개**: num_filters=cnn_num_filters, kernel_size=cnn_kernel_size, max_channel_width=cnn_max_channels, device_type_encoded=device_encoded, num_attention_layers=vit_num_encoder_layers, sequence_length=seq_length, gpu_cores=gpu_core_count, cpu_cores=cpu_cores_physical, use_batchnorm=has_batch_norm.

**신규 20개**: num_bn_layers, num_pool_layers, num_activation_layers, flops_per_sample, params_per_flop, has_layer_norm, has_dropout, hidden_size, model_family, model_arch, stride, padding, generator_layers, discriminator_layers, dataset_type, input_dtype, input_elements, gpu_tensor_core_count, gpu_compute_capability, gpu_clock_ghz.

---

## 4. 권장 결론

**논문 뼈대 = ijunsoo v3.0** (크로스플랫폼 관찰·데이터·Zero-Config가 novelty의 핵심),
**분석 깊이 = main v2.0에서 흡수** (메모리바운드 피처 중요도, 산점도 SVG, ONNX 경로 서술).

수치 재생성·그림은 **ijunsoo 브랜치로 전환 후** 진행하는 것을 권장(데이터 698샘플·플랫폼 모듈과 일치). 단, git 이력상 두 브랜치는 병합이 까다로우므로 **논문 문서(`paper/`)는 브랜치 무관하게 유지**하고, 코드 실행만 필요 시 ijunsoo로 전환.
