# 우수 논문 교차분석 & v4 개선안 — 실행시간 예측 메타모델 논문

> 대상: `30_draft_v3.md` (KIISE 2~3쪽, "이종 컴퓨팅 플랫폼을 아우르는 딥러닝 실행시간 예측 메타모델")
> 방법: 성능 예측 분야 우수 논문 11편을 3개 군으로 나눠 다중 렌즈(전제·검증·포지셔닝·피처·정직성)로 교차분석 → v3에 매핑.
> 분석 논문: Habitat, PreNeT, nn-Meter, NeuralPower(학습형) / Paleo, Justus, Daydream, Ithemal(분석·프로파일) / Roofline, ShuffleNet V2, MLPerf(전제·엄밀성).

---

## 0. 한 페이지 결론 (가장 중요한 것부터)

1. **[치명적] 핵심 novelty가 선행연구에 이미 있다.** "하드웨어를 피처로 넣은 단일 학습형 예측기 + 미지 하드웨어 일반화"는 **Justus et al. (IEEE BigData 2018)** 와 **PreNeT (ICPE 2025)** 가 이미 했다. Justus는 6개 GPU에서 leave-one-GPU-out까지 수행한다. → **"단일 메타모델 + 하드웨어 피처"를 핵심 기여로 내세우면 안 된다.** 방어 가능한 진짜 novelty로 피벗 필요(§2.2).
2. **[높음] 전제(14×)를 토대에서 삽화로 강등.** 강한 논문(Roofline·nn-Meter·Habitat·ShuffleNetV2)은 전제를 *메커니즘*으로 세우고 숫자는 삽화로 쓴다. 14×는 (a) 측정값이라 신뢰 필요, (b) oneDNN 소프트웨어 아티팩트라 "빌드 버그"로 반박당함. → **"방향이 뒤집히는 플랫폼 의존성"을 토대로**, 14×는 삽화로(§4.2).
3. **[높음] 검증을 leave-one-out 중심으로 승격.** 크로스플랫폼이 제목인데 랜덤 분할은 그걸 검증하지 못한다. 우리 §4.4에 **이미 leave-one-family/backend-out이 있으니** 이를 §4로 승격하고, 우리가 홀드아웃하는 건 같은 벤더 GPU가 아니라 **디바이스 클래스 전체**임을 강조(경쟁 논문보다 강함).
4. **[중간] 지표를 경쟁 논문과 비교 가능하게.** R²(log) 0.98은 Habitat(평균오차 11.8%)·nn-Meter(±10% 99%)와 비교 불가하고 로그변환으로 부풀어 보인다. → **MAPE·"±10%/±20% 이내 비율" 추가**.
5. **[중간] FLOPs 단독 대신 분석적 baseline 추가.** FLOPs 단독(R²log 0.40)은 허수아비다(Justus가 이미 이김). **Paleo식 분석 baseline(FLOPs/peak + bytes/대역폭 + 백엔드별 PPP)** 을 넣고, **PPP 수작업 튜닝 없이** 그걸 이기는 걸 보이면 학습의 가치가 입증된다.
6. **[중간] 메모리-쓰기 주장을 "레짐 주장"으로 완화 + 인과 보강.** Justus 데이터상 지배성은 레이어 타입 의존(FC=쓰기바운드, conv=연산바운드). Roofline은 time≈max(연산,메모리). → "우리 워크로드 믹스에서 가장 일관된 예측 인자"로 한정하고, **arithmetic intensity 산점도**로 우리 워크로드가 메모리바운드 영역에 있음을 보여 상관→기전으로 승격.

> 좋은 소식: v3는 이미 성숙하다(정직한 한계 서술, GroupKFold, permutation 검증, 메모리 주장의 인과 유보). 위 항목은 "구조 재작성"이 아니라 **정밀 강화 + novelty 재조정 + 인용 보강**이다.

---

## 1. 논문 × 렌즈 분석 매트릭스

| 논문 (연도) | 접근/입도 | 하드웨어 범위 | 전제 세우는 법 | 검증·일반화 | 메모리 취급 | 우리와의 관계 |
|---|---|---|---|---|---|---|
| **Justus** (BigData 2018) | 학습·레이어 | 6 NVIDIA GPU | FLOPs는 하한일 뿐; Paleo 비판 | **leave-one-GPU-out** RMSE 3.88ms; FLOPs-선형 baseline | **FC=가중치 쓰기 바운드** 명시 | **가장 가까운 선행. 반드시 인용·차별화** |
| **PreNeT** (ICPE 2025) | 학습·레이어 | 7 NVIDIA GPU | FLOP 단일지표 실패; 대역폭 고려 | **미지 GPU 홀드아웃**(L4,A4000); FLOP 대비 +72.6% | attention/embed/RNN 메모리바운드 → 우리 주장 외부근거 | 아이디어 선점자. 우리 novelty 재조정 근거 |
| **Habitat** (ATC 2021) | 스케일링+MLP·연산 | 6 NVIDIA GPU | Fig.1: FLOPS 비례 스케일링 **오차 42.5~64.9%** | 평균오차 **11.8%**; 소스 GPU 측정 필요(pairwise) | γ=roofline arithmetic intensity 중심 | **최근접 경쟁. "소스 기기 불필요"가 우리 쐐기** |
| **nn-Meter** (MobiSys 2021) | 학습·커널(fusion) | 모바일 CPU/GPU/VPU | fusion 무시하면 오차 12~50% | ±10% 정확도 99/99/83%; **디바이스별** 예측기 | fusion 암묵 | 미지 모델 일반화 vs 우리 미지 계열 |
| **NeuralPower** (ACML 2017) | 학습·다항·레이어 | 2 NVIDIA GPU | 유사 정확도 CNN 에너지 **40× 차이** | 10-fold; Paleo 대비 +68.5%; 동일 플랫폼 | 메모리 접근수 = special term | 메모리 피처 선례 |
| **Paleo** (ICLR 2017) | **분석적** | 상수 대입(K20~K80) | T=read+compute+write; **PPP**(최대 40% 미달) | AlexNet/VGG 병렬 사례; 집계 오차 미보고 | **read/write를 연산과 동등 항으로** | 분석 baseline·기전 근거 |
| **Daydream** (ATC 2020) | 프로파일·의존그래프 | what-if | 프로파일러는 서술적일 뿐 | BERT AMP +17.2%(<3% 오차) | elementwise=비연산바운드 기전 내장 | 반대극(트레이스 필요) → 우리 "미실행 예측" 대비 |
| **Ithemal** (ICML 2019) | 학습·LSTM·명령블록 | 3 x86 마이크로아키 | 분석도구는 벤더문서 부정확 | MAPE; 분석도구 대비 오차 절반 | — | "손모델링 대신 학습" 템플릿 |
| **Roofline** (CACM 2009) | 분석 이론 | 일반 | **분석적 상한**(정의라 반박불가) | — | 연산 vs 메모리 바운드 정의 | 전제 프레이밍·인과 근거 |
| **ShuffleNet V2** (ECCV 2018) | 설계 가이드 | GPU+ARM | FLOPs는 **간접지표**; 방향-가변 | GPU·ARM 양쪽 측정 | 원소연산 런타임 불균형 | FLOPs≠시간 정본 인용 |
| **MLPerf** (MLSys 2020) | 벤치 방법론 | 이종 | — | **5~10회 반복·최소최대버림; Closed division** | — | 이종 스택 비교 공정성 기준 |

---

## 2. 우리 논문 진단

### 2.1 Novelty 위협 (정면으로 봐야 함)
- **Justus(2018)**: 학습형 FFNN이 레이어별 시간을 예측, 입력에 **하드웨어 피처(메모리 대역폭·GFLOPS·코어 등)** 포함, **6 GPU에서 5개로 학습→6번째 예측**. 구조가 우리와 거의 동일.
- **PreNeT(2025)**: 단일 회귀기가 하드웨어 피처(one-hot 기기·대역폭·코어·메모리) 입력받아 **미지 GPU로 일반화**.
- 결론: "하드웨어 피처를 넣은 단일 학습형 크로스하드웨어 예측기"는 **신규가 아니다.** 이를 1번 기여로 두면 심사 직격.

### 2.2 방어 가능한 진짜 Novelty (여기로 피벗)
경쟁 논문 전부 **단일 벤더 GPU 아니면 단일 엣지 스택**이다. 우리만의 것:
1. **디바이스 클래스 이질성** — x86 CPU + CUDA + Apple ARM CPU + Apple MPS를 **한 모델**로. (Justus/PreNeT/Habitat 전부 NVIDIA GPU only)
2. **Apple Silicon 통합메모리** — 어느 경쟁 논문에도 없음.
3. **학습 + 추론을 한 프레임워크로** (nn-Meter/NeuralPower는 추론만, Habitat/PreNeT는 학습만).
4. **디바이스 클래스 전체를 홀드아웃하는 leave-one-backend-out 평가** — Justus/PreNeT은 같은 벤더 GPU 홀드아웃, 우리는 CPU↔GPU↔MPS 클래스 경계를 넘는다(더 어려운 일반화).

### 2.3 전제(14×) 취약성 → §4.2에서 해결
### 2.4 검증/지표 격차 → §4.5에서 해결
### 2.5 메모리-쓰기 인과 격차 → §4.5에서 해결

---

## 3. 우선순위 개선안

**P0 (novelty·포지셔닝 — 안 하면 위험)**
- [ ] 기여문에서 "단일 메타모델+하드웨어 피처" 단독 주장을 빼고 **§2.2의 4가지**로 재작성.
- [ ] 관련연구에 **Habitat·Justus** 추가하고 차별점 명시(현재 둘 다 없거나 Justus만 [2]로 약하게). PreNeT[5]는 "우리 아이디어의 최근접 선행"으로 정직하게 인정.

**P1 (전제·검증 — 방어력)**
- [ ] 서론·초록 전제를 **방향-가변 명제**로 재배열, 14×는 삽화로(§4.2).
- [ ] "소프트웨어 아티팩트" 반론 **선제 흡수 문장** 추가.
- [ ] §4.4의 leave-one-family/backend-out을 **§4의 주 결과로 승격**, "클래스 경계 일반화"로 프레이밍.
- [ ] **MAPE + ±10%/±20% 이내 비율** 지표 추가(경쟁 비교 가능하게).

**P2 (baseline·인과 — 완성도)**
- [ ] **Paleo식 분석 baseline** 추가, PPP 수작업 없이 이김을 보임.
- [ ] FLOPs baseline을 **진단적으로**: 잔차 vs 실행시간 그래프로 "큰 모델일수록 FLOPs가 더 틀린다"(Justus 관찰) 재현.
- [ ] 메모리 주장 **레짐으로 완화** + **arithmetic intensity 산점도**로 인과 보강.
- [ ] 외삽 한계 문단에 **훈련 봉투 밖 미검증**(Justus의 V100 과대예측)과 같은 톤 유지(이미 있음, 강화).

---

## 4. 붙여넣기용 재작성

### 4.1 초록 — 전제·novelty 재조정판
```
딥러닝 모델의 실행 시간은 플랫폼에 따라 크게, 그리고 방향까지 뒤바뀌며 달라진다.
본 연구 벤치마크에서 Apple MPS는 MobileNet에서 ARM CPU보다 약 35배 빠르지만, 작은
ANN에서는 오히려 약 2배 느리다. 이처럼 연산량(FLOPs)이나 플랫폼별 고정 배율 같은 단일
규칙으로는 실행 시간을 예측할 수 없다. 본 논문은 모델 구조 피처와 하드웨어 피처를 함께
입력받는 단일 메타모델을 제안한다. 하드웨어를 피처로 인코딩함으로써, x86 CPU·CUDA GPU·
Apple ARM CPU·Apple MPS의 네 이종 백엔드를 — GPU 단일 벤더에 머문 기존 학습형 예측기와
달리 디바이스 클래스를 가로질러 — 하나의 모델로 예측한다. 두 기기·네 백엔드의 698개
벤치마크로 학습한 이 모델은 학습·추론 시간을 R²(log) 0.985·0.981(구성 단위 분할에서도
동일)로 예측하고, 학습에서 통째로 제외한 백엔드·모델 계열로도 부분 일반화하며(R²(log)
0.61~0.89), MAPE와 ±20% 이내 비율에서도 [값]을 보인다. 나아가 실행 시간의 가장 일관된
예측 인자가 FLOPs가 아니라 메모리 쓰기 트래픽임을 네 백엔드 모두에서 확인한다.
```
> `[값]`은 §4.5 MAPE 측정 후 채움. MPS 35×/2× 수치는 사용자 측정값이니 인쇄 전 재확인.

### 4.2 서론 도입부 — 전제를 메커니즘으로
```
같은 딥러닝 모델도 실행 플랫폼에 따라 실행 시간이 크게 달라지며, 그 차이는 단순히
"어떤 하드웨어가 더 빠르다"로 정리되지 않는다. 본 연구의 벤치마크에서 Apple MPS는
MobileNet에서 ARM CPU보다 약 35배 빨랐지만, 가장 작은 ANN 계열에서는 오히려 약 2배
느렸다(표 2). 방향이 모델에 따라 뒤집히는 것이다. 이는 실행 시간이 연산 처리량, 메모리
대역폭, 그리고 플랫폼마다 다른 커널 라이브러리(oneDNN, cuDNN, MPSGraph, Accelerate)의
성숙도가 서로 다르게 절충된 결과이기 때문이다. 실제로 같은 MobileNet의 depthwise 합성곱은
Apple ARM CPU에서 데스크톱 x86 CPU보다 약 14배 느린데, 이는 PyTorch ARM 빌드에 oneDNN
최적화가 빠져 있기 때문이다. 이 사례가 시사하는 바는 중요하다. 모델에서 실행 시간으로의
매핑은 배포된 소프트웨어 스택에 의존하므로, 구조만으로 계산하는 정적 프록시(FLOPs 등)나
플랫폼별 고정 배율로는 예측할 수 없고, 스택을 학습한 경험적 모델이 필요하다.
```
> 핵심: ① 방향-가변(부호 역전)을 앞세워 "버그로 치부 불가"하게, ② oneDNN을 숨기지 않고 오히려 "그래서 학습형이 필요하다"는 논거로 전환.

### 4.3 관련연구 — Habitat·Justus·PreNeT 포지셔닝 (교체·확장)
```
실행 시간을 데이터로 예측하는 시도는 여럿 있었다. Justus 등[신규]은 레이어별 시간을
하드웨어 피처와 함께 학습해 미지의 GPU로 일반화했고, PreNeT[5]은 이를 트랜스포머 학습
시간으로 확장하였다. Habitat[신규]은 한 GPU에서 측정한 반복 시간을 다른 GPU로 스케일링해
학습 시간을 예측한다. nn-Meter[4]는 커널 융합을 반영해 엣지 추론 지연을 높은 정확도로
예측한다. 이들은 각자의 영역에서 성과를 거두었으나 공통된 범위 한계가 있다. Justus·PreNeT·
Habitat은 모두 단일 벤더 GPU에 국한되고, nn-Meter는 디바이스마다 별도 예측기를 학습하며,
Habitat은 예측하려는 GPU를 실제로 보유해 측정해야 한다. 본 논문은 (i) CPU·GPU·Apple
Silicon을 아우르는 디바이스 클래스 이질성, (ii) 통합메모리를 포함한 Apple MPS, (iii) 학습과
추론을 하나의 모델로 다루고, (iv) 소스 기기 측정 없이 정적 피처만으로 예측한다는 점에서
구별된다.
```
> [신규] = 아래 §5 참고문헌 추가.

### 4.4 기여문 — novelty 재조정
```
- 디바이스 클래스를 아우르는 단일 예측기: 기존 학습형 예측기가 단일 벤더 GPU에 머문 것과
  달리, x86 CPU·CUDA·Apple ARM CPU·Apple MPS 네 백엔드를 하드웨어 피처를 통해 하나의
  모델로 예측한다(§4.1). 통합메모리 기반 Apple Silicon을 포함한다.
- 클래스 경계 일반화 검증: 백엔드·모델 계열을 통째로 제외하고 예측하는 leave-one-out으로,
  같은 벤더 GPU가 아니라 디바이스 클래스 경계를 넘는 일반화를 정량화한다(§4.4).
- 메모리 트래픽 우위의 교차 검증: 실행 시간의 가장 일관된 예측 인자가 FLOPs가 아니라 메모리
  쓰기 트래픽임을 네 백엔드 모두에서 permutation 검증으로 확인한다(§4.3).
```

### 4.5 실험 보강 설계
1. **지표 추가**: 각 표에 R²(log)와 함께 **MAPE**, **±10%/±20% 이내 예측 비율**. Habitat(11.8%)·nn-Meter(±10% 99%)와 한 줄 대조.
2. **분석적 baseline(신규 행)**: `t̂ = FLOPs/peakFLOPS + (bytes_read+bytes_write)/BW`, 백엔드별 PPP는 짧은 벤치로 1개 상수 fit. 표 3에 "분석적(Paleo식)" 행 추가 → 우리 학습 모델이 **PPP 수작업 없이** 이를 상회함을 보임.
3. **FLOPs 진단 그래프(신규 그림)**: FLOPs-단독 예측의 잔차를 실행시간에 대해 플롯 → 큰 모델일수록 오차 증가(Justus 재현). "FLOPs는 비싼 케이스에서 가장 크게 틀린다."
4. **leave-one-backend-out 승격**: §4.4 둘째 문단을 표로 승격(각 백엔드 홀드아웃 R²log/MAPE). "클래스 경계 일반화"로 프레이밍.
5. **arithmetic intensity 산점도(신규, 인과 보강)**: 샘플별 bytes_write/FLOPs vs 실행시간 → 우리 워크로드가 메모리바운드 영역에 위치함을 시각화. §4.3의 "인과엔 직접 측정 필요" 유보를 부분 해소.
6. **메모리 주장 문장 완화**: "메모리 쓰기 트래픽이 **우리 워크로드 믹스에서** 가장 일관된 예측 인자이며, 이는 벤치마크한 계열의 메모리바운드 레짐과 부합한다. 대형 배치 합성곱 등 연산바운드 반례도 존재한다"(Justus의 FC=쓰기바운드/conv=연산바운드 인용).
7. **측정 위생 명시(신규 2~3문장, §3.1 보강)**: 워밍업 제외·타이밍 경계 정의, 10회 반복의 **중앙값 또는 최소최대 버린 평균**, 시드 고정에도 존재하는 run-to-run 분산 언급(MLPerf 관행). 그리고 크로스플랫폼 비교가 **모델·입력·배치·수치 과제를 고정하고 하드웨어+스택만 달리한 통제된 비교**임을 한 문장으로 선언(MLPerf Closed division).

---

## 5. 추가할 참고문헌 (정확한 서지)

- **Habitat**: G. Yu, Y. Gao, P. Golikov, G. Pekhimenko, "Habitat: A Runtime-Based Computational Performance Predictor for Deep Neural Network Training," in *Proc. USENIX ATC*, 2021. (arXiv:2102.00527)
- **Paleo**: H. Qi, E. R. Sparks, A. Talwalkar, "PALEO: A Performance Model for Deep Neural Networks," in *Proc. ICLR*, 2017.
- **ShuffleNet V2** (FLOPs≠latency 정본): N. Ma, X. Zhang, H.-T. Zheng, J. Sun, "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design," in *Proc. ECCV*, pp. 116–131, 2018.
- **MLPerf** (측정 위생): P. Mattson et al., "MLPerf Training Benchmark," in *Proc. MLSys*, 2020.
- (선택) **NeuralPower**: E. Cai, D.-C. Juan, D. Stamoulis, D. Marculescu, ACML 2017, PMLR 77:622–637.
- (선택) **Daydream**: H. Zhu, A. Phanishayee, G. Pekhimenko, USENIX ATC 2020.
- (선택) **Ithemal**: C. Mendis, A. Renda, S. Amarasinghe, M. Carbin, ICML 2019.

> 3쪽 분량 제약상 필수는 **Habitat·Paleo·ShuffleNetV2·MLPerf** 4편. 나머지는 지면 여유 시.

---

## 검증 주의 (인쇄 전 확인)
- 우리 **MPS 35× / 2× 수치**는 사용자 벤치 측정값 → 표 2에서 재확인.
- ShuffleNet V2의 정확한 GPU/ARM 기기명은 미검증(1080Ti/Snapdragon 추정) → 인용 시 본문 확인.
- Habitat/Justus/PreNeT 수치는 각 논문 원문에서 교차검증됨. Daydream 일부 집계 오차·Ithemal 셀 수치는 부분 미검증(본문 재확인 권장).
