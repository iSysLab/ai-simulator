"""그림 1: 제안 파이프라인 다이어그램 → paper/figures/fig1_pipeline.png (300dpi, 흑백)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

FIG_W, FIG_H = 3.6, 3.9  # inch — 2단 조판의 1단 폭
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 100)
ax.set_ylim(-3, 108)
ax.axis("off")


def box(x, y, w, h, text, fs=7.2, fc="white", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.6,rounding_size=1.5",
                                linewidth=0.9, edgecolor="black", facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(x1, y1, x2, y2, style="-", label=None, lx=0, ly=0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="-|>", mutation_scale=8,
                                 linewidth=0.9, linestyle=style, color="black"))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label,
                fontsize=6.3, ha="center", va="center", style="italic")


# 최상단: 모델 구성
box(18, 96, 64, 10, "6개 모델 계열 × 160개 구성\n(ANN · CNN · ResNet · MobileNet · ViT · GAN)")

# 2행: 좌 = 벤치마크 실측, 우 = ONNX export
box(3, 72, 44, 16, "PyTorch 벤치마크\n2기기 · 4백엔드\n632개 학습·추론 시간 실측", fc="0.93")
box(53, 72, 44, 16, "ONNX export\n(opset 17)\n프레임워크 중립 그래프")
arrow(38, 96, 27, 89)
arrow(62, 96, 73, 89)

# 3행: 좌 = 타깃/하드웨어, 우 = 피처 추출
box(3, 50, 44, 14, "타깃: 실측 시간 (log1p)\n+ 하드웨어 자동 감지\n(36개 피처)", fc="0.93")
box(53, 50, 44, 14, "그래프 피처 추출\n구조 · 입력 · op-level 14개\n(shape inference)")
arrow(25, 72, 25, 65)
arrow(75, 72, 75, 65)

# 4행: 130차원 피처 벡터 (하드웨어 + 그래프 피처 병합)
box(28, 32, 44, 10, "130차원 피처 벡터", bold=True)
arrow(75, 50, 58, 43)
arrow(33, 49.5, 41, 43)
ax.text(28, 46.5, "하드웨어 피처", fontsize=6.3, ha="right", va="center",
        style="italic")

# 5행: 통합 회귀 모델
box(22, 14, 56, 10, "XGBoost 단일 크로스플랫폼 모델\n(네 백엔드 통합 학습)")
arrow(50, 32, 50, 25)
arrow(10, 49.5, 10, 19, style=":")
arrow(10, 19, 21, 19, style=":")
ax.text(8, 33, "타깃", fontsize=6.3, ha="right", va="center", style="italic")

# 최하단: 출력
box(22, 0, 56, 8, "학습·추론 시간 예측\nR²(log) 0.988 / 0.989", fs=7.2)
arrow(50, 14, 50, 8.5)

fig.savefig("paper/figures/fig1_pipeline.png", dpi=300, bbox_inches="tight",
            facecolor="white")
print("저장: paper/figures/fig1_pipeline.png")
