# ============================================================
# 3단계 (개선판): DNN 실행 시간 예측 모델 개발
# ============================================================
#
# 기존 버전 대비 개선 사항:
#   1. 로그 변환 (Log Transform)
#      - 추론 시간: 0.03ms ~ 799ms (약 26,000배 차이)
#      - 학습 시간: 3.8초 ~ 6332초 (약 1,600배 차이)
#      - 이렇게 넓은 범위를 그대로 학습하면 큰 값에만 끌려다님
#      - log1p() 변환으로 값 범위를 균일하게 만들어 예측 성능 향상
#
#   2. K-Fold 교차검증 (K=5)
#      - 10개짜리 테스트 1번 → 5번 나눠서 평균 성능 측정
#      - 50개처럼 적은 데이터에서 훨씬 안정적인 성능 평가 가능
#
#   3. 하이퍼파라미터 튜닝 (GridSearchCV)
#      - RF와 XGBoost의 최적 파라미터 자동 탐색
#      - 기본값보다 더 좋은 예측 성능 확보
#
# 작성자: 김홍근 / 하드웨어: MacBook Air M1
# ============================================================


# ──────────────────────────────────────────────────────────
# 라이브러리 불러오기
# ──────────────────────────────────────────────────────────
import os
import sys
import ast
import numpy as np
import pandas as pd
import joblib   # 학습된 모델을 파일로 저장/불러오기 위한 라이브러리
import warnings

# sklearn 내부에서 joblib 버전 불일치로 발생하는 UserWarning 억제
# (GridSearchCV 병렬 처리 시 수백 줄 반복 출력되는 무해한 경고)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
# matplotlib 한글 폰트 미설치 시 findfont/Glyph 경고 억제 (Windows 등)
warnings.filterwarnings('ignore', message='.*findfont.*')
warnings.filterwarnings('ignore', message='.*Glyph.*missing.*')

# 기계학습 관련 (scikit-learn)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# XGBoost: Gradient Boosting 기반 고성능 회귀 모델
import xgboost as xgb

# 시각화
import matplotlib
matplotlib.use('Agg')  # GUI 없이 파일로 저장
import matplotlib.pyplot as plt

# 한국어 폰트 설정 (Mac: AppleGothic, Windows: Malgun Gothic)
import platform
plt.rcParams['axes.unicode_minus'] = False
if platform.system() == 'Darwin':
    try:
        plt.rcParams['font.family'] = 'AppleGothic'
    except Exception:
        pass
elif platform.system() == 'Windows':
    try:
        plt.rcParams['font.family'] = 'Malgun Gothic'
    except Exception:
        try:
            plt.rcParams['font.family'] = 'NanumGothic'
        except Exception:
            pass


# ──────────────────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, 'data')
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, BASE_DIR)
try:
    from benchmark.support.hardware_info import get_hardware_info
except ImportError:
    get_hardware_info = None

# 단계별 서브폴더 경로
STAGE1_DIR  = os.path.join(DATA_DIR, 'stage1')   # ANN 실험 결과
STAGE2_DIR  = os.path.join(DATA_DIR, 'stage2')   # CNN 실험 결과
STAGE3_DIR  = os.path.join(DATA_DIR, 'stage3', 'v2')  # 예측 모델 결과 저장

STAGE4_DIR  = os.path.join(DATA_DIR, 'stage4')   # Transformer / GAN 실험 결과
STAGE3_V3_DIR = os.path.join(DATA_DIR, 'stage3', 'v3')  # v3 결과 저장 (4종 통합)

ANN_CSV         = os.path.join(STAGE1_DIR, 'ann_mnist_results.csv')
CNN_CSV         = os.path.join(STAGE2_DIR, 'cnn_cifar10_results.csv')
TRANSFORMER_CSV = os.path.join(STAGE4_DIR, 'transformer_results.csv')
GAN_CSV         = os.path.join(STAGE4_DIR, 'gan_results.csv')


# ──────────────────────────────────────────────────────────
# 하드웨어 정보 (MacBook Air M1 기준)
# ──────────────────────────────────────────────────────────
# 교수님 유의사항: 하드웨어 사양을 반드시 기록할 것
HARDWARE_INFO = {
    'cpu_cores': 8,           # M1 CPU 코어 수 (성능 4 + 효율 4)
    'cpu_freq_ghz': 3.2,      # 최대 클럭 속도 (GHz)
    'cpu_cache_l2_mb': 12.0,  # L2 캐시 크기 (MB)
    'cpu_cache_l3_mb': 0.0,   # M1은 별도 L3 없음 (SLC로 대체)
    'ram_total_gb': 8.0,      # 통합 메모리 (GB)
    'gpu_memory_gb': 8.0,     # M1 GPU 메모리 (CPU와 공유)
}

# ──────────────────────────────────────────────────────────
# CNN 모델 구조 정보 (ResNet18, MobileNetV2 수동 정의)
# ──────────────────────────────────────────────────────────
CNN_ARCHITECTURE_INFO = {
    'SimpleCNN': {
        'num_conv_layers': None,
        'base_channels': None,
        'has_pooling': 1,
        'has_batch_norm': 0,
        'cnn_num_fc_layers': 1,
        'cnn_kernel_size': 3,
        'cnn_has_residual': 0,
        'cnn_has_depthwise': 0,
        'cnn_stem_channels': None,
    },
    'ResNet18': {
        'num_conv_layers': 17,
        'base_channels': 64,
        'has_pooling': 1,
        'has_batch_norm': 1,
        'cnn_num_fc_layers': 1,
        'cnn_kernel_size': 3,
        'cnn_has_residual': 1,
        'cnn_has_depthwise': 0,
        'cnn_stem_channels': 64,
    },
    'MobileNetV2': {
        'num_conv_layers': 52,
        'base_channels': 32,
        'has_pooling': 1,
        'has_batch_norm': 1,
        'cnn_num_fc_layers': 1,
        'cnn_kernel_size': 3,
        'cnn_has_residual': 1,
        'cnn_has_depthwise': 1,
        'cnn_stem_channels': 32,
    },
}

# ──────────────────────────────────────────────────────────
# Feature 컬럼 목록 (통합 스키마 92개)
# ──────────────────────────────────────────────────────────
# 순서: (1) 공통 모델 구조 33개 → (2) 모델 전용 18개 → (3) 입력 데이터 8개 → (4) 하드웨어 33개
# 모델 타입별로 채우지 않는 피처는 0 또는 NaN → merge 후 fillna(0)
FEATURE_COLUMNS = [
    # ─── (1) 공통 모델 구조 33개 ─────────────────────────────
    'total_params',
    'log_total_params',
    'trainable_params',
    'model_size_mb',
    'log_model_size_mb',
    'total_layers',
    'num_hidden_layers',
    'num_linear_layers',
    'num_conv_layers',
    'max_width',
    'log_max_width',
    'min_width',
    'avg_width',
    'base_channels',
    'model_family_encoded',
    'has_pooling',
    'has_batch_norm',
    'cnn_num_fc_layers',
    'cnn_kernel_size',
    'flops',
    'has_residual',
    'has_depthwise',
    'has_attention',
    'has_cls_token',
    'num_blocks',
    'num_mult_adds',
    'activation_memory_mb',
    'first_layer_width',
    'last_layer_width',
    'is_sequential',
    'has_skip_connection',
    'max_channels',
    'min_channels',
    # ─── (2) 모델 전용 18개 (ANN / CNN / Transformer / GAN) ──
    'ann_max_hidden',
    'ann_min_hidden',
    'ann_avg_hidden',
    'cnn_num_filters',
    'cnn_max_channels',
    'cnn_has_residual',
    'cnn_has_depthwise',
    'embed_dim',
    'num_heads',
    'patch_size',
    'ffn_dim',
    'vit_has_cls_token',
    'latent_dim',
    'generator_params',
    'discriminator_params',
    'ann_num_layers',
    'cnn_stem_channels',
    'vit_num_encoder_layers',
    # ─── (3) 입력 데이터 8개 ────────────────────────────────
    'input_height',
    'input_width',
    'input_channels',
    'num_classes',
    'batch_size',
    'dataset_encoded',
    'input_pixels',
    'seq_length',
    # ─── (4) 하드웨어 33개 ─────────────────────────────────
    'device_type',
    'os_type',
    'accelerator_brand',
    'accelerator_name',
    'cpu_cores_physical',
    'cpu_cores_logical',
    'cpu_perf_cores',
    'cpu_efficiency_cores',
    'cpu_freq_base_ghz',
    'cpu_freq_boost_ghz',
    'cpu_cache_l2_mb',
    'cpu_cache_l3_mb',
    'ram_total_gb',
    'memory_type',
    'memory_bandwidth_gbs',
    'is_unified_memory',
    'shared_memory_gb',
    'dedicated_vram_gb',
    'gpu_count',
    'gpu_memory_gb',
    'gpu_core_count',
    'peak_bandwidth_gbs',
    'tflops_fp32',
    'tflops_fp16',
    'fp16_support',
    'bf16_support',
    'interconnect_type',
    'host_to_device_bandwidth_gbs',
    'is_discrete_gpu',
    'is_integrated_gpu',
    'device_encoded',
    'cpu_freq_ghz',
    'memory_channels',
]

TARGET_INFERENCE = 'inference_time_mean_ms'
TARGET_TRAINING  = 'training_time_mean_sec'

# ──────────────────────────────────────────────────────────
# 로그 변환 대상 Feature 목록
# ──────────────────────────────────────────────────────────
# 값의 범위가 매우 넓은 Feature에 log1p 변환 적용
# log1p(x) = log(1+x): x=0일 때도 안전하게 작동 (log(0) = -∞ 방지)
LOG_FEATURES = ['total_params', 'model_size_mb', 'max_width']


# ──────────────────────────────────────────────────────────
# 함수 정의
# ──────────────────────────────────────────────────────────

def parse_config(config_str):
    """
    ANN config 문자열 "[64, 128]" → 파이썬 리스트 [64, 128] 변환
    """
    try:
        return ast.literal_eval(config_str)
    except Exception:
        return [64]


def _fill_hardware_features(df, device_col='device'):
    """device 컬럼에 따라 하드웨어 피처 33개 채움. device_encoded: cpu=0, cuda/mps=1."""
    hw_keys = FEATURE_COLUMNS[59:]
    for device in df[device_col].dropna().unique():
        mask = df[device_col] == device
        d = str(device).strip().lower()
        hw = get_hardware_info(d) if get_hardware_info else {k: 0 for k in hw_keys}
        for key in hw_keys:
            df.loc[mask, key] = hw.get(key, 0)
    df['device_encoded'] = df[device_col].map({'cpu': 0, 'mps': 1, 'cuda': 1}).fillna(0)


def load_ann_data(csv_path):
    """
    ANN 실험 결과 CSV 로드 및 Feature 추출 함수

    기존 버전 대비 추가된 Feature:
    - log_total_params: log1p(total_params) - 파라미터 수 로그 변환
    - log_model_size_mb: log1p(model_size_mb) - 모델 크기 로그 변환
    - log_max_width: log1p(max_width) - 최대 너비 로그 변환

    이 log 변환 Feature들이 예측 모델의 성능 개선에 핵심 역할을 함

    Args:
        csv_path (str): ANN CSV 파일 경로

    Returns:
        pd.DataFrame: Feature가 추가된 데이터프레임
    """
    print(f"\n[ANN 데이터 로드] {os.path.basename(csv_path)}")
    df = pd.read_csv(csv_path)
    print(f"  행 수: {len(df)}개")

    # config 문자열 파싱
    df['config_list'] = df['config'].apply(parse_config)

    # (1) 공통 모델 구조
    df['max_width']         = df['config_list'].apply(max)
    df['min_width']         = df['config_list'].apply(min)
    df['avg_width']         = df['config_list'].apply(np.mean)
    df['num_hidden_layers'] = df['config_list'].apply(len)
    df['total_layers']      = df['num_hidden_layers']
    df['num_linear_layers'] = df['num_hidden_layers']
    df['num_conv_layers']   = 0
    df['base_channels']     = 0
    df['model_family_encoded'] = 0
    df['has_pooling']       = 0
    df['has_batch_norm']    = 0
    df['cnn_num_fc_layers'] = 1
    df['cnn_kernel_size']   = 0
    df['trainable_params'] = df['total_params']
    df['model_size_mb']     = df['total_params'] * 4 / (1024 * 1024)
    df['log_total_params'] = np.log1p(df['total_params'])
    df['log_model_size_mb'] = np.log1p(df['model_size_mb'])
    df['log_max_width']     = np.log1p(df['max_width'])
    df['flops'] = 0
    df['has_residual'] = 0
    df['has_depthwise'] = 0
    df['has_attention'] = 0
    df['has_cls_token'] = 0
    df['num_blocks'] = 0
    df['num_mult_adds'] = 0
    df['activation_memory_mb'] = 0
    df['first_layer_width'] = df['config_list'].apply(lambda x: x[0] if x else 0)
    df['last_layer_width']  = df['config_list'].apply(lambda x: x[-1] if x else 0)
    df['is_sequential'] = 1
    df['has_skip_connection'] = 0
    df['max_channels'] = 0
    df['min_channels'] = 0

    # (2) 모델 전용 — ANN
    df['ann_max_hidden'] = df['max_width']
    df['ann_min_hidden'] = df['min_width']
    df['ann_avg_hidden'] = df['avg_width']
    df['ann_num_layers'] = df['num_hidden_layers']
    df['cnn_num_filters'] = 0
    df['cnn_max_channels'] = 0
    df['cnn_has_residual'] = 0
    df['cnn_has_depthwise'] = 0
    df['embed_dim'] = 0
    df['num_heads'] = 0
    df['patch_size'] = 0
    df['ffn_dim'] = 0
    df['vit_has_cls_token'] = 0
    df['latent_dim'] = 0
    df['generator_params'] = 0
    df['discriminator_params'] = 0
    df['cnn_stem_channels'] = 0
    df['vit_num_encoder_layers'] = 0

    # (3) 입력 데이터
    df['input_height']   = 28
    df['input_width']    = 28
    df['input_channels'] = 1
    df['num_classes']    = 10
    df['batch_size']     = 64
    df['dataset_encoded'] = 0
    df['input_pixels']   = 28 * 28 * 1
    df['seq_length']     = 0

    # (4) 하드웨어
    _fill_hardware_features(df)

    df['dataset_name'] = 'MNIST'
    print(f"  Feature 수: {len(FEATURE_COLUMNS)}개 (92 feature)")
    return df


def load_cnn_data(csv_path):
    """
    CNN 실험 결과 CSV 로드 및 Feature 추출 함수

    기존 버전 대비 추가된 Feature:
    - log_total_params, log_model_size_mb, log_max_width (로그 변환)

    Args:
        csv_path (str): CNN CSV 파일 경로

    Returns:
        pd.DataFrame: Feature가 추가된 데이터프레임
    """
    print(f"\n[CNN 데이터 로드] {os.path.basename(csv_path)}")
    df = pd.read_csv(csv_path)
    print(f"  행 수: {len(df)}개")

    model_type_map = {'SimpleCNN': 1, 'ResNet18': 2, 'MobileNetV2': 3}
    df['model_family_encoded'] = df['model_type'].map(model_type_map)

    for model_name, info in CNN_ARCHITECTURE_INFO.items():
        mask = df['model_type'] == model_name
        for key in ['num_conv_layers', 'base_channels', 'cnn_stem_channels']:
            if info.get(key) is not None:
                df.loc[mask, key] = info[key]
        df.loc[mask, 'has_pooling'] = info['has_pooling']
        df.loc[mask, 'has_batch_norm'] = info['has_batch_norm']
        df.loc[mask, 'cnn_num_fc_layers'] = info['cnn_num_fc_layers']
        df.loc[mask, 'cnn_kernel_size'] = info['cnn_kernel_size']
        df.loc[mask, 'cnn_has_residual'] = info['cnn_has_residual']
        df.loc[mask, 'cnn_has_depthwise'] = info['cnn_has_depthwise']
        if info.get('cnn_stem_channels') is not None:
            df.loc[mask, 'cnn_stem_channels'] = info['cnn_stem_channels']

    df['num_conv_layers'] = df['num_conv_layers'].fillna(0).astype(int)
    df['base_channels']   = df['base_channels'].fillna(0).astype(int)
    df['cnn_stem_channels'] = df.get('cnn_stem_channels', pd.Series(0, index=df.index)).fillna(0)

    df['total_layers']      = df['num_conv_layers']
    df['num_hidden_layers'] = 0
    df['num_linear_layers'] = 0
    df['max_width']         = df['base_channels']
    df['min_width']         = df['base_channels']
    df['avg_width']         = df['base_channels'].astype(float)
    df['trainable_params']  = df['total_params']
    df['model_size_mb']     = df['total_params'] * 4 / (1024 * 1024)
    df['log_total_params']  = np.log1p(df['total_params'])
    df['log_model_size_mb'] = np.log1p(df['model_size_mb'])
    df['log_max_width']     = np.log1p(df['max_width'])
    df['flops'] = 0
    df['has_residual'] = df['cnn_has_residual']
    df['has_depthwise'] = df['cnn_has_depthwise']
    df['has_attention'] = 0
    df['has_cls_token'] = 0
    df['num_blocks'] = 0
    df['num_mult_adds'] = 0
    df['activation_memory_mb'] = 0
    df['first_layer_width'] = df['base_channels']
    df['last_layer_width']  = df['base_channels']
    df['is_sequential'] = 1
    df['has_skip_connection'] = df['cnn_has_residual']
    df['max_channels'] = df['base_channels']
    df['min_channels'] = df['base_channels']

    df['ann_max_hidden'] = 0
    df['ann_min_hidden'] = 0
    df['ann_avg_hidden'] = 0
    df['ann_num_layers'] = 0
    df['cnn_num_filters'] = df['base_channels']
    df['cnn_max_channels'] = df['base_channels']
    df['embed_dim'] = 0
    df['num_heads'] = 0
    df['patch_size'] = 0
    df['ffn_dim'] = 0
    df['vit_has_cls_token'] = 0
    df['latent_dim'] = 0
    df['generator_params'] = 0
    df['discriminator_params'] = 0
    df['vit_num_encoder_layers'] = 0

    df['input_height']   = 32
    df['input_width']    = 32
    df['input_channels'] = 3
    df['num_classes']    = 10
    df['batch_size']     = 64
    df['dataset_encoded'] = 1
    df['input_pixels']   = 32 * 32 * 3
    df['seq_length']     = 0

    _fill_hardware_features(df)
    df['dataset_name'] = 'CIFAR-10'
    print(f"  Feature 수: {len(FEATURE_COLUMNS)}개 (92 feature)")
    return df


def load_transformer_data(csv_path):
    """
    Transformer 실험 결과 CSV 로드 및 Feature 추출 함수

    Transformer(Vision Transformer, ViT)는 이미지를 패치로 나눠
    Self-Attention으로 처리하는 모델입니다.

    [Transformer 고유 Feature]
    - embed_dim: 임베딩 차원 (각 패치를 표현하는 벡터 크기)
    - num_heads: Multi-Head Attention의 헤드 수
    - patch_size: 이미지를 나누는 패치 크기 (예: 4 → 4×4 패치)

    [기존 Feature와의 매핑]
    - max_width / avg_width / min_width = embed_dim (Transformer의 주요 너비)
    - num_layers = num_layers (인코더 블록 수)
    - num_conv_layers = 0 (Convolution 없음)
    - model_type_encoded = 4 (신규: Transformer)

    Args:
        csv_path (str): transformer_results.csv 경로

    Returns:
        pd.DataFrame: Feature가 추출된 데이터프레임
    """
    print(f"\n[Transformer 데이터 로드] {os.path.basename(csv_path)}")
    df = pd.read_csv(csv_path)
    print(f"  행 수: {len(df)}개")

    df['embed_dim']  = df['embed_dim']
    df['num_heads']  = df['num_heads']
    df['patch_size'] = df['patch_size']

    df['max_width']         = df['embed_dim']
    df['min_width']         = df['embed_dim'].astype(float)
    df['avg_width']         = df['embed_dim'].astype(float)
    df['num_hidden_layers'] = df['num_layers']
    df['total_layers']      = df['num_layers']
    df['num_conv_layers']   = 0
    df['num_linear_layers'] = 0
    df['base_channels']     = 0
    df['model_family_encoded'] = 4
    df['has_pooling']       = 0
    df['has_batch_norm']    = 0
    df['cnn_num_fc_layers'] = 1
    df['cnn_kernel_size']   = 0
    df['trainable_params']  = df['total_params']
    df['model_size_mb']     = df['total_params'] * 4 / (1024 * 1024)
    df['log_total_params'] = np.log1p(df['total_params'])
    df['log_model_size_mb'] = np.log1p(df['model_size_mb'])
    df['log_max_width']     = np.log1p(df['max_width'])
    df['flops'] = 0
    df['has_residual'] = 0
    df['has_depthwise'] = 0
    df['has_attention'] = 1
    df['has_cls_token'] = 1
    df['num_blocks'] = df['num_layers']
    df['num_mult_adds'] = 0
    df['activation_memory_mb'] = 0
    df['first_layer_width'] = df['embed_dim']
    df['last_layer_width']  = df['embed_dim']
    df['is_sequential'] = 1
    df['has_skip_connection'] = 0
    df['max_channels'] = 0
    df['min_channels'] = 0

    df['ann_max_hidden'] = 0
    df['ann_min_hidden'] = 0
    df['ann_avg_hidden'] = 0
    df['ann_num_layers'] = 0
    df['cnn_num_filters'] = 0
    df['cnn_max_channels'] = 0
    df['cnn_has_residual'] = 0
    df['cnn_has_depthwise'] = 0
    df['ffn_dim'] = df['embed_dim'] * 4
    df['vit_has_cls_token'] = 1
    df['latent_dim'] = 0
    df['generator_params'] = 0
    df['discriminator_params'] = 0
    df['cnn_stem_channels'] = 0
    df['vit_num_encoder_layers'] = df['num_layers']

    df['input_height']   = df['img_size'] if 'img_size' in df.columns else 32
    df['input_width']    = df['img_size'] if 'img_size' in df.columns else 32
    df['input_channels'] = df['img_channels'] if 'img_channels' in df.columns else 3
    df['num_classes']    = 10
    df['batch_size']     = df['batch_size'] if 'batch_size' in df.columns else 64
    df['dataset_encoded'] = 1
    df['input_pixels']   = df['input_height'] * df['input_width'] * df['input_channels']
    df['seq_length']     = 0

    _fill_hardware_features(df)
    df['dataset_name'] = 'CIFAR-10'
    print(f"  Feature 수: {len(FEATURE_COLUMNS)}개 (92 feature)")
    return df


def load_gan_data(csv_path):
    """
    GAN 실험 결과 CSV 로드 및 Feature 추출 함수

    GAN(Generative Adversarial Network)은 Generator와 Discriminator
    두 신경망이 서로 경쟁하며 학습하는 구조입니다.

    [GAN 고유 Feature]
    - latent_dim: Generator 입력 노이즈 벡터 차원 (생성 다양성 제어)

    [기존 Feature와의 매핑]
    - max_width / min_width / avg_width = Generator 히든 레이어의 최대/최소/평균
    - num_layers = num_layers (Generator 히든 레이어 수)
    - num_conv_layers = 0 (FC 기반 GAN)
    - model_type_encoded = 5 (신규: GAN)

    Args:
        csv_path (str): gan_results.csv 경로

    Returns:
        pd.DataFrame: Feature가 추출된 데이터프레임
    """
    print(f"\n[GAN 데이터 로드] {os.path.basename(csv_path)}")
    df = pd.read_csv(csv_path)
    print(f"  행 수: {len(df)}개")

    # g_hidden_dims 컬럼 파싱: "[128, 256]" 문자열 → 파이썬 리스트
    def parse_dims(s):
        try:
            return ast.literal_eval(s)
        except Exception:
            return [128]

    df['g_dims_list'] = df['g_hidden_dims'].apply(parse_dims)

    df['max_width']         = df['g_dims_list'].apply(max)
    df['min_width']         = df['g_dims_list'].apply(min)
    df['avg_width']         = df['g_dims_list'].apply(np.mean)
    df['num_hidden_layers'] = df['num_layers']
    df['total_layers']      = df['num_layers']
    df['num_conv_layers']   = 0
    df['num_linear_layers'] = df['num_layers']
    df['base_channels']     = 0
    df['model_family_encoded'] = 5
    df['has_pooling']       = 0
    df['has_batch_norm']    = 0
    df['cnn_num_fc_layers'] = df['num_layers']
    df['cnn_kernel_size']   = 0
    df['trainable_params']  = df['total_params']
    df['model_size_mb']     = df['total_params'] * 4 / (1024 * 1024)
    df['log_total_params']  = np.log1p(df['total_params'])
    df['log_model_size_mb'] = np.log1p(df['model_size_mb'])
    df['log_max_width']     = np.log1p(df['max_width'])
    df['flops'] = 0
    df['has_residual'] = 0
    df['has_depthwise'] = 0
    df['has_attention'] = 0
    df['has_cls_token'] = 0
    df['num_blocks'] = 0
    df['num_mult_adds'] = 0
    df['activation_memory_mb'] = 0
    df['first_layer_width'] = df['g_dims_list'].apply(lambda x: x[0] if x else 0)
    df['last_layer_width']  = df['g_dims_list'].apply(lambda x: x[-1] if x else 0)
    df['is_sequential'] = 1
    df['has_skip_connection'] = 0
    df['max_channels'] = 0
    df['min_channels'] = 0

    df['ann_max_hidden'] = 0
    df['ann_min_hidden'] = 0
    df['ann_avg_hidden'] = 0
    df['ann_num_layers'] = 0
    df['cnn_num_filters'] = 0
    df['cnn_max_channels'] = 0
    df['cnn_has_residual'] = 0
    df['cnn_has_depthwise'] = 0
    df['embed_dim'] = 0
    df['num_heads'] = 0
    df['patch_size'] = 0
    df['ffn_dim'] = 0
    df['vit_has_cls_token'] = 0
    df['latent_dim'] = df['latent_dim']
    df['generator_params'] = (df['total_params'] // 2)
    df['discriminator_params'] = (df['total_params'] - df['generator_params'])
    df['cnn_stem_channels'] = 0
    df['vit_num_encoder_layers'] = 0

    df['input_height']   = df['img_size'] if 'img_size' in df.columns else 32
    df['input_width']    = df['img_size'] if 'img_size' in df.columns else 32
    df['input_channels'] = df['img_channels'] if 'img_channels' in df.columns else 1
    df['num_classes']    = 0
    df['batch_size']     = df['batch_size'] if 'batch_size' in df.columns else 64
    df['dataset_encoded'] = 0
    df['input_pixels']   = df['input_height'] * df['input_width'] * df['input_channels']
    df['seq_length']     = 0

    _fill_hardware_features(df)
    df['dataset_name'] = 'MNIST'
    print(f"  Feature 수: {len(FEATURE_COLUMNS)}개 (92 feature)")
    return df


def merge_data(ann_df, cnn_df, transformer_df, gan_df):
    """
    ANN / CNN / Transformer / GAN 데이터를 하나로 병합하는 함수

    v3 업데이트: 4종 모델 타입 통합 (총 190~250행 목표)
    - ANN: 80~100행, CNN: 50~70행, Transformer: 36~48행, GAN: 24~32행

    Args:
        ann_df (pd.DataFrame): ANN 데이터프레임
        cnn_df (pd.DataFrame): CNN 데이터프레임
        transformer_df (pd.DataFrame): Transformer 데이터프레임
        gan_df (pd.DataFrame): GAN 데이터프레임

    Returns:
        pd.DataFrame: 병합된 데이터프레임
    """
    print(f"\n[데이터 병합]")
    needed_cols = FEATURE_COLUMNS + [TARGET_INFERENCE, TARGET_TRAINING, 'device', 'dataset_name']
    merged = pd.concat(
        [
            ann_df[needed_cols],
            cnn_df[needed_cols],
            transformer_df[needed_cols],
            gan_df[needed_cols],
        ],
        ignore_index=True,
    )
    print(f"  ANN {len(ann_df)}행 + CNN {len(cnn_df)}행 "
          f"+ Transformer {len(transformer_df)}행 + GAN {len(gan_df)}행 = 총 {len(merged)}행")
    return merged


def evaluate_predictions(y_true_log, y_pred_log, model_name, target_name):
    """
    로그 공간에서 예측한 값을 원래 단위로 변환하여 성능 평가

    평가 방법:
    1. 예측값(log 공간) → expm1()으로 원래 단위 복원
    2. 원래 단위에서 RMSE, MAE, R² 계산

    이렇게 하면 교수님/연구 보고서에 실제 단위(ms, 초)로 결과를 보고할 수 있음

    Args:
        y_true_log: 실제값 (log 변환된)
        y_pred_log: 예측값 (log 변환된)
        model_name: 모델 이름
        target_name: 타겟 이름

    Returns:
        dict: 평가 지표 딕셔너리
    """
    # log 공간 → 원래 단위로 변환
    # expm1(x) = exp(x) - 1: log1p의 역변환
    y_true_orig = np.expm1(y_true_log)
    y_pred_orig = np.expm1(y_pred_log)

    # 원래 단위에서 성능 평가
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    mae  = mean_absolute_error(y_true_orig, y_pred_orig)
    r2   = r2_score(y_true_orig, y_pred_orig)

    # log 공간에서도 R² 계산 (학습 품질 참고용)
    r2_log = r2_score(y_true_log, y_pred_log)

    print(f"\n    [{model_name}] {target_name}:")
    print(f"      RMSE (원래 단위) : {rmse:.4f}  (작을수록 좋음)")
    print(f"      MAE  (원래 단위) : {mae:.4f}  (작을수록 좋음)")
    print(f"      R²   (원래 단위) : {r2:.4f}  (1에 가까울수록 좋음)")
    print(f"      R²   (log 공간)  : {r2_log:.4f}  (학습 품질 참고용)")

    return {
        'model_name': model_name,
        'target': target_name,
        'RMSE': round(rmse, 4),
        'MAE':  round(mae, 4),
        'R2':   round(r2, 4),
        'R2_log': round(r2_log, 4),
    }


def tune_and_evaluate(X, y_log, model_type, target_name):
    """
    GridSearchCV로 하이퍼파라미터를 튜닝하고 K-Fold 교차검증으로 성능 평가

    왜 K-Fold를 사용하나?
    - 데이터가 50~70개로 매우 적음
    - 단순 train/test 분리 시 10개짜리 테스트 1번으로는 결과가 불안정
    - 5-Fold: 데이터를 5등분하여 5번 반복 → 평균 성능이 훨씬 신뢰성 있음

    왜 GridSearchCV를 사용하나?
    - 기본 파라미터(n_estimators=100 등)가 최적이 아닐 수 있음
    - 여러 파라미터 조합을 자동으로 탐색하여 최적값 찾음

    Args:
        X: Feature 행렬
        y_log: log 변환된 타겟 벡터
        model_type: 'rf' (Random Forest) 또는 'xgb' (XGBoost)
        target_name: 타겟 이름 (결과 출력용)

    Returns:
        tuple: (최적 모델, 교차검증 예측값 배열, 최적 파라미터 딕셔너리)
    """
    print(f"\n  {'Random Forest' if model_type == 'rf' else 'XGBoost'} 하이퍼파라미터 튜닝 중...")

    # K-Fold 설정: 5등분, shuffle=True로 데이터 순서 섞음
    # random_state=42: 재현 가능한 결과를 위해 고정
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    if model_type == 'rf':
        # Random Forest 하이퍼파라미터 탐색 공간
        base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [50, 100, 200],      # 결정 트리 개수
            'max_depth': [None, 5, 10, 20],       # 트리 최대 깊이 (None=제한 없음)
            'min_samples_split': [2, 5, 10],      # 노드 분할을 위한 최소 샘플 수
            'min_samples_leaf': [1, 2, 4],        # 리프 노드의 최소 샘플 수
        }
        model_name = "Random Forest"
    else:
        # XGBoost 하이퍼파라미터 탐색 공간
        base_model = xgb.XGBRegressor(random_state=42, verbosity=0)
        param_grid = {
            'n_estimators': [50, 100, 200],       # 부스팅 라운드 수
            'learning_rate': [0.05, 0.1, 0.2],    # 학습률
            'max_depth': [3, 4, 6],               # 트리 최대 깊이
            'subsample': [0.8, 1.0],              # 각 트리에 사용할 데이터 비율
        }
        model_name = "XGBoost"

    # GridSearchCV: 모든 파라미터 조합을 시도하여 최적값 탐색
    # cv=kf: K-Fold로 교차검증
    # scoring='r2': R²를 기준으로 최적 파라미터 선택
    # n_jobs=-1: 모든 CPU 코어 사용 (병렬 처리)
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=kf,
        scoring='r2',
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X, y_log)

    best_model  = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_r2  = grid_search.best_score_

    print(f"    최적 파라미터: {best_params}")
    print(f"    K-Fold CV R² (log 공간): {best_cv_r2:.4f}")

    # 최적 파라미터로 K-Fold 교차검증 예측값 수집
    # cross_val_predict: 각 fold에서 예측한 값을 모아서 반환
    from sklearn.model_selection import cross_val_predict
    y_pred_log_cv = cross_val_predict(best_model, X, y_log, cv=kf)

    return best_model, y_pred_log_cv, best_params, best_cv_r2, model_name


def plot_feature_importance(model, feature_names, model_name, target_name, save_path):
    """
    Feature 중요도를 막대 그래프로 시각화하고 저장

    Feature 중요도: 해당 Feature가 예측에 얼마나 기여하는지 (0~1)
    값이 클수록 그 Feature가 예측에 더 중요한 역할을 함
    """
    importances = model.feature_importances_
    indices     = np.argsort(importances)
    sorted_f    = [feature_names[i] for i in indices]
    sorted_imp  = importances[indices]

    plt.figure(figsize=(12, 9))
    plt.barh(range(len(sorted_f)), sorted_imp, color='steelblue', alpha=0.8)
    plt.yticks(range(len(sorted_f)), sorted_f, fontsize=8)
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title(f'{model_name} Feature Importance\n({target_name} prediction)', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    → 저장: {os.path.basename(save_path)}")


def plot_pred_vs_actual(y_true_log, y_pred_log, model_name, target_name, save_path):
    """
    예측값 vs 실제값 산점도 그래프 저장

    점이 빨간 대각선(y=x)에 가까울수록 예측이 정확함
    log 공간과 원래 단위 두 가지 그래프를 나란히 표시
    """
    y_true_orig = np.expm1(y_true_log)
    y_pred_orig = np.expm1(y_pred_log)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, y_true, y_pred, space in zip(
        axes,
        [y_true_log, y_true_orig],
        [y_pred_log, y_pred_orig],
        ['log 공간', '원래 단위']
    ):
        ax.scatter(y_true, y_pred, color='steelblue', alpha=0.75,
                   edgecolors='white', linewidths=0.5, s=70)
        all_vals = list(y_true) + list(y_pred)
        mn, mx = min(all_vals), max(all_vals)
        ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5, label='Perfect (y=x)')
        ax.set_xlabel(f'Actual ({space})', fontsize=10)
        ax.set_ylabel(f'Predicted ({space})', fontsize=10)
        ax.set_title(f'{space}', fontsize=11)
        ax.legend(fontsize=8)

    fig.suptitle(f'{model_name}: Predicted vs Actual\n({target_name})', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    → 저장: {os.path.basename(save_path)}")


# ──────────────────────────────────────────────────────────
# 메인 실행 함수
# ──────────────────────────────────────────────────────────

def main():
    """
    전체 파이프라인 실행 (개선판)

    실행 순서:
    1. ANN / CNN 데이터 로드 (로그 변환 Feature 포함)
    2. 데이터 병합
    3. 타겟 값 로그 변환 (핵심 개선!)
    4. RF / XGBoost 하이퍼파라미터 튜닝 (GridSearchCV)
    5. K-Fold 교차검증으로 성능 평가
    6. 결과 시각화 및 저장
    """

    print("=" * 70)
    print("  3단계 v3: DNN 실행 시간 예측 모델 (Transformer/GAN 통합)")
    print("  개선: 로그 변환 + K-Fold(5) + GridSearchCV + 4종 모델 통합")
    print("=" * 70)

    # ── 1. 데이터 로드 (4종 모델) ──────────────────────────
    ann_df         = load_ann_data(ANN_CSV)
    cnn_df         = load_cnn_data(CNN_CSV)
    transformer_df = load_transformer_data(TRANSFORMER_CSV)
    gan_df         = load_gan_data(GAN_CSV)

    # ── 2. 데이터 병합 ─────────────────────────────────────
    merged_df = merge_data(ann_df, cnn_df, transformer_df, gan_df)

    # NaN 처리
    nan_count = merged_df[FEATURE_COLUMNS].isna().sum().sum()
    if nan_count > 0:
        print(f"\n  [주의] NaN {nan_count}개 → 0으로 대체")
        merged_df[FEATURE_COLUMNS] = merged_df[FEATURE_COLUMNS].fillna(0)

    print(f"\n[데이터 요약]")
    print(f"  전체 데이터     : {len(merged_df)}개")
    print(f"  CPU 데이터      : {(merged_df['device_encoded'] == 0).sum()}개")
    print(f"  MPS 데이터      : {(merged_df['device_encoded'] == 1).sum()}개")
    print(f"  ANN 데이터      : {(merged_df['model_family_encoded'] == 0).sum()}개")
    print(f"  CNN 데이터      : {(merged_df['model_family_encoded'].isin([1,2,3])).sum()}개")
    print(f"  Transformer 데이터: {(merged_df['model_family_encoded'] == 4).sum()}개")
    print(f"  GAN 데이터      : {(merged_df['model_family_encoded'] == 5).sum()}개")
    print(f"  Feature 수      : {len(FEATURE_COLUMNS)}개")

    # ── 3. Feature(X)와 타겟(y) 분리 + 로그 변환 ──────────
    X = merged_df[FEATURE_COLUMNS].values

    # ★ 핵심 개선: 타겟에 log1p 변환 적용
    # 원래 값의 범위가 너무 넓어 그대로는 회귀가 어려움
    # log1p로 변환하면 값의 범위가 균일해져 학습이 훨씬 쉬워짐
    #
    # 예시 (추론 시간):
    #   원래:    0.03ms → 799ms   (26,000배 차이)
    #   log1p 후: 0.03  → 6.68   (약 220배 차이 → 훨씬 관리하기 쉬운 범위)
    y_inf_log = np.log1p(merged_df[TARGET_INFERENCE].values)
    y_tr_log  = np.log1p(merged_df[TARGET_TRAINING].values)

    print(f"\n[타겟 값 분포 (로그 변환 전 → 후)]")
    print(f"  추론 시간: {merged_df[TARGET_INFERENCE].min():.3f} ~ {merged_df[TARGET_INFERENCE].max():.1f} ms")
    print(f"  → log 변환: {y_inf_log.min():.3f} ~ {y_inf_log.max():.3f}")
    print(f"  학습 시간: {merged_df[TARGET_TRAINING].min():.1f} ~ {merged_df[TARGET_TRAINING].max():.1f} sec")
    print(f"  → log 변환: {y_tr_log.min():.3f} ~ {y_tr_log.max():.3f}")

    # ── 4. 하이퍼파라미터 튜닝 + K-Fold 교차검증 ──────────
    print(f"\n{'=' * 70}")
    print(f"  [하이퍼파라미터 튜닝 + K-Fold(5) 교차검증]")
    print(f"  GridSearchCV로 최적 파라미터 탐색 → 시간이 걸릴 수 있습니다...")
    print(f"{'=' * 70}")

    results = []

    # Random Forest - 추론 시간
    print(f"\n  ▶ Random Forest × 추론 시간")
    rf_inf_model, rf_inf_pred, rf_inf_params, rf_inf_cv_r2, _ = tune_and_evaluate(
        X, y_inf_log, 'rf', '추론 시간'
    )
    r = evaluate_predictions(y_inf_log, rf_inf_pred, "Random Forest (K-Fold CV)", "추론 시간(ms)")
    r['best_params'] = str(rf_inf_params)
    r['cv_r2_log'] = round(rf_inf_cv_r2, 4)
    results.append(r)

    # XGBoost - 추론 시간
    print(f"\n  ▶ XGBoost × 추론 시간")
    xgb_inf_model, xgb_inf_pred, xgb_inf_params, xgb_inf_cv_r2, _ = tune_and_evaluate(
        X, y_inf_log, 'xgb', '추론 시간'
    )
    r = evaluate_predictions(y_inf_log, xgb_inf_pred, "XGBoost (K-Fold CV)", "추론 시간(ms)")
    r['best_params'] = str(xgb_inf_params)
    r['cv_r2_log'] = round(xgb_inf_cv_r2, 4)
    results.append(r)

    # Random Forest - 학습 시간
    print(f"\n  ▶ Random Forest × 학습 시간")
    rf_tr_model, rf_tr_pred, rf_tr_params, rf_tr_cv_r2, _ = tune_and_evaluate(
        X, y_tr_log, 'rf', '학습 시간'
    )
    r = evaluate_predictions(y_tr_log, rf_tr_pred, "Random Forest (K-Fold CV)", "학습 시간(sec)")
    r['best_params'] = str(rf_tr_params)
    r['cv_r2_log'] = round(rf_tr_cv_r2, 4)
    results.append(r)

    # XGBoost - 학습 시간
    print(f"\n  ▶ XGBoost × 학습 시간")
    xgb_tr_model, xgb_tr_pred, xgb_tr_params, xgb_tr_cv_r2, _ = tune_and_evaluate(
        X, y_tr_log, 'xgb', '학습 시간'
    )
    r = evaluate_predictions(y_tr_log, xgb_tr_pred, "XGBoost (K-Fold CV)", "학습 시간(sec)")
    r['best_params'] = str(xgb_tr_params)
    r['cv_r2_log'] = round(xgb_tr_cv_r2, 4)
    results.append(r)

    # ── 5. 성능 비교표 출력 ────────────────────────────────
    results_df = pd.DataFrame(results)
    print(f"\n{'=' * 70}")
    print("[최종 성능 비교표]")
    print(f"{'=' * 70}")
    display_cols = ['model_name', 'target', 'RMSE', 'MAE', 'R2', 'R2_log', 'cv_r2_log']
    print(results_df[display_cols].to_string(index=False))

    print(f"\n[개선 포인트]")
    print(f"  - R2_log: log 공간에서의 R² (학습 품질 직접 지표)")
    print(f"  - cv_r2_log: K-Fold CV R² in log space (GridSearchCV 기준)")
    print(f"  - R2: 원래 단위(ms, sec)에서의 R² (실제 예측 성능)")

    # ── 6. 시각화 저장 ─────────────────────────────────────
    # stage3/v3 폴더가 없으면 생성 (v3: Transformer/GAN 통합 버전)
    os.makedirs(STAGE3_V3_DIR, exist_ok=True)

    print(f"\n{'=' * 70}")
    print("[시각화 저장 → data/stage3/v3/ 폴더]")
    print(f"{'=' * 70}")

    print(f"\n  Feature 중요도 그래프:")
    plot_feature_importance(rf_inf_model,  FEATURE_COLUMNS, "RandomForest", "Inference Time",
                            os.path.join(STAGE3_V3_DIR, "stage3_v3_rf_feature_importance_inference.png"))
    plot_feature_importance(xgb_inf_model, FEATURE_COLUMNS, "XGBoost",      "Inference Time",
                            os.path.join(STAGE3_V3_DIR, "stage3_v3_xgb_feature_importance_inference.png"))
    plot_feature_importance(rf_tr_model,   FEATURE_COLUMNS, "RandomForest", "Training Time",
                            os.path.join(STAGE3_V3_DIR, "stage3_v3_rf_feature_importance_training.png"))
    plot_feature_importance(xgb_tr_model,  FEATURE_COLUMNS, "XGBoost",      "Training Time",
                            os.path.join(STAGE3_V3_DIR, "stage3_v3_xgb_feature_importance_training.png"))

    print(f"\n  예측 vs 실제 산점도 (log 공간 + 원래 단위 동시 표시):")
    plot_pred_vs_actual(y_inf_log, rf_inf_pred,  "RandomForest", "Inference Time (ms)",
                        os.path.join(STAGE3_V3_DIR, "stage3_v3_rf_pred_vs_actual_inference.png"))
    plot_pred_vs_actual(y_inf_log, xgb_inf_pred, "XGBoost",      "Inference Time (ms)",
                        os.path.join(STAGE3_V3_DIR, "stage3_v3_xgb_pred_vs_actual_inference.png"))
    plot_pred_vs_actual(y_tr_log,  rf_tr_pred,   "RandomForest", "Training Time (sec)",
                        os.path.join(STAGE3_V3_DIR, "stage3_v3_rf_pred_vs_actual_training.png"))
    plot_pred_vs_actual(y_tr_log,  xgb_tr_pred,  "XGBoost",      "Training Time (sec)",
                        os.path.join(STAGE3_V3_DIR, "stage3_v3_xgb_pred_vs_actual_training.png"))

    # ── 7. 결과 저장 ───────────────────────────────────────
    results_path = os.path.join(STAGE3_V3_DIR, "stage3_v3_prediction_results.csv")
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')

    merged_path = os.path.join(STAGE3_V3_DIR, "stage3_v3_merged_features.csv")
    merged_df.to_csv(merged_path, index=False, encoding='utf-8-sig')

    # ── 8. 학습된 모델 joblib으로 저장 (Stage 5 ONNX 예측에서 사용) ───────
    # joblib: 파이썬 객체(모델, 배열 등)를 파일로 저장하고 불러오는 도구
    # 저장된 모델은 predict_from_onnx.py에서 불러와서 새 모델 실행 시간 예측에 사용됨
    MODEL_DIR = os.path.join(BASE_DIR, 'models', 'trained')
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 추론 시간 예측 모델 저장 (XGBoost + RandomForest)
    joblib.dump(xgb_inf_model,  os.path.join(MODEL_DIR, 'xgb_inference.pkl'))
    joblib.dump(xgb_tr_model,   os.path.join(MODEL_DIR, 'xgb_training.pkl'))
    joblib.dump(rf_inf_model,   os.path.join(MODEL_DIR, 'rf_inference.pkl'))
    joblib.dump(rf_tr_model,    os.path.join(MODEL_DIR, 'rf_training.pkl'))

    # Feature 컬럼 목록 저장 (predict_from_onnx.py에서 Feature 순서 맞추는 데 필요)
    joblib.dump(FEATURE_COLUMNS, os.path.join(MODEL_DIR, 'feature_columns.pkl'))

    print(f"\n{'=' * 70}")
    print(f"  3단계 v3 완료! (ANN+CNN+Transformer+GAN 통합)")
    print(f"  결과: {results_path}")
    print(f"  Feature 데이터: {merged_path}")
    print(f"  그래프: data/stage3/v3/stage3_v3_*.png (8개)")
    print(f"\n  [Stage 5용 모델 저장 완료]")
    print(f"  → {MODEL_DIR}/xgb_inference.pkl  (추론 시간 XGBoost)")
    print(f"  → {MODEL_DIR}/xgb_training.pkl   (학습 시간 XGBoost)")
    print(f"  → {MODEL_DIR}/rf_inference.pkl   (추론 시간 RandomForest)")
    print(f"  → {MODEL_DIR}/rf_training.pkl    (학습 시간 RandomForest)")
    print(f"  → {MODEL_DIR}/feature_columns.pkl (Feature 컬럼 순서)")
    print(f"{'=' * 70}\n")


if __name__ == '__main__':
    main()
