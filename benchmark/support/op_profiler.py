# ============================================================
# Op-Level 프로파일러: 모델을 연산 단위로 분해하여 개별 시간/메모리 측정
# ============================================================
#
# 출처: ijunsoo 브랜치 (benchmark/features/op_profiler.py) 포팅
# 수정: MPS(Apple Silicon) 지원 추가, 한국어 주석 보강
#
# 교수님 요구사항: "모델을 작은 연산 단위로 나누고 예측"
#   → Conv, Linear, BN, Pool, Activation 등 각 op의 시간을 개별 측정
#   → 합산하여 전체 시간 시뮬레이션 (bottom-up 예측)
#
# 기존 방식 (end-to-end, 방법 1):
#     전체 모델 → 33개 통계 피처 → ML(XGBoost/RF) → 전체 시간 예측
#
# 시뮬레이션 방식 (op-level, 방법 2):
#     모델 → [op1, op2, op3, ...] 분해
#     → 각 op의 (타입, 파라미터, FLOPs, 입출력 크기) 추출
#     → 각 op 시간 개별 측정 → 합산 = 전체 시간 예측
#
# 작성자: ijunsoo (원본) / 김홍근 (MPS 지원 포팅)
# ============================================================

import time
import torch
import torch.nn as nn
import numpy as np


# ──────────────────────────────────────────────────────────
# 지원하는 연산 타입 (Op = Operation, 연산 단위)
# ──────────────────────────────────────────────────────────
# Conv2d: 합성곱, Linear: 행렬곱, BatchNorm: 정규화, Pool: 풀링, ReLU 등: 활성화
OP_TYPES = {
    'Conv2d': nn.Conv2d,
    'Linear': nn.Linear,
    'BatchNorm2d': nn.BatchNorm2d,
    'BatchNorm1d': nn.BatchNorm1d,
    'MaxPool2d': nn.MaxPool2d,
    'AvgPool2d': nn.AvgPool2d,
    'AdaptiveAvgPool2d': nn.AdaptiveAvgPool2d,
    'ReLU': nn.ReLU,
    'ReLU6': nn.ReLU6,
    'GELU': nn.GELU,
    'LeakyReLU': nn.LeakyReLU,
    'LayerNorm': nn.LayerNorm,
    'Dropout': nn.Dropout,
    'Sigmoid': nn.Sigmoid,
    'Tanh': nn.Tanh,
}


def _synchronize_device(device):
    """디바이스별 동기화: GPU 연산 완료 대기 (시간 측정 정확도 향상)"""
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        # Apple Silicon MPS: 비동기 실행이므로 완료 대기 필요
        if torch.backends.mps.is_available():
            torch.mps.synchronize()


def decompose_model(model, input_shape=(1, 1, 28, 28)):
    """모델을 개별 연산(op) 단위로 분해

    Forward hook으로 각 op의 입출력 shape, FLOPs, 파라미터 수를 기록합니다.
    hook: PyTorch가 각 레이어를 실행할 때 자동으로 호출되는 콜백 함수.

    Args:
        model: PyTorch 모델 (nn.Module)
        input_shape: 입력 텐서 shape. 예: (배치, 채널, 높이, 너비) = (1, 1, 28, 28)

    Returns:
        list[dict]: 각 op의 상세 정보
            - name: 모듈 경로명
            - op_type: 연산 타입 (Conv2d, Linear, ...)
            - params: 파라미터 수
            - flops: 추정 FLOPs (Floating Point Operations, 연산 횟수)
            - input_shape, output_shape: 입출력 텐서 크기
            - memory_read, memory_write: 메모리 접근량 (바이트)
            - weight_bytes: 가중치 크기 (바이트)
    """
    ops = []
    hooks = []

    def make_hook(name, module):
        def hook_fn(mod, inp, out):
            # 입출력 shape 추출 (tuple로 감싸진 경우 첫 번째 요소 사용)
            if isinstance(inp, tuple) and len(inp) > 0:
                in_tensor = inp[0]
            else:
                in_tensor = inp

            if isinstance(out, tuple) and len(out) > 0:
                out_tensor = out[0]
            else:
                out_tensor = out

            in_shape = tuple(in_tensor.shape) if hasattr(in_tensor, 'shape') else ()
            out_shape = tuple(out_tensor.shape) if hasattr(out_tensor, 'shape') else ()

            # 파라미터 수 (가중치 + 편향)
            params = sum(p.numel() for p in mod.parameters())

            # FLOPs 추정 (연산 횟수)
            flops = _calc_op_flops(mod, in_tensor, out_tensor)

            # 메모리 접근량 추정 (float32 = 4바이트)
            in_bytes = in_tensor.numel() * 4 if hasattr(in_tensor, 'numel') else 0
            out_bytes = out_tensor.numel() * 4 if hasattr(out_tensor, 'numel') else 0
            weight_bytes = params * 4

            op_type = type(mod).__name__

            ops.append({
                'name': name,
                'op_type': op_type,
                'params': params,
                'flops': flops,
                'input_shape': in_shape,
                'output_shape': out_shape,
                'memory_read': in_bytes + weight_bytes,  # 입력 + 가중치 읽기
                'memory_write': out_bytes,                # 출력 쓰기
                'weight_bytes': weight_bytes,
            })
        return hook_fn

    # 리프 모듈에만 hook 등록 (중복 방지: 자식 모듈이 없는 최하위 레이어만)
    for name, module in model.named_modules():
        is_leaf = len(list(module.children())) == 0
        is_tracked = isinstance(module, tuple(OP_TYPES.values()))
        if is_leaf and is_tracked:
            hooks.append(module.register_forward_hook(make_hook(name, module)))

    # Forward pass로 hook 실행 (더미 입력으로 한 번 돌려서 정보 수집)
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(*input_shape)
        try:
            model(dummy)
        except Exception:
            pass

    for h in hooks:
        h.remove()

    return ops


def _calc_op_flops(module, input_tensor, output_tensor):
    """개별 op의 FLOPs 계산 (Floating Point Operations, 부동소수점 연산 횟수)

    Conv2d, Linear 등 연산 타입별로 대략적인 연산 수를 추정합니다.
    Attention의 Q@K, attn@V matmul은 nn.Linear로 잡힙니다.
    """
    if isinstance(module, nn.Conv2d):
        out_h, out_w = output_tensor.size(2), output_tensor.size(3)
        k = module.kernel_size[0] * module.kernel_size[1]
        return 2 * module.in_channels * module.out_channels * k * out_h * out_w // module.groups

    elif isinstance(module, nn.Linear):
        return 2 * module.in_features * module.out_features

    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
        return 2 * input_tensor.numel()

    elif isinstance(module, nn.LayerNorm):
        return 5 * input_tensor.numel()

    elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.GELU, nn.LeakyReLU,
                             nn.Sigmoid, nn.Tanh)):
        return input_tensor.numel()

    elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
        return output_tensor.numel()

    return 0


def measure_op_times(model, input_shape=(1, 1, 28, 28), device='cpu',
                     warmup=3, repeats=10):
    """각 op의 실제 실행 시간을 개별 측정

    워밍업 후 반복 측정하여 평균/표준편차를 계산합니다.
    CPU, CUDA, MPS(Apple Silicon) 모두 지원합니다.

    Args:
        model: PyTorch 모델
        input_shape: 입력 shape
        device: 'cpu', 'cuda', 'mps' 중 하나
        warmup: 워밍업 횟수 (초기화 비용 제거)
        repeats: 반복 측정 횟수

    Returns:
        list[dict]: 각 op 정보 + avg_time_ms, std_time_ms
    """
    device = torch.device(device) if isinstance(device, str) else device
    model = model.to(device).eval()

    # 먼저 op 분해 (CPU에서, input_shape 기준)
    model_cpu = model.cpu()
    ops = decompose_model(model_cpu, input_shape)
    model = model.to(device)

    # 각 op에 시간 측정 hook 부착
    time_records = {}
    time_hooks = []

    leaf_modules = []
    for name, module in model.named_modules():
        is_leaf = len(list(module.children())) == 0
        is_tracked = isinstance(module, tuple(OP_TYPES.values()))
        if is_leaf and is_tracked:
            leaf_modules.append((name, module))

    # 각 리프 모듈에 pre_hook(시작 시간) + post_hook(종료 시간) 등록
    for idx, (name, module) in enumerate(leaf_modules):
        time_records[idx] = {'start': 0, 'times': []}

        def make_pre_hook(i):
            def pre_hook(mod, inp):
                _synchronize_device(device)
                time_records[i]['start'] = time.perf_counter()
            return pre_hook

        def make_post_hook(i):
            def post_hook(mod, inp, out):
                _synchronize_device(device)
                elapsed = time.perf_counter() - time_records[i]['start']
                time_records[i]['times'].append(elapsed)
            return post_hook

        time_hooks.append(module.register_forward_pre_hook(make_pre_hook(idx)))
        time_hooks.append(module.register_forward_hook(make_post_hook(idx)))

    # 워밍업 (캐시, JIT 컴파일 등 초기화)
    dummy = torch.zeros(*input_shape, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)

    # 워밍업 기록 초기화
    for idx in time_records:
        time_records[idx]['times'] = []

    # 반복 측정
    with torch.no_grad():
        for _ in range(repeats):
            model(dummy)

    # hook 제거
    for h in time_hooks:
        h.remove()

    # 결과 병합: ops에 avg_time_ms, std_time_ms 추가
    for idx in range(min(len(ops), len(time_records))):
        times = time_records[idx]['times']
        if times:
            ops[idx]['avg_time_ms'] = round(float(np.mean(times)) * 1000, 6)
            ops[idx]['std_time_ms'] = round(float(np.std(times)) * 1000, 6)
        else:
            ops[idx]['avg_time_ms'] = 0.0
            ops[idx]['std_time_ms'] = 0.0

    return ops


def simulate_total_time(ops):
    """op별 시간을 합산하여 전체 시간 시뮬레이션

    Returns:
        dict: 시뮬레이션 결과
            - total_time_ms: op 시간 합산 (ms)
            - total_flops: 전체 FLOPs
            - total_memory_bytes: 전체 메모리 접근량
            - op_summary: op 타입별 시간/비율
    """
    total_time = sum(op.get('avg_time_ms', 0) for op in ops)
    total_flops = sum(op['flops'] for op in ops)
    total_memory = sum(op['memory_read'] + op['memory_write'] for op in ops)

    # op 타입별 집계 (Conv2d, Linear 등)
    by_type = {}
    for op in ops:
        t = op['op_type']
        if t not in by_type:
            by_type[t] = {'count': 0, 'time_ms': 0, 'flops': 0, 'params': 0}
        by_type[t]['count'] += 1
        by_type[t]['time_ms'] += op.get('avg_time_ms', 0)
        by_type[t]['flops'] += op['flops']
        by_type[t]['params'] += op['params']

    # 비율 계산
    for t in by_type:
        if total_time > 0:
            by_type[t]['time_pct'] = round(by_type[t]['time_ms'] / total_time * 100, 1)
        else:
            by_type[t]['time_pct'] = 0

    return {
        'total_time_ms': round(total_time, 4),
        'total_flops': total_flops,
        'total_memory_bytes': total_memory,
        'num_ops': len(ops),
        'op_summary': by_type,
    }


def get_op_level_features(ops):
    """op 분해 결과에서 피처 벡터 생성 (예측 모델 입력용)

    op-level 집계 통계를 딕셔너리로 반환합니다.
    향후 op 기반 시간 예측 모델 학습에 사용할 수 있습니다.

    Returns:
        dict: op-level 집계 피처
    """
    features = {
        'num_ops': len(ops),
        'total_op_flops': sum(op['flops'] for op in ops),
        'total_op_memory_read': sum(op['memory_read'] for op in ops),
        'total_op_memory_write': sum(op['memory_write'] for op in ops),
        'total_weight_bytes': sum(op['weight_bytes'] for op in ops),
    }

    # op 타입별 FLOPs 비율
    type_flops = {}
    for op in ops:
        t = op['op_type']
        type_flops[t] = type_flops.get(t, 0) + op['flops']

    total_f = features['total_op_flops']
    for op_type in ['Conv2d', 'Linear', 'BatchNorm2d', 'LayerNorm',
                    'MaxPool2d', 'ReLU', 'GELU']:
        key = f'flops_ratio_{op_type}'
        if total_f > 0:
            features[key] = round(type_flops.get(op_type, 0) / total_f, 4)
        else:
            features[key] = 0.0

    # 최대/평균 op FLOPs
    op_flops_list = [op['flops'] for op in ops if op['flops'] > 0]
    if op_flops_list:
        features['max_op_flops'] = max(op_flops_list)
        features['avg_op_flops'] = round(np.mean(op_flops_list), 0)
        features['std_op_flops'] = round(np.std(op_flops_list), 0)
    else:
        features['max_op_flops'] = 0
        features['avg_op_flops'] = 0
        features['std_op_flops'] = 0

    return features


def print_op_profile(ops, top_n=15):
    """op 프로파일 결과를 콘솔에 출력"""
    sim = simulate_total_time(ops)

    print(f"\n  Op-Level 프로파일 ({sim['num_ops']}개 연산)")
    print(f"  {'='*70}")
    print(f"  {'Op Type':<20s} | {'개수':>4s} | {'시간(ms)':>10s} | "
          f"{'비율':>6s} | {'FLOPs':>12s} | {'파라미터':>10s}")
    print(f"  {'-'*20} | {'-'*4} | {'-'*10} | {'-'*6} | {'-'*12} | {'-'*10}")

    for t, info in sorted(sim['op_summary'].items(),
                          key=lambda x: x[1]['time_ms'], reverse=True):
        print(f"  {t:<20s} | {info['count']:>4d} | "
              f"{info['time_ms']:>10.4f} | {info['time_pct']:>5.1f}% | "
              f"{info['flops']:>12,} | {info['params']:>10,}")

    print(f"  {'-'*70}")
    print(f"  {'합계':<20s} |      | {sim['total_time_ms']:>10.4f} | "
          f"100.0% | {sim['total_flops']:>12,} | ")
    print(f"  총 메모리 접근: {sim['total_memory_bytes']:,} bytes")

    # 개별 op 상위 출력
    if any(op.get('avg_time_ms', 0) > 0 for op in ops):
        print(f"\n  개별 Op 상위 {top_n}개 (시간 순):")
        sorted_ops = sorted(ops, key=lambda x: x.get('avg_time_ms', 0),
                            reverse=True)
        for i, op in enumerate(sorted_ops[:top_n]):
            print(f"    {i+1:2d}. {op['name']:<40s} | "
                  f"{op['op_type']:<15s} | "
                  f"{op.get('avg_time_ms', 0):>8.4f}ms | "
                  f"FLOPs: {op['flops']:>10,}")
