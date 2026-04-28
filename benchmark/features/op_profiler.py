"""Op-Level 프로파일러: 모델을 연산 단위로 분해하여 개별 시간/메모리 측정

교수님 요구사항: "모델을 작은 연산 단위로 나누고 예측"
→ Conv, Linear, BN, Pool, Activation 등 각 op의 시간을 개별 측정
→ 합산하여 전체 시간 시뮬레이션 (bottom-up 예측)

기존 방식 (end-to-end):
    전체 모델 → 30개 통계 피처 → ML → 전체 시간 예측

시뮬레이션 방식 (op-level):
    모델 → [op1, op2, op3, ...] 분해
    → 각 op의 (타입, 파라미터, FLOPs, 입출력 크기) 추출
    → 각 op 시간 예측 → 합산 = 전체 시간 예측
"""
import time
import torch
import torch.nn as nn
import numpy as np


# 지원하는 연산 타입
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


def decompose_model(model, input_shape=(1, 1, 28, 28)):
    """모델을 개별 연산(op) 단위로 분해

    Forward hook으로 각 op의 입출력 shape, FLOPs, 파라미터 수를 기록.

    Args:
        model: PyTorch 모델
        input_shape: 입력 텐서 shape

    Returns:
        list[dict]: 각 op의 상세 정보
            - name: 모듈 경로명
            - op_type: 연산 타입 (Conv2d, Linear, ...)
            - params: 파라미터 수
            - flops: 추정 FLOPs
            - input_shape: 입력 텐서 shape
            - output_shape: 출력 텐서 shape
            - memory_read: 입력 데이터 읽기 바이트
            - memory_write: 출력 데이터 쓰기 바이트
            - weight_bytes: 가중치 크기 (바이트)
    """
    ops = []
    hooks = []

    def make_hook(name, module):
        def hook_fn(mod, inp, out):
            # 입출력 shape 추출
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

            # 파라미터 수
            params = sum(p.numel() for p in mod.parameters())

            # FLOPs 추정
            flops = _calc_op_flops(mod, in_tensor, out_tensor)

            # 메모리 접근량 추정
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

    # 리프 모듈에만 hook 등록 (중복 방지)
    for name, module in model.named_modules():
        is_leaf = len(list(module.children())) == 0
        is_tracked = isinstance(module, tuple(OP_TYPES.values()))
        if is_leaf and is_tracked:
            hooks.append(module.register_forward_hook(make_hook(name, module)))

    # Forward pass로 hook 실행
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
    """개별 op의 FLOPs 계산 (Attention matmul 포함)"""
    if isinstance(module, nn.Conv2d):
        out_h, out_w = output_tensor.size(2), output_tensor.size(3)
        k = module.kernel_size[0] * module.kernel_size[1]
        return 2 * module.in_channels * module.out_channels * k * out_h * out_w // module.groups

    elif isinstance(module, nn.Linear):
        # Attention의 QKV projection, out_proj 등도 여기서 잡힘
        return 2 * module.in_features * module.out_features

    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
        return 2 * input_tensor.numel()

    elif isinstance(module, nn.LayerNorm):
        # LayerNorm: mean + variance + normalize + scale + shift
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

    Args:
        model: PyTorch 모델
        input_shape: 입력 shape
        device: 측정 장치
        warmup: 워밍업 횟수
        repeats: 반복 측정 횟수

    Returns:
        list[dict]: 각 op 정보 + avg_time_ms, std_time_ms
    """
    device = torch.device(device) if isinstance(device, str) else device
    model = model.to(device).eval()

    # 먼저 op 분해 (CPU에서)
    model_cpu = model.cpu()
    ops = decompose_model(model_cpu, input_shape)
    model = model.to(device)

    # 각 op에 시간 측정 hook 부착
    op_times = {i: [] for i in range(len(ops))}
    hooks = []
    op_index = [0]

    def make_time_hook(idx):
        def hook_fn(mod, inp, out):
            pass  # 시간은 아래에서 별도 측정
        return hook_fn

    # op별 개별 시간 측정 (forward hook 내 perf_counter)
    time_records = {}
    time_hooks = []

    leaf_modules = []
    for name, module in model.named_modules():
        is_leaf = len(list(module.children())) == 0
        is_tracked = isinstance(module, tuple(OP_TYPES.values()))
        if is_leaf and is_tracked:
            leaf_modules.append((name, module))

    # 각 리프 모듈에 시간 측정 hook 등록
    for idx, (name, module) in enumerate(leaf_modules):
        time_records[idx] = {'start': 0, 'times': []}

        def make_pre_hook(i):
            def pre_hook(mod, inp):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                time_records[i]['start'] = time.perf_counter()
            return pre_hook

        def make_post_hook(i):
            def post_hook(mod, inp, out):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - time_records[i]['start']
                time_records[i]['times'].append(elapsed)
            return post_hook

        time_hooks.append(module.register_forward_pre_hook(make_pre_hook(idx)))
        time_hooks.append(module.register_forward_hook(make_post_hook(idx)))

    # 워밍업
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

    # 결과 병합
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

    # op 타입별 집계
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

    # op 타입별 개수 및 FLOPs 비율
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
    """op 프로파일 결과 출력"""
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
