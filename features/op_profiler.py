"""Op-Level 프로파일러: 모델을 연산 단위로 분해하여 개별 FLOPs/메모리 집계

출처: khg9859/present utils/op_profiler.py 포팅
수정: MPS → CUDA/CPU 지원, dal-merge feature 스키마에 맞게 조정
"""

import torch
import torch.nn as nn
import numpy as np


OP_TYPES = {
    'Conv2d':           nn.Conv2d,
    'Linear':           nn.Linear,
    'BatchNorm2d':      nn.BatchNorm2d,
    'BatchNorm1d':      nn.BatchNorm1d,
    'MaxPool2d':        nn.MaxPool2d,
    'AvgPool2d':        nn.AvgPool2d,
    'AdaptiveAvgPool2d': nn.AdaptiveAvgPool2d,
    'ReLU':             nn.ReLU,
    'ReLU6':            nn.ReLU6,
    'GELU':             nn.GELU,
    'LeakyReLU':        nn.LeakyReLU,
    'LayerNorm':        nn.LayerNorm,
    'Sigmoid':          nn.Sigmoid,
    'Tanh':             nn.Tanh,
}


def _calc_op_flops(module, input_tensor, output_tensor):
    """개별 op의 FLOPs 계산"""
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


def decompose_model(model, input_shape=(1, 1, 28, 28)):
    """모델을 개별 연산 단위로 분해하여 FLOPs/메모리 정보 수집"""
    ops = []
    hooks = []

    def make_hook(name, module):
        def hook_fn(mod, inp, out):
            in_tensor  = inp[0] if isinstance(inp, tuple) and inp else inp
            out_tensor = out[0] if isinstance(out, tuple) and out else out

            in_shape  = tuple(in_tensor.shape)  if hasattr(in_tensor,  'shape') else ()
            out_shape = tuple(out_tensor.shape) if hasattr(out_tensor, 'shape') else ()

            params       = sum(p.numel() for p in mod.parameters())
            flops        = _calc_op_flops(mod, in_tensor, out_tensor)
            in_bytes     = in_tensor.numel()  * 4 if hasattr(in_tensor,  'numel') else 0
            out_bytes    = out_tensor.numel() * 4 if hasattr(out_tensor, 'numel') else 0
            weight_bytes = params * 4

            ops.append({
                'name':         name,
                'op_type':      type(mod).__name__,
                'params':       params,
                'flops':        flops,
                'input_shape':  in_shape,
                'output_shape': out_shape,
                'memory_read':  in_bytes + weight_bytes,
                'memory_write': out_bytes,
                'weight_bytes': weight_bytes,
            })
        return hook_fn

    for name, module in model.named_modules():
        if len(list(module.children())) == 0 and isinstance(module, tuple(OP_TYPES.values())):
            hooks.append(module.register_forward_hook(make_hook(name, module)))

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


def get_op_level_features(model, input_shape=(1, 1, 28, 28)):
    """op 분해 결과에서 feature 벡터 생성

    Args:
        model: PyTorch 모델
        input_shape: 입력 shape (배치 포함)

    Returns:
        dict: op-level feature 딕셔너리
    """
    ops = decompose_model(model, input_shape)

    features = {
        'num_ops':               len(ops),
        'total_op_flops':        sum(op['flops'] for op in ops),
        'total_op_memory_read':  sum(op['memory_read'] for op in ops),
        'total_op_memory_write': sum(op['memory_write'] for op in ops),
        'memory_bytes':          sum(op['weight_bytes'] for op in ops),
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
        features[key] = round(type_flops.get(op_type, 0) / total_f, 4) if total_f > 0 else 0.0

    # op FLOPs 통계
    op_flops_list = [op['flops'] for op in ops if op['flops'] > 0]
    if op_flops_list:
        features['max_op_flops'] = int(max(op_flops_list))
        features['avg_op_flops'] = round(float(np.mean(op_flops_list)), 0)
        features['std_op_flops'] = round(float(np.std(op_flops_list)), 0)
    else:
        features['max_op_flops'] = 0
        features['avg_op_flops'] = 0
        features['std_op_flops'] = 0

    return features
