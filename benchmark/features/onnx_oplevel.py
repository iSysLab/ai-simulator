"""ONNX 그래프에서 op-level 피처 추출 — op_profiler.py의 ONNX 대응물.

torch 비의존. onnx.shape_inference로 각 노드의 입출력 텐서 shape을 복원한 뒤,
op_profiler._calc_op_flops와 동일한 정의로 op별 FLOPs·메모리 읽기/쓰기를 계산해
get_op_level_features와 같은 스키마로 집계한다.

의미 일치 규칙 (op_profiler와 동일):
  - 배치 1 기준 (export 시 dummy 배치 1)
  - memory_read = 입력 텐서 바이트 + 가중치 바이트, memory_write = 출력 텐서 바이트
  - 추적 대상 op만 집계 (Conv/Linear/BN/LN/Pool/활성화 — 그 외 Add·Reshape 등은 무시)
  - Linear(Gemm/MatMul)는 가중치가 initializer인 경우만 (activation×activation
    matmul은 PyTorch 훅에서도 nn.Linear가 아니므로 추적 안 됨)
"""
import numpy as np

# ONNX op_type → op_profiler의 torch 모듈 이름 (flops_ratio_* 키와 일치)
ONNX_TO_TORCH_OP = {
    "Conv": "Conv2d",
    "Gemm": "Linear",
    "MatMul": "Linear",
    "BatchNormalization": "BatchNorm2d",
    "LayerNormalization": "LayerNorm",
    "MaxPool": "MaxPool2d",
    "AveragePool": "AvgPool2d",
    "GlobalAveragePool": "AdaptiveAvgPool2d",
    "Relu": "ReLU",
    "Clip": "ReLU",      # ReLU6는 ONNX에서 Clip으로 export됨
    "LeakyRelu": "LeakyReLU",
    "Gelu": "GELU",
    "Erf": "GELU",       # opset<20에서 GELU가 Erf 체인으로 분해됨
    "Sigmoid": "Sigmoid",
    "Tanh": "Tanh",
    "Dropout": "Dropout",
}

ACTIVATION_OPS = {"ReLU", "LeakyReLU", "GELU", "Sigmoid", "Tanh"}


def _shape_dict(model):
    """텐서 이름 → shape(list[int]) 매핑 (shape inference 결과 포함)."""
    import onnx

    inferred = onnx.shape_inference.infer_shapes(model)
    shapes = {}
    for vi in (list(inferred.graph.value_info) + list(inferred.graph.input)
               + list(inferred.graph.output)):
        dims = []
        for d in vi.type.tensor_type.shape.dim:
            dims.append(d.dim_value if d.dim_value > 0 else 1)
        shapes[vi.name] = dims
    return shapes


def decompose_onnx(onnx_path):
    """ONNX 그래프를 op 리스트로 분해 (op_profiler.decompose_model 대응).

    Returns:
        list[dict]: op_type(torch 이름), flops, memory_read, memory_write,
                    weight_bytes, params
    """
    import onnx
    import onnx.numpy_helper as onp

    model = onnx.load(onnx_path)
    shapes = _shape_dict(model)
    init_shapes = {}   # initializer 이름 → shape
    init_sizes = {}    # initializer 이름 → 원소 수
    for init in model.graph.initializer:
        init_shapes[init.name] = list(init.dims)
        init_sizes[init.name] = int(np.prod(init.dims)) if init.dims else 1

    def elems(name):
        s = shapes.get(name)
        return int(np.prod(s)) if s else 0

    ops = []
    for node in model.graph.node:
        t = ONNX_TO_TORCH_OP.get(node.op_type)
        if t is None:
            continue

        acts_in = [i for i in node.input if i and i not in init_sizes]
        weights = [i for i in node.input if i in init_sizes]
        in_elems = elems(acts_in[0]) if acts_in else 0
        out_elems = elems(node.output[0]) if node.output else 0
        w_params = sum(init_sizes[w] for w in weights)

        # Linear: 가중치 없는 MatMul(attention 내부)은 훅 대상이 아니므로 제외
        if t == "Linear":
            w2 = [w for w in weights if len(init_shapes[w]) == 2]
            if not w2:
                continue
            flops = 2 * init_sizes[w2[0]]           # 2 × in_features × out_features
        elif t == "Conv2d":
            w4 = [w for w in weights if len(init_shapes[w]) == 4]
            if not w4:
                continue
            out_s = shapes.get(node.output[0], [])
            out_hw = int(np.prod(out_s[2:])) if len(out_s) > 2 else 1
            # weight 원소 수 = cout × cin/g × kh × kw → ×2×out_h×out_w
            flops = 2 * init_sizes[w4[0]] * out_hw
        elif t in ("BatchNorm2d", "BatchNorm1d"):
            flops = 2 * in_elems
        elif t == "LayerNorm":
            flops = 5 * in_elems
        elif t in ACTIVATION_OPS:
            flops = in_elems
        elif t in ("MaxPool2d", "AvgPool2d", "AdaptiveAvgPool2d"):
            flops = out_elems
        else:  # Dropout 등
            flops = 0

        ops.append({
            "op_type": t,
            "params": w_params,
            "flops": int(flops),
            "memory_read": in_elems * 4 + w_params * 4,
            "memory_write": out_elems * 4,
            "weight_bytes": w_params * 4,
        })
    return ops


def get_op_level_features_onnx(onnx_path):
    """op_profiler.get_op_level_features와 동일 스키마의 집계 피처."""
    ops = decompose_onnx(onnx_path)
    features = {
        "num_ops": len(ops),
        "total_op_flops": sum(op["flops"] for op in ops),
        "total_op_memory_read": sum(op["memory_read"] for op in ops),
        "total_op_memory_write": sum(op["memory_write"] for op in ops),
        "total_weight_bytes": sum(op["weight_bytes"] for op in ops),
    }
    type_flops = {}
    for op in ops:
        type_flops[op["op_type"]] = type_flops.get(op["op_type"], 0) + op["flops"]
    total_f = features["total_op_flops"]
    for t in ["Conv2d", "Linear", "BatchNorm2d", "LayerNorm",
              "MaxPool2d", "ReLU", "GELU"]:
        features[f"flops_ratio_{t}"] = (type_flops.get(t, 0) / total_f
                                        if total_f > 0 else 0)
    fl = [op["flops"] for op in ops]
    features["max_op_flops"] = max(fl) if fl else 0
    features["avg_op_flops"] = float(np.mean(fl)) if fl else 0
    features["std_op_flops"] = float(np.std(fl)) if fl else 0
    return features


def get_structural_features_onnx(onnx_path):
    """ONNX 그래프에서 구조 피처 (extractor.py 대응 항목만)."""
    import onnx
    import onnx.numpy_helper as onp

    model = onnx.load(onnx_path)
    g = model.graph

    total_params = conv_params = linear_params = 0
    widths = []
    for init in g.initializer:
        size = int(np.prod(init.dims)) if init.dims else 1
        total_params += size
        if len(init.dims) == 4:
            conv_params += size
        elif len(init.dims) == 2:
            linear_params += size
        if init.dims:
            widths.append(int(init.dims[0]))

    cnt = {}
    has_depthwise = 0
    for node in g.node:
        cnt[node.op_type] = cnt.get(node.op_type, 0) + 1
        if node.op_type == "Conv":
            for attr in node.attribute:
                if attr.name == "group" and attr.i > 1:
                    has_depthwise = 1

    num_conv = cnt.get("Conv", 0)
    # 가중치 있는 MatMul만 Linear로 (attention 내부 matmul 제외)
    init_names = {i.name for i in g.initializer}
    num_linear = cnt.get("Gemm", 0)
    for node in g.node:
        if node.op_type == "MatMul" and any(i in init_names for i in node.input):
            num_linear += 1
    num_bn = cnt.get("BatchNormalization", 0)
    num_pool = sum(cnt.get(k, 0) for k in
                   ["MaxPool", "AveragePool", "GlobalAveragePool"])
    num_act = sum(cnt.get(k, 0) for k in
                  ["Relu", "LeakyRelu", "Gelu", "Sigmoid", "Tanh"])

    return {
        "total_params": total_params,
        "trainable_params": total_params,
        "conv_params": conv_params,
        "linear_params": linear_params,
        "bn_params": max(0, total_params - conv_params - linear_params),
        "other_params": 0,
        "total_layers": num_conv + num_linear + num_bn,
        "num_hidden_layers": num_conv + num_linear,
        "num_conv_layers": num_conv,
        "num_linear_layers": num_linear,
        "num_bn_layers": num_bn,
        "num_pool_layers": num_pool,
        "num_activation_layers": num_act,
        "max_width": max(widths) if widths else 0,
        "min_width": min(widths) if widths else 0,
        "avg_width": float(np.mean(widths)) if widths else 0,
        "max_channel_width": max(widths) if widths else 0,
        "model_size_mb": round(total_params * 4 / (1024 * 1024), 4),
        "memory_bytes": total_params * 4,
        "has_batch_norm": 1 if num_bn > 0 else 0,
        "has_pooling": 1 if num_pool > 0 else 0,
        "has_layer_norm": 1 if cnt.get("LayerNormalization", 0) > 0 else 0,
        "has_depthwise": has_depthwise,
        "has_attention": 1 if cnt.get("Softmax", 0) > 0 else 0,
        "has_dropout": 1 if cnt.get("Dropout", 0) > 0 else 0,
    }
