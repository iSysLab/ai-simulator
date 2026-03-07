# 모델 레지스트리: 이름 → 모델 생성 함수 매핑
MODEL_REGISTRY = {}


def register_model(name):
    """데코레이터: 모델을 레지스트리에 등록"""
    def decorator(fn):
        MODEL_REGISTRY[name] = fn
        return fn
    return decorator


def create_model(name, **kwargs):
    """레지스트리에서 모델 생성"""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"알 수 없는 모델: {name}. 등록된 모델: {list_models()}")
    return MODEL_REGISTRY[name](**kwargs)


def list_models():
    """등록된 모델 이름 목록"""
    return list(MODEL_REGISTRY.keys())
