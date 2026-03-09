"""통합 벤치마크 실행 진입점

6개 모델 타입: ANN, CNN, ResNet, MobileNet, Transformer, GAN
2개 데이터셋: MNIST (ANN, CNN, ResNet, MobileNet) / CIFAR-10 (Transformer, GAN)

사용법:
    python run_benchmark.py                           # 전체 실행
    python run_benchmark.py --model simple_ann        # ANN만 실행
    python run_benchmark.py --model transformer       # Transformer만 실행
    python run_benchmark.py --model gan               # GAN만 실행
    python run_benchmark.py --device cpu              # CPU만 사용
    python run_benchmark.py --repeats 3               # 빠른 테스트 (3회)
    python run_benchmark.py --resume                  # 중단 후 이어서 실행
"""
import argparse

# 모델 등록을 위해 모든 모델 모듈 import
from benchmark.models import (
    simple_ann, simple_cnn, resnet_mnist, mobilenet_mnist,
    transformer, gan,
)
from benchmark.models.registry import create_model, list_models
from benchmark.features.extractor import extract_features
from benchmark.features.op_profiler import (
    decompose_model, measure_op_times, simulate_total_time,
    get_op_level_features, print_op_profile,
)
from benchmark.runner.device import DeviceManager
from benchmark.runner.data import MNISTDataManager, CIFAR10DataManager
from benchmark.runner.experiment import ExperimentRunner
from benchmark.configs.generator import generate_configs
from benchmark.results.io import ResultsManager

# MNIST 모델 / CIFAR-10 모델 구분
MNIST_MODELS = {'simple_ann', 'simple_cnn', 'resnet_mnist', 'mobilenet_mnist'}
CIFAR10_MODELS = {'transformer', 'gan'}
GAN_MODELS = {'gan'}


def parse_args():
    parser = argparse.ArgumentParser(description='통합 벤치마크 실행')
    parser.add_argument('--model', type=str, default=None,
                        choices=['simple_ann', 'simple_cnn',
                                 'resnet_mnist', 'mobilenet_mnist',
                                 'transformer', 'gan'],
                        help='특정 모델만 실행')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cpu', 'cuda', 'mps'],
                        help='특정 장치만 사용')
    parser.add_argument('--repeats', type=int, default=10,
                        help='반복 횟수 (기본: 10)')
    parser.add_argument('--resume', action='store_true',
                        help='기존 결과에서 이어서 실행')
    parser.add_argument('--output', type=str,
                        default='results/benchmark_results.json',
                        help='결과 저장 경로')
    parser.add_argument('--profile-ops', action='store_true',
                        help='Op-level 프로파일링 수행 (개별 연산 시간 측정)')
    return parser.parse_args()


def run_configs_on_device(configs, device, dev_label, data_mgr, runner,
                          results_mgr, args):
    """한 장치에서 설정 목록 벤치마크 실행"""
    print(f"\n  데이터를 {dev_label}에 사전 로딩 중...")
    train_batches, test_batches = data_mgr.preload_to_device(device)
    print(f"  사전 로딩 완료.\n")

    for cfg in configs:
        model_name = cfg['model_name']
        model_type = cfg['model_type']
        is_gan = model_type in GAN_MODELS

        # resume: 이미 완료된 설정 건너뛰기
        if args.resume and results_mgr.is_completed(model_name, dev_label):
            print(f"  [건너뜀] {model_name} ({dev_label}) — 이미 완료")
            continue

        # 모델 생성 함수
        def model_fn(c=cfg):
            return create_model(c['model_type'], **c['config'])

        # 더미 모델로 피처 추출
        try:
            dummy_model = model_fn()
        except Exception as e:
            print(f"  [실패] {model_name}: 모델 생성 오류 — {e}")
            continue

        param_count = sum(p.numel() for p in dummy_model.parameters())

        try:
            features = extract_features(
                dummy_model, model_type,
                input_shape=data_mgr.input_shape,
                device_str=device.type)
        except Exception as e:
            print(f"  [실패] {model_name}: 피처 추출 오류 — {e}")
            del dummy_model
            continue
        # Op-level 프로파일링 (선택적)
        op_features = {}
        if args.profile_ops and not is_gan:
            try:
                ops = measure_op_times(
                    dummy_model, input_shape=data_mgr.input_shape,
                    device=device.type, warmup=3, repeats=5)
                sim = simulate_total_time(ops)
                op_features = get_op_level_features(ops)
                op_features['sim_total_time_ms'] = sim['total_time_ms']
                print_op_profile(ops, top_n=10)
            except Exception as e:
                print(f"  [경고] Op 프로파일링 실패 — {e}")

        del dummy_model

        print(f"  --- {model_name} (파라미터: {param_count:,}) ---")

        # GPU 워밍업 (GAN이 아닌 경우에만)
        if not is_gan:
            runner.dm.warmup(device, model_fn)

        # 벤치마크 실행
        try:
            if is_gan:
                timing = runner.run_gan(
                    model_fn, device, train_batches,
                    config_name=model_name)
            else:
                timing = runner.run(
                    model_fn, device,
                    train_batches, test_batches,
                    data_mgr.num_test_samples,
                    config_name=model_name)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"    [OOM] GPU 메모리 부족 — 건너뜀")
                runner.dm.clear_cache(device)
                continue
            raise

        # 결과 조합 및 저장
        result = {
            'model_type': model_type,
            'model_name': model_name,
            'device': dev_label,
            'config': cfg['config'],
            **features,
            **timing,
            **op_features,
        }
        results_mgr.append_and_save(result)

        if is_gan:
            print(f"    >> 평균 학습: {timing['avg_train']}s | "
                  f"평균 추론(생성): {timing['avg_infer']}s\n")
        else:
            print(f"    >> 평균 학습: {timing['avg_train']}s | "
                  f"평균 추론: {timing['avg_infer']}s | "
                  f"정확도: {timing['avg_accuracy']}%\n")

    # 정리
    del train_batches, test_batches
    runner.dm.clear_cache(device)


def main():
    args = parse_args()

    print(f"등록된 모델: {list_models()}")

    # 장치 관리
    dm = DeviceManager()
    print(f"측정 대상 장치: {[dm.label(d) for d in dm.devices]}")

    # 실험 러너
    runner = ExperimentRunner(dm, repeats=args.repeats)

    # 결과 관리
    results_mgr = ResultsManager(args.output)
    if args.resume:
        print(f"기존 결과 {len(results_mgr.results)}개 로딩. 이어서 실행합니다.")

    # 설정 생성
    configs = generate_configs(model_type=args.model)
    print(f"총 {len(configs)}개 설정 실행 예정\n")

    # MNIST / CIFAR-10 설정 분리
    mnist_configs = [c for c in configs if c['model_type'] in MNIST_MODELS]
    cifar_configs = [c for c in configs if c['model_type'] in CIFAR10_MODELS]

    for device in dm.devices:
        if args.device and device.type != args.device:
            continue

        dev_label = dm.label(device)
        print(f"\n{'='*60}")
        print(f"=== [{dev_label}] 벤치마크 시작 ===")
        print(f"{'='*60}")

        # MNIST 모델 벤치마크
        if mnist_configs:
            print(f"\n--- MNIST 데이터셋 ({len(mnist_configs)}개 설정) ---")
            mnist_data = MNISTDataManager()
            run_configs_on_device(
                mnist_configs, device, dev_label, mnist_data,
                runner, results_mgr, args)
            del mnist_data

        # CIFAR-10 모델 벤치마크
        if cifar_configs:
            print(f"\n--- CIFAR-10 데이터셋 ({len(cifar_configs)}개 설정) ---")
            cifar_data = CIFAR10DataManager()
            run_configs_on_device(
                cifar_configs, device, dev_label, cifar_data,
                runner, results_mgr, args)
            del cifar_data

    # CSV 내보내기
    results_mgr.to_csv()

    # === 최종 보고서 ===
    print(f"\n\n{'='*60}")
    print(f"   벤치마크 최종 결과 ({args.repeats}회 반복)")
    print(f"{'='*60}")

    results_by_device = {}
    for r in results_mgr.results:
        dev = r['device']
        if dev not in results_by_device:
            results_by_device[dev] = []
        results_by_device[dev].append(r)

    for dev_name, results in results_by_device.items():
        print(f"\n[{dev_name}]")
        print(f"  {'모델':<40} | {'파라미터':>12} | {'FLOPs':>12} | "
              f"{'학습(s)':>10} | {'추론(s)':>10} | {'정확도':>7}")
        print(f"  {'-'*40} | {'-'*12} | {'-'*12} | "
              f"{'-'*10} | {'-'*10} | {'-'*7}")
        for r in results:
            acc_str = f"{r['avg_accuracy']:>6.2f}%" if r['avg_accuracy'] > 0 else "  N/A  "
            print(f"  {r['model_name']:<40} | "
                  f"{r['total_params']:>12,} | "
                  f"{r['flops']:>12,} | "
                  f"{r['avg_train']:>10.5f} | "
                  f"{r['avg_infer']:>10.5f} | "
                  f"{acc_str}")

    print(f"\n결과 저장: {args.output}")
    print("벤치마크 완료!")


if __name__ == '__main__':
    main()
