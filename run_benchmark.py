"""통합 벤치마크 실행 진입점 — Zero-Config Cross-Platform

6개 모델 타입: ANN, CNN, ResNet, MobileNet, Transformer, GAN
2개 데이터셋: MNIST (ANN, CNN, ResNet, MobileNet) / CIFAR-10 (Transformer, GAN)

사용법:
    python run_benchmark.py                           # 전체 실행 (자동 디바이스 감지)
    python run_benchmark.py --quick                   # 빠른 테스트 (3 반복, 대표 모델)
    python run_benchmark.py --full-pipeline           # 벤치마크 → 학습 → 시각화 원클릭
    python run_benchmark.py --model simple_ann        # ANN만 실행
    python run_benchmark.py --device auto             # 자동 디바이스 감지 (기본값)
    python run_benchmark.py --device cpu              # CPU만 사용
    python run_benchmark.py --repeats 3               # 반복 횟수 지정
    python run_benchmark.py --resume                  # 중단 후 이어서 실행
    python run_benchmark.py --profile-ops             # Op-level 프로파일링

측정 프로토콜 v2 옵션 (2026-09 재측정, 심사 대응):
    --require-idle                # 각 구성 시작 전 CPU/GPU 유휴 확인, 아니면 대기
    --tag remeasure_v2            # 결과 행에 태그 기록
    --subset stratified:60        # 계열별 균등 60구성만 (외부 백엔드 시험, 배치 스윕용)
    --subset every:4 | names:A,B  # 다른 부분집합 지정 방식
    --batch-size 128              # 학습 배치 크기 (기본 64, 배치 스윕용)
    --discard-first               # 첫 반복을 통계에서 제외 (원시값에는 유지)
  모든 행에 소프트웨어 환경(sw_*), 측정 직전 부하(load_*), 반복별 원시값, 피크 메모리가 기록된다.
"""
import argparse
import math
import socket
import subprocess
import sys
import time
from datetime import datetime

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
from benchmark.platform import PlatformInfo
from benchmark.platform.compatibility import CompatibilityMatrix
from benchmark.platform.software_env import (
    collect_software_env, measure_system_load, is_idle,
)

# MNIST 모델 / CIFAR-10 모델 구분
MNIST_MODELS = {'simple_ann', 'simple_cnn', 'resnet_mnist', 'mobilenet_mnist'}
CIFAR10_MODELS = {'transformer', 'gan'}
GAN_MODELS = {'gan'}
FAMILY_ORDER = ['simple_ann', 'simple_cnn', 'resnet_mnist',
                'mobilenet_mnist', 'transformer', 'gan']


def parse_args():
    parser = argparse.ArgumentParser(description='통합 벤치마크 실행')
    parser.add_argument('--model', type=str, default=None,
                        choices=['simple_ann', 'simple_cnn',
                                 'resnet_mnist', 'mobilenet_mnist',
                                 'transformer', 'gan'],
                        help='특정 모델만 실행')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='디바이스 선택 (기본: auto)')
    parser.add_argument('--repeats', type=int, default=None,
                        help='반복 횟수 (기본: 10, --quick: 3)')
    parser.add_argument('--resume', action='store_true',
                        help='기존 결과에서 이어서 실행')
    parser.add_argument('--output', type=str,
                        default='results/benchmark_results.json',
                        help='결과 저장 경로')
    parser.add_argument('--profile-ops', action='store_true',
                        help='Op-level 프로파일링 수행')
    parser.add_argument('--quick', action='store_true',
                        help='빠른 테스트 (3 반복, 대표 모델)')
    parser.add_argument('--full-pipeline', action='store_true',
                        help='벤치마크 → 학습 → 시각화 원클릭')
    # --- 측정 프로토콜 v2 ---
    parser.add_argument('--subset', type=str, default=None,
                        help='부분집합: stratified:N (계열별 균등 N개), every:K, names:A,B,C')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='학습 배치 크기 (기본 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        help='추론 배치 크기 (기본 1000)')
    parser.add_argument('--discard-first', action='store_true',
                        help='첫 반복을 통계에서 제외 (원시값에는 유지)')
    parser.add_argument('--require-idle', action='store_true',
                        help='각 구성 시작 전 CPU/GPU 유휴 상태를 확인하고 아니면 대기')
    parser.add_argument('--idle-cpu-max', type=float, default=20.0,
                        help='유휴 판정 CPU 사용률 상한 %% (기본 20)')
    parser.add_argument('--idle-gpu-max', type=float, default=10.0,
                        help='유휴 판정 GPU 사용률 상한 %% (기본 10, CUDA만)')
    parser.add_argument('--idle-wait', type=int, default=600,
                        help='유휴 대기 최대 초 (기본 600, 초과 시 경고 후 진행)')
    parser.add_argument('--tag', type=str, default='',
                        help='결과 행에 기록할 태그 (예: remeasure_v2)')
    return parser.parse_args()


def select_subset(configs, spec):
    """부분집합 선택. 결정적(시드 불필요)이라 같은 spec은 항상 같은 구성을 고른다.

    stratified:N — 계열별로 ceil(N/계열 수)개를 생성 순서에서 균등 간격으로 뽑는다
                   (생성 순서가 크기 순에 가까우므로 소·중·대가 고르게 들어간다).
    every:K      — 전체에서 K개마다 1개.
    names:A,B    — model_name 목록.
    """
    kind, _, arg = spec.partition(':')
    if kind == 'stratified':
        n_total = int(arg)
        families = [f for f in FAMILY_ORDER if any(c['model_type'] == f for c in configs)]
        per = max(1, math.ceil(n_total / max(1, len(families))))
        picked = []
        for fam in families:
            fam_cfgs = [c for c in configs if c['model_type'] == fam]
            k = min(per, len(fam_cfgs))
            if k == 1:
                idx = [0]
            else:
                idx = sorted({round(j * (len(fam_cfgs) - 1) / (k - 1)) for j in range(k)})
            picked.extend(fam_cfgs[i] for i in idx)
        return picked
    if kind == 'every':
        return configs[::max(1, int(arg))]
    if kind == 'names':
        wanted = {s.strip() for s in arg.split(',') if s.strip()}
        return [c for c in configs if c['model_name'] in wanted]
    raise ValueError(f"알 수 없는 --subset 형식: {spec}")


def wait_until_idle(device_type, args):
    """유휴 상태가 될 때까지 대기. 마지막으로 측정한 부하와 대기 초를 반환."""
    waited = 0
    while True:
        load = measure_system_load(device_type)
        if not args.require_idle or is_idle(load, args.idle_cpu_max, args.idle_gpu_max):
            return load, waited
        if waited >= args.idle_wait:
            print(f"  [경고] 유휴 대기 {args.idle_wait}s 초과 — 부하 상태로 진행 "
                  f"(CPU {load['load_cpu_pct']}%, GPU {load['load_gpu_util_pct']}%)")
            return load, waited
        print(f"  [대기] 기기 사용 중 (CPU {load['load_cpu_pct']}%, "
              f"GPU {load['load_gpu_util_pct']}%) — 15초 후 재확인")
        time.sleep(15)
        waited += 15


def print_platform_info(dm, compat):
    """플랫폼 정보 출력"""
    devices = compat.get_available_devices()
    print(f"\n{'='*60}")
    print("  플랫폼 자동 감지 결과")
    print(f"{'='*60}")
    for dev in devices:
        hw = compat.get_platform(dev)
        label = dm.label(
            next(d for d in dm.devices if d.type == dev)
        ) if any(d.type == dev for d in dm.devices) else dev
        print(f"  [{label}]")
        print(f"    OS: {hw['os_type']}, 가속기: {hw['accelerator_name']}")
        if dev != 'cpu':
            print(f"    GPU: {hw['gpu_core_count']}코어, "
                  f"{hw['gpu_memory_gb']}GB, "
                  f"FP32: {hw['tflops_fp32']} TFLOPS")
        print(f"    CPU: {hw['cpu_cores_physical']}P+{hw['cpu_efficiency_cores']}E, "
              f"RAM: {hw['ram_total_gb']}GB "
              f"({'unified' if hw['is_unified_memory'] else hw.get('memory_type', 'ddr')})")
        batch = compat.get_optimal_batch_size(dev)
        print(f"    권장 배치: {batch}")
    print(f"{'='*60}\n")


def run_configs_on_device(configs, device, dev_label, data_mgr, runner,
                          results_mgr, args, compat=None, sw_env=None):
    """한 장치에서 설정 목록 벤치마크 실행"""
    print(f"\n  데이터를 {dev_label}에 사전 로딩 중...")
    train_batches, test_batches = data_mgr.preload_to_device(device)
    print(f"  사전 로딩 완료. (학습 배치 {len(train_batches)}개 × {args.batch_size}, "
          f"추론 배치 {len(test_batches)}개 × {args.test_batch_size})\n")

    for cfg in configs:
        model_name = cfg['model_name']
        model_type = cfg['model_type']
        is_gan = model_type in GAN_MODELS

        # resume: 이미 완료된 설정 건너뛰기
        if args.resume and results_mgr.is_completed(model_name, dev_label):
            print(f"  [건너뜀] {model_name} ({dev_label}) — 이미 완료")
            continue

        # 호환성 검사
        if compat:
            skip, reason = compat.should_skip_config(cfg, device.type)
            if skip:
                print(f"  [건너뜀] {model_name}: {reason}")
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
                device_str=device.type,
                config=cfg['config'],
                batch_size=args.batch_size)
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

        # 측정 직전 시스템 부하 확인 (유휴 증빙, --require-idle 시 대기)
        load, waited = wait_until_idle(device.type, args)

        # 워밍업: 모든 계열·백엔드 동일 (실제 학습 스텝 1회 + 평가 순전파 1회)
        try:
            if is_gan:
                runner.warmup_gan(model_fn, device, train_batches,
                                  batch_size=args.batch_size)
            else:
                runner.warmup(model_fn, device, train_batches, test_batches)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"    [OOM] 워밍업 중 GPU 메모리 부족 — 건너뜀")
                runner.dm.clear_cache(device)
                continue
            raise

        # 벤치마크 실행
        try:
            if is_gan:
                timing = runner.run_gan(
                    model_fn, device, train_batches,
                    batch_size=args.batch_size,
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
            'n_train_batches': len(train_batches),
            'n_test_batches': len(test_batches),
            'measured_at': datetime.now().isoformat(timespec='seconds'),
            'host': socket.gethostname(),
            'tag': args.tag,
            'idle_wait_s': waited,
            **load,
            **(sw_env or {}),
        }
        results_mgr.append_and_save(result)

        if is_gan:
            print(f"    >> 평균 학습: {timing['avg_train']}s | "
                  f"평균 추론(생성): {timing['avg_infer']}s | "
                  f"피크 메모리: {timing['peak_mem_mb']}MB\n")
        else:
            print(f"    >> 평균 학습: {timing['avg_train']}s | "
                  f"평균 추론: {timing['avg_infer']}s | "
                  f"정확도: {timing['avg_accuracy']}% | "
                  f"피크 메모리: {timing['peak_mem_mb']}MB\n")

    # 정리
    del train_batches, test_batches
    runner.dm.clear_cache(device)


def run_full_pipeline(args):
    """벤치마크 → 학습 → 시각화 원클릭"""
    print("\n" + "=" * 60)
    print("  Full Pipeline: 벤치마크 → 학습 → 시각화")
    print("=" * 60)

    # 1. 벤치마크 실행 (이미 main에서 실행됨)

    # 2. 예측 모델 학습
    print("\n\n" + "=" * 60)
    print("  [2/3] 예측 모델 학습")
    print("=" * 60)
    try:
        subprocess.run(
            [sys.executable, 'train_predictor.py',
             '--input', args.output, '--features', 'core'],
            check=True)
    except subprocess.CalledProcessError as e:
        print(f"  [경고] 예측 모델 학습 실패: {e}")

    # 3. 시각화
    print("\n\n" + "=" * 60)
    print("  [3/3] 시각화 생성")
    print("=" * 60)
    try:
        subprocess.run(
            [sys.executable, 'visualize_results.py',
             '--input', args.output],
            check=True)
    except subprocess.CalledProcessError as e:
        print(f"  [경고] 시각화 생성 실패: {e}")

    print("\n\nFull Pipeline 완료!")


def main():
    args = parse_args()

    # Quick 모드 기본값
    if args.quick:
        if args.repeats is None:
            args.repeats = 3
    if args.repeats is None:
        args.repeats = 10

    print(f"등록된 모델: {list_models()}")

    # 소프트웨어 환경 (모든 결과 행에 기록)
    sw_env = collect_software_env()
    print("소프트웨어 환경: " + ", ".join(
        f"{k[3:]}={v}" for k, v in sw_env.items()
        if k in ('sw_python', 'sw_torch', 'sw_cuda_runtime', 'sw_cudnn',
                 'sw_onednn', 'sw_gpu_driver', 'sw_macos') and v not in ('', None)))

    # 장치 관리 + 호환성
    dm = DeviceManager()
    compat = CompatibilityMatrix()

    # 디바이스 필터링
    if args.device == 'auto':
        target_devices = dm.devices
    else:
        target_devices = [d for d in dm.devices if d.type == args.device]
        if not target_devices:
            print(f"경고: {args.device} 디바이스를 사용할 수 없습니다. CPU로 fallback.")
            target_devices = [d for d in dm.devices if d.type == 'cpu']

    print(f"측정 대상 장치: {[dm.label(d) for d in target_devices]}")

    # 플랫폼 정보 출력
    print_platform_info(dm, compat)

    # 실험 러너
    runner = ExperimentRunner(dm, repeats=args.repeats,
                              discard_first=args.discard_first)

    # 결과 관리
    results_mgr = ResultsManager(args.output)
    if args.resume:
        print(f"기존 결과 {len(results_mgr.results)}개 로딩. 이어서 실행합니다.")

    # 설정 생성
    configs = generate_configs(model_type=args.model)

    # Quick 모드: 대표 모델만 선택
    if args.quick and args.model is None:
        quick_models = {'simple_ann', 'simple_cnn', 'transformer'}
        configs = [c for c in configs if c['model_type'] in quick_models]
        # 각 타입에서 소/중/대 3개만
        quick_configs = []
        for mt in quick_models:
            mt_configs = [c for c in configs if c['model_type'] == mt]
            if len(mt_configs) >= 3:
                quick_configs.extend([mt_configs[0], mt_configs[len(mt_configs)//2], mt_configs[-1]])
            else:
                quick_configs.extend(mt_configs)
        configs = quick_configs

    # 부분집합 (외부 백엔드 시험, 배치 스윕)
    if args.subset:
        configs = select_subset(configs, args.subset)
        print(f"부분집합 {args.subset}: {len(configs)}개 구성 "
              f"({', '.join(sorted({c['model_type'] for c in configs}))})")

    print(f"총 {len(configs)}개 설정 실행 예정 → {args.output}"
          + (f" [tag={args.tag}]" if args.tag else "")
          + (" [require-idle]" if args.require_idle else "") + "\n")

    # MNIST / CIFAR-10 설정 분리
    mnist_configs = [c for c in configs if c['model_type'] in MNIST_MODELS]
    cifar_configs = [c for c in configs if c['model_type'] in CIFAR10_MODELS]

    for device in target_devices:
        dev_label = dm.label(device)
        print(f"\n{'='*60}")
        print(f"=== [{dev_label}] 벤치마크 시작 ===")
        print(f"{'='*60}")

        # MNIST 모델 벤치마크
        if mnist_configs:
            print(f"\n--- MNIST 데이터셋 ({len(mnist_configs)}개 설정) ---")
            mnist_data = MNISTDataManager(batch_size=args.batch_size,
                                          test_batch_size=args.test_batch_size)
            run_configs_on_device(
                mnist_configs, device, dev_label, mnist_data,
                runner, results_mgr, args, compat, sw_env)
            del mnist_data

        # CIFAR-10 모델 벤치마크
        if cifar_configs:
            print(f"\n--- CIFAR-10 데이터셋 ({len(cifar_configs)}개 설정) ---")
            cifar_data = CIFAR10DataManager(batch_size=args.batch_size,
                                            test_batch_size=args.test_batch_size)
            run_configs_on_device(
                cifar_configs, device, dev_label, cifar_data,
                runner, results_mgr, args, compat, sw_env)
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

    # Full pipeline: 벤치마크 후 학습 + 시각화
    if args.full_pipeline:
        run_full_pipeline(args)


if __name__ == '__main__':
    main()
