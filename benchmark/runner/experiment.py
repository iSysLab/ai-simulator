"""벤치마크 실험 실행기 — 분류 모델 + GAN 모델 지원

측정 프로토콜 v2 (2026-09 재측정, 심사 대응 R1-10/R1-11):
  - 워밍업: 모든 백엔드(CPU 포함)·모든 계열에서 실제 학습 배치 1개로 학습 스텝 1회 +
    평가 순전파 1회. GAN은 D 스텝 + G 스텝 1회 + 생성 1회. 워밍업에 쓴 모델은 버린다.
    (v1은 GPU의 비-GAN 모델에만 배치 1 no_grad 순전파 1회였다.)
  - 반복: repeats회, 매 반복 새 모델. 반복별 원시값(train_times, infer_times)을 저장한다.
  - 통계: avg_*는 산술평균, std_*는 모표준편차(ddof=0). discard_first=True면 첫 반복을
    통계에서 제외한다(원시값에는 남기고 n_used로 표시).
  - 피크 메모리: CUDA는 max_memory_allocated, MPS는 driver_allocated_memory의 최댓값(MB).
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .device import DeviceManager


def _stats(times, discard_first):
    """(통계에 쓴 값들, 평균, 모표준편차)"""
    used = times[1:] if (discard_first and len(times) > 1) else times
    return used, round(float(np.mean(used)), 5), round(float(np.std(used)), 5)


class ExperimentRunner:
    """단일 모델 설정에 대한 벤치마크 실험 실행

    분류 모델 (ANN, CNN, ResNet, MobileNet, Transformer):
      sync → perf_counter → 학습/추론 → sync → perf_counter

    GAN:
      Generator/Discriminator 적대적 학습 시간 + Generator 추론 시간 측정
    """

    PROTOCOL = 'v2-uniform-warmup'

    def __init__(self, device_manager, epochs=1, repeats=10, lr=0.01,
                 discard_first=False):
        self.dm = device_manager
        self.epochs = epochs
        self.repeats = repeats
        self.lr = lr
        self.discard_first = discard_first

    # ------------------------------------------------------------------ 워밍업
    def warmup(self, model_fn, device, train_batches, test_batches):
        """분류 모델 워밍업: 실제 배치 1개로 학습 스텝 1회 + 평가 순전파 1회 (모든 백엔드)"""
        model = model_fn().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.lr)

        model.train()
        data, target = train_batches[0]
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()

        # 평가 경로도 측정 루프와 같은 커널(argmax → eq → sum → item)을 한 번 거친다.
        # CUDA는 커널을 첫 사용 시점에 로드하므로 경로가 다르면 첫 반복에 잔여 비용이 남는다.
        model.eval()
        with torch.no_grad():
            data, target = test_batches[0]
            pred = model(data).argmax(dim=1, keepdim=True)
            _ = pred.eq(target.view_as(pred)).sum().item()

        self.dm.sync(device)
        del model, optimizer, criterion, loss
        self.dm.clear_cache(device)

    def warmup_gan(self, model_fn, device, train_batches, batch_size=64):
        """GAN 워밍업: D 스텝 1회 + G 스텝 1회 + 생성 1회 (모든 백엔드)"""
        gan = model_fn().to(device)
        G, D, latent_dim = gan.generator, gan.discriminator, gan.latent_dim
        optimizer_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
        criterion = nn.BCELoss()

        G.train()
        D.train()
        real_imgs, _ = train_batches[0]
        bs = real_imgs.size(0)
        real_label = torch.ones(bs, 1, device=device)
        fake_label = torch.zeros(bs, 1, device=device)

        z = torch.randn(bs, latent_dim, device=device)
        fake_imgs = G(z)
        loss_d = (criterion(D(real_imgs), real_label)
                  + criterion(D(fake_imgs.detach()), fake_label))
        optimizer_D.zero_grad()
        loss_d.backward()
        optimizer_D.step()

        z = torch.randn(bs, latent_dim, device=device)
        loss_g = criterion(D(G(z)), real_label)
        optimizer_G.zero_grad()
        loss_g.backward()
        optimizer_G.step()

        G.eval()
        with torch.no_grad():
            _ = G(torch.randn(batch_size, latent_dim, device=device))

        self.dm.sync(device)
        del gan, G, D, optimizer_G, optimizer_D, criterion, loss_d, loss_g
        self.dm.clear_cache(device)

    # ------------------------------------------------------------------ 측정
    def run(self, model_fn, device, train_batches, test_batches,
            num_test_samples, config_name=""):
        """분류 모델 벤치마크 실행

        Returns:
            dict: avg/std train/infer time + accuracy + 반복별 원시값 + 피크 메모리
        """
        train_times = []
        infer_times = []
        accuracies = []
        peak_mb, peak_method = 0.0, 'none'
        self.dm.reset_peak_memory(device)

        for i in range(self.repeats):
            model = model_fn().to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=self.lr)

            # --- 학습 시간 측정 ---
            model.train()
            self.dm.sync(device)
            start = time.perf_counter()

            for _ in range(self.epochs):
                for data, target in train_batches:
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

            self.dm.sync(device)
            train_time = time.perf_counter() - start
            mb, peak_method = self.dm.memory_snapshot_mb(device)
            peak_mb = max(peak_mb, mb)

            # --- 추론 시간 측정 ---
            model.eval()
            self.dm.sync(device)
            start = time.perf_counter()

            correct = 0
            with torch.no_grad():
                for data, target in test_batches:
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()

            self.dm.sync(device)
            infer_time = time.perf_counter() - start
            mb, _ = self.dm.memory_snapshot_mb(device)
            peak_mb = max(peak_mb, mb)
            accuracy = 100. * correct / num_test_samples

            train_times.append(train_time)
            infer_times.append(infer_time)
            accuracies.append(accuracy)

            print(f"    [{i+1}/{self.repeats}] "
                  f"학습: {train_time:.4f}s | "
                  f"추론: {infer_time:.4f}s | "
                  f"정확도: {accuracy:.2f}%")

            del model
            self.dm.clear_cache(device)

        used_tr, avg_tr, std_tr = _stats(train_times, self.discard_first)
        used_inf, avg_inf, std_inf = _stats(infer_times, self.discard_first)
        return {
            'avg_train': avg_tr,
            'std_train': std_tr,
            'avg_infer': avg_inf,
            'std_infer': std_inf,
            'avg_accuracy': round(float(np.mean(accuracies)), 2),
            'train_times': [round(t, 5) for t in train_times],
            'infer_times': [round(t, 5) for t in infer_times],
            'n_repeats': self.repeats,
            'n_used': len(used_tr),
            'std_ddof': 0,
            'peak_mem_mb': round(peak_mb, 1),
            'peak_mem_method': peak_method,
            'protocol': self.PROTOCOL,
        }

    def run_gan(self, model_fn, device, train_batches, batch_size=64,
                config_name=""):
        """GAN 벤치마크 실행

        GAN은 분류가 아닌 생성 모델이므로 별도 루프:
        - 학습: Generator + Discriminator 적대적 학습
        - 추론: Generator만으로 이미지 생성

        Returns:
            dict: avg/std train/infer time (accuracy는 N/A → 0) + 원시값 + 피크 메모리
        """
        train_times = []
        infer_times = []
        peak_mb, peak_method = 0.0, 'none'
        self.dm.reset_peak_memory(device)

        for i in range(self.repeats):
            gan = model_fn().to(device)
            G = gan.generator
            D = gan.discriminator
            latent_dim = gan.latent_dim

            optimizer_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
            optimizer_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
            criterion = nn.BCELoss()

            # --- 학습 시간 측정 (1 epoch) ---
            G.train()
            D.train()
            self.dm.sync(device)
            start = time.perf_counter()

            for real_imgs, _ in train_batches:
                bs = real_imgs.size(0)
                real_label = torch.ones(bs, 1, device=device)
                fake_label = torch.zeros(bs, 1, device=device)

                # Discriminator 학습
                z = torch.randn(bs, latent_dim, device=device)
                fake_imgs = G(z)
                d_real = D(real_imgs)
                d_fake = D(fake_imgs.detach())
                loss_d = criterion(d_real, real_label) + criterion(d_fake, fake_label)

                optimizer_D.zero_grad()
                loss_d.backward()
                optimizer_D.step()

                # Generator 학습
                z = torch.randn(bs, latent_dim, device=device)
                fake_imgs = G(z)
                d_fake = D(fake_imgs)
                loss_g = criterion(d_fake, real_label)

                optimizer_G.zero_grad()
                loss_g.backward()
                optimizer_G.step()

            self.dm.sync(device)
            train_time = time.perf_counter() - start
            mb, peak_method = self.dm.memory_snapshot_mb(device)
            peak_mb = max(peak_mb, mb)

            # --- 추론 시간 측정 (Generator만) ---
            G.eval()
            self.dm.sync(device)
            start = time.perf_counter()

            with torch.no_grad():
                # 학습 배치 수만큼 생성
                for _ in range(len(train_batches)):
                    z = torch.randn(batch_size, latent_dim, device=device)
                    G(z)

            self.dm.sync(device)
            infer_time = time.perf_counter() - start
            mb, _ = self.dm.memory_snapshot_mb(device)
            peak_mb = max(peak_mb, mb)

            train_times.append(train_time)
            infer_times.append(infer_time)

            print(f"    [{i+1}/{self.repeats}] "
                  f"학습: {train_time:.4f}s | "
                  f"추론(생성): {infer_time:.4f}s")

            del gan, G, D
            self.dm.clear_cache(device)

        used_tr, avg_tr, std_tr = _stats(train_times, self.discard_first)
        used_inf, avg_inf, std_inf = _stats(infer_times, self.discard_first)
        return {
            'avg_train': avg_tr,
            'std_train': std_tr,
            'avg_infer': avg_inf,
            'std_infer': std_inf,
            'avg_accuracy': 0.0,  # GAN은 정확도 N/A
            'train_times': [round(t, 5) for t in train_times],
            'infer_times': [round(t, 5) for t in infer_times],
            'n_repeats': self.repeats,
            'n_used': len(used_tr),
            'std_ddof': 0,
            'peak_mem_mb': round(peak_mb, 1),
            'peak_mem_method': peak_method,
            'protocol': self.PROTOCOL,
        }
