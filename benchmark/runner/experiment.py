"""벤치마크 실험 실행기 — 분류 모델 + GAN 모델 지원"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .device import DeviceManager


class ExperimentRunner:
    """단일 모델 설정에 대한 벤치마크 실험 실행

    분류 모델 (ANN, CNN, ResNet, MobileNet, Transformer):
      sync → perf_counter → 학습/추론 → sync → perf_counter

    GAN:
      Generator/Discriminator 적대적 학습 시간 + Generator 추론 시간 측정
    """

    def __init__(self, device_manager, epochs=1, repeats=10, lr=0.01,
                 adaptive_training=False, max_adaptive_epochs=10,
                 target_accuracy=65.0):
        self.dm = device_manager
        self.epochs = epochs
        self.repeats = repeats
        self.lr = lr
        # 선택: 정확도 목표에 도달하거나 max_adaptive_epochs까지 에폭 추가
        self.adaptive_training = adaptive_training
        self.max_adaptive_epochs = max(1, int(max_adaptive_epochs))
        self.target_accuracy = float(target_accuracy)

    def run(self, model_fn, device, train_batches, test_batches,
            num_test_samples, config_name=""):
        """분류 모델 벤치마크 실행

        Returns:
            dict: avg/std train/infer time + accuracy
        """
        train_times = []
        infer_times = []
        accuracies = []
        epochs_per_repeat = []

        for i in range(self.repeats):
            model = model_fn().to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=self.lr)

            # --- 학습 시간 측정 ---
            model.train()
            self.dm.sync(device)
            start = time.perf_counter()

            if self.adaptive_training:
                epochs_done = 0
                accuracy = 0.0
                while epochs_done < self.max_adaptive_epochs:
                    for data, target in train_batches:
                        optimizer.zero_grad()
                        output = model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()
                    epochs_done += 1
                    model.eval()
                    correct = 0
                    with torch.no_grad():
                        for data, target in test_batches:
                            output = model(data)
                            pred = output.argmax(dim=1, keepdim=True)
                            correct += pred.eq(target.view_as(pred)).sum().item()
                    accuracy = 100. * correct / num_test_samples
                    model.train()
                    if accuracy >= self.target_accuracy:
                        break
                epochs_per_repeat.append(epochs_done)
            else:
                for _ in range(self.epochs):
                    for data, target in train_batches:
                        optimizer.zero_grad()
                        output = model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()
                epochs_per_repeat.append(self.epochs)

            self.dm.sync(device)
            train_time = time.perf_counter() - start

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
            accuracy = 100. * correct / num_test_samples

            train_times.append(train_time)
            infer_times.append(infer_time)
            accuracies.append(accuracy)

            ep_info = ""
            if self.adaptive_training:
                ep_info = f" | 에폭: {epochs_per_repeat[-1]}"

            print(f"    [{i+1}/{self.repeats}] "
                  f"학습: {train_time:.4f}s | "
                  f"추론: {infer_time:.4f}s | "
                  f"정확도: {accuracy:.2f}%{ep_info}")

            del model
            self.dm.clear_cache(device)

        out = {
            'avg_train': round(float(np.mean(train_times)), 5),
            'std_train': round(float(np.std(train_times)), 5),
            'avg_infer': round(float(np.mean(infer_times)), 5),
            'std_infer': round(float(np.std(infer_times)), 5),
            'avg_accuracy': round(float(np.mean(accuracies)), 2),
        }
        if self.adaptive_training:
            out['avg_benchmark_epochs'] = round(float(np.mean(epochs_per_repeat)), 2)
            out['std_benchmark_epochs'] = round(float(np.std(epochs_per_repeat)), 2)
            out['benchmark_adaptive'] = True
            out['benchmark_target_accuracy'] = self.target_accuracy
            out['benchmark_max_epochs'] = self.max_adaptive_epochs
        else:
            out['avg_benchmark_epochs'] = float(self.epochs)
            out['std_benchmark_epochs'] = 0.0
            out['benchmark_adaptive'] = False
        return out

    def run_gan(self, model_fn, device, train_batches, batch_size=64,
                config_name=""):
        """GAN 벤치마크 실행

        GAN은 분류가 아닌 생성 모델이므로 별도 루프:
        - 학습: Generator + Discriminator 적대적 학습
        - 추론: Generator만으로 이미지 생성

        Returns:
            dict: avg/std train/infer time (accuracy는 N/A → 0)
        """
        train_times = []
        infer_times = []

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

            # --- 추론 시간 측정 (Generator만) ---
            G.eval()
            self.dm.sync(device)
            start = time.perf_counter()

            with torch.no_grad():
                # 테스트 배치 수만큼 생성
                for _ in range(len(train_batches)):
                    z = torch.randn(batch_size, latent_dim, device=device)
                    G(z)

            self.dm.sync(device)
            infer_time = time.perf_counter() - start

            train_times.append(train_time)
            infer_times.append(infer_time)

            print(f"    [{i+1}/{self.repeats}] "
                  f"학습: {train_time:.4f}s | "
                  f"추론(생성): {infer_time:.4f}s")

            del gan, G, D
            self.dm.clear_cache(device)

        return {
            'avg_train': round(float(np.mean(train_times)), 5),
            'std_train': round(float(np.std(train_times)), 5),
            'avg_infer': round(float(np.mean(infer_times)), 5),
            'std_infer': round(float(np.std(infer_times)), 5),
            'avg_accuracy': 0.0,  # GAN은 정확도 N/A
        }
