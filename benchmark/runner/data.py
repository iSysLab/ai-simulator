"""데이터셋 관리: MNIST + CIFAR-10 로딩, 장치별 사전 전송"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class MNISTDataManager:
    """MNIST 데이터셋 관리"""

    def __init__(self, data_root='./data', batch_size=64, test_batch_size=1000):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        print("MNIST 데이터 다운로드 및 로딩 중...")
        self.train_dataset = datasets.MNIST(
            root=data_root, train=True, download=True, transform=transform)
        self.test_dataset = datasets.MNIST(
            root=data_root, train=False, download=True, transform=transform)

        self.train_loader = DataLoader(
            dataset=self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(
            dataset=self.test_dataset, batch_size=test_batch_size, shuffle=False)

        self.num_test_samples = len(self.test_dataset)
        self.input_shape = (1, 1, 28, 28)
        print("MNIST 로딩 완료.\n")

    def preload_to_device(self, device):
        """배치를 장치 메모리에 사전 로딩"""
        train_batches = [(d.to(device), t.to(device))
                         for d, t in self.train_loader]
        test_batches = [(d.to(device), t.to(device))
                        for d, t in self.test_loader]
        return train_batches, test_batches


class CIFAR10DataManager:
    """CIFAR-10 데이터셋 관리 (Transformer, GAN 벤치마크용)"""

    def __init__(self, data_root='./data', batch_size=64, test_batch_size=1000):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616))
        ])

        print("CIFAR-10 데이터 다운로드 및 로딩 중...")
        self.train_dataset = datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=transform)
        self.test_dataset = datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=transform)

        self.train_loader = DataLoader(
            dataset=self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(
            dataset=self.test_dataset, batch_size=test_batch_size, shuffle=False)

        self.num_test_samples = len(self.test_dataset)
        self.input_shape = (1, 3, 32, 32)
        print("CIFAR-10 로딩 완료.\n")

    def preload_to_device(self, device):
        """배치를 장치 메모리에 사전 로딩"""
        train_batches = [(d.to(device), t.to(device))
                         for d, t in self.train_loader]
        test_batches = [(d.to(device), t.to(device))
                        for d, t in self.test_loader]
        return train_batches, test_batches
