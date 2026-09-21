"""CIFAR-10 loading with a correct distributed sampler.

Two things here are easy to get wrong and both are worth being able to explain:

1. `set_epoch` must be called on the sampler every epoch or every rank replays
   the same shuffle order for the whole run, which quietly costs accuracy.
2. `drop_last=True` on the training sampler keeps every rank's step count
   identical. Without it a ragged final batch makes ranks disagree on how many
   all reduces to perform and the job hangs on the slowest one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from .dist import DistContext

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@dataclass(frozen=True)
class DataConfig:
    root: str = "data"
    batch_size: int = 128
    num_workers: int = 4
    augment: bool = True
    subset_fraction: float = 1.0


def build_transforms(train: bool, augment: bool):
    norm = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    if train and augment:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([transforms.ToTensor(), norm])


class MirrorCIFAR10(Dataset):
    """CIFAR-10 read from data/cifar10-mirror/*.npz.

    Used only when the canonical torchvision dataset is not present. Returns
    (PIL.Image, int) so it is a drop in for datasets.CIFAR10 under the same
    transforms.
    """

    def __init__(self, root: str | Path, train: bool, transform=None) -> None:
        import numpy as np

        path = Path(root) / "cifar10-mirror" / f"{'train' if train else 'test'}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        blob = np.load(path)
        self.data = blob["data"]          # (N, 32, 32, 3) uint8
        self.labels = blob["labels"]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int):
        from PIL import Image

        img = Image.fromarray(self.data[i])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[i])


def canonical_available(root: str | Path) -> bool:
    try:
        datasets.CIFAR10(str(root), train=True, download=False)
        return True
    except Exception:
        return False


def describe_source(root: str | Path) -> str:
    """Which copy of CIFAR-10 is in use. Recorded in every run artifact."""
    if canonical_available(root):
        return "canonical (cs.toronto.edu tarball, md5 verified by torchvision)"
    if (Path(root) / "cifar10-mirror" / "train.npz").exists():
        return "huggingface parquet mirror (rebuilt, no original md5)"
    return "missing"


def load_cifar10(root: str | Path, train: bool, transform=None) -> Dataset:
    """Canonical dataset when present, mirror otherwise, error if neither."""
    if canonical_available(root):
        return datasets.CIFAR10(str(root), train=train, download=False,
                                transform=transform)
    try:
        return MirrorCIFAR10(root, train=train, transform=transform)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"No CIFAR-10 under {root}. Run: python scripts/fetch_cifar10.py "
            f"--root {root}"
        ) from e


def _maybe_subset(ds: Dataset, fraction: float, seed: int = 0) -> Dataset:
    if fraction >= 1.0:
        return ds
    n = max(1, int(len(ds) * fraction))
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n].tolist()
    return torch.utils.data.Subset(ds, idx)


def build_loaders(cfg: DataConfig, ctx: DistContext) -> tuple[DataLoader, DataLoader, DistributedSampler | None]:
    root = Path(cfg.root)
    root.mkdir(parents=True, exist_ok=True)

    # Nothing downloads at train time. Fetching is an explicit, separate step
    # (scripts/fetch_cifar10.py) so that N ranks cannot race on the same files
    # and so a network failure surfaces before a multi hour job starts.
    train_ds = load_cifar10(root, train=True,
                            transform=build_transforms(True, cfg.augment))
    test_ds = load_cifar10(root, train=False,
                           transform=build_transforms(False, cfg.augment))

    train_ds = _maybe_subset(train_ds, cfg.subset_fraction)
    test_ds = _maybe_subset(test_ds, cfg.subset_fraction)

    train_sampler = None
    test_sampler = None
    if ctx.is_distributed:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=ctx.world_size, rank=ctx.rank,
            shuffle=True, drop_last=True,
        )
        test_sampler = DistributedSampler(
            test_ds, num_replicas=ctx.world_size, rank=ctx.rank,
            shuffle=False, drop_last=False,
        )

    common = dict(
        num_workers=cfg.num_workers,
        pin_memory=(ctx.device.type == "cuda"),
        persistent_workers=cfg.num_workers > 0,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None), drop_last=True, **common,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size * 2, sampler=test_sampler,
        shuffle=False, drop_last=False, **common,
    )
    return train_loader, test_loader, train_sampler
