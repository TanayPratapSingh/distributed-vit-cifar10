"""Make CIFAR-10 available locally, from the canonical host or a mirror.

Why this exists: www.cs.toronto.edu is unreachable from some networks and
sandboxes. It failed here with an SSL EOF before a single byte of payload
arrived.

What it deliberately does NOT do: reconstruct the original
`cifar-10-batches-py` pickles. Those carry published md5 sums that a
rebuilt file cannot match, so torchvision would reject them as corrupt, and
the only way to make that path "work" would be to disable the integrity
check. Silencing a checksum to make a download appear to succeed is exactly
the kind of shortcut that makes later results untrustworthy.

Instead the mirror writes `data/cifar10-mirror/{train,test}.npz` in its own
format, and `dvit.data` selects it only when the canonical dataset is absent.
Which source a run used is recorded in the run artifact, so a reader can tell.

    python scripts/fetch_cifar10.py --root data
"""

from __future__ import annotations

import argparse
import io
import pickle
import sys
import tarfile
import urllib.request
from pathlib import Path

CANONICAL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
MIRROR = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text"
EXPECTED = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4",
            "data_batch_5", "test_batch", "batches.meta"]
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def already_present(root: Path) -> bool:
    d = root / "cifar-10-batches-py"
    return all((d / f).exists() for f in EXPECTED)


def try_canonical(root: Path, timeout: int = 60) -> bool:
    try:
        print(f"trying canonical source {CANONICAL}")
        with urllib.request.urlopen(CANONICAL, timeout=timeout) as r:
            blob = r.read()
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
            tf.extractall(root)
        print("  canonical source ok")
        return True
    except Exception as e:
        print(f"  canonical source unavailable: {type(e).__name__}: {e}")
        return False


def _load_parquet(url: str, timeout: int):
    import pyarrow.parquet as pq
    print(f"  fetching {url}")
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return pq.read_table(io.BytesIO(r.read()))


def try_mirror(root: Path, timeout: int = 180) -> bool:
    """Write data/cifar10-mirror/{train,test}.npz from the parquet mirror."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError as e:
        print(f"  mirror needs numpy and pillow: {e}")
        return False

    out = root / "cifar10-mirror"
    out.mkdir(parents=True, exist_ok=True)

    for split, fname in [("train", "train-00000-of-00001.parquet"),
                         ("test", "test-00000-of-00001.parquet")]:
        try:
            table = _load_parquet(f"{MIRROR}/{fname}", timeout)
        except Exception as e:
            print(f"  mirror failed on {split}: {type(e).__name__}: {e}")
            return False

        img_col = "img" if "img" in table.column_names else "image"
        images = table.column(img_col).to_pylist()
        labels = np.asarray(table.column("label").to_pylist(), dtype=np.int64)

        # (N, 32, 32, 3) uint8, which is what PIL and the transforms want.
        data = np.zeros((len(images), 32, 32, 3), dtype=np.uint8)
        for i, rec in enumerate(images):
            raw = rec["bytes"] if isinstance(rec, dict) else rec
            data[i] = np.array(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)

        np.savez_compressed(out / f"{split}.npz", data=data, labels=labels)
        print(f"  wrote {split}.npz ({len(data)} images)")

    (out / "SOURCE.txt").write_text(
        "CIFAR-10 rebuilt from the HuggingFace parquet mirror because the\n"
        "canonical host was unreachable. This is NOT the original tarball and\n"
        "does not carry its md5 sums.\n"
        f"mirror: {MIRROR}\n"
    )
    return True


def verify(root: Path) -> bool:
    """Load through the same code path training uses."""
    from dvit.data import load_cifar10, describe_source

    print(f"  source: {describe_source(root)}")
    for train, expect in [(True, 50000), (False, 10000)]:
        ds = load_cifar10(root, train=train)
        if len(ds) != expect:
            print(f"  FAIL: {'train' if train else 'test'} has {len(ds)}, want {expect}")
            return False
        img, label = ds[0]
        if img.size != (32, 32) or not 0 <= int(label) < 10:
            print(f"  FAIL: bad sample {img.size} label={label}")
            return False
    print("  verified through dvit.data.load_cifar10")
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    if already_present(root) and not args.force:
        print(f"CIFAR-10 already present under {root}")
        return 0 if verify(root) else 1

    if not try_canonical(root):
        print("falling back to mirror")
        if not try_mirror(root):
            print("all sources failed", file=sys.stderr)
            return 1

    return 0 if verify(root) else 1


if __name__ == "__main__":
    raise SystemExit(main())
