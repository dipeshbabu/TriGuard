import os
import shutil
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from torchvision.datasets.utils import check_integrity, extract_archive

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_CIFAR_DOWNLOAD_WORKERS = max(1, int(os.environ.get("TRIGUARD_CIFAR_DOWNLOAD_WORKERS", "8")))
_CIFAR_DOWNLOAD_TIMEOUT = float(os.environ.get("TRIGUARD_CIFAR_DOWNLOAD_TIMEOUT", "180"))
_CIFAR_DOWNLOAD_RETRIES = max(1, int(os.environ.get("TRIGUARD_CIFAR_DOWNLOAD_RETRIES", "5")))
_CIFAR_DOWNLOAD_CHUNK_BYTES = max(
    1024 * 1024,
    int(os.environ.get("TRIGUARD_CIFAR_DOWNLOAD_CHUNK_MB", "1")) * 1024 * 1024,
)
_CIFAR_FALLBACK_URLS = {}


def _worker_count(requested: int | None = None) -> int:
    if requested is not None:
        return max(0, requested)
    cpu = os.cpu_count() or 0
    return min(4, cpu)


def _probe_download(url: str) -> tuple[int | None, bool]:
    length = None
    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=_CIFAR_DOWNLOAD_TIMEOUT) as response:
            length_header = response.headers.get("Content-Length")
            length = int(length_header) if length_header else None
            ranges = response.headers.get("Accept-Ranges", "")
            if ranges.lower() == "bytes":
                return length, True
    except (OSError, urllib.error.URLError, ValueError):
        pass

    try:
        request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(request, timeout=_CIFAR_DOWNLOAD_TIMEOUT) as response:
            if response.status == 206:
                content_range = response.headers.get("Content-Range", "")
                if "/" in content_range:
                    length = int(content_range.rsplit("/", 1)[1])
                return length, True
    except (OSError, urllib.error.URLError, ValueError):
        pass
    return length, False


def _candidate_cifar_urls(dataset_cls, env_var: str) -> list[str]:
    urls = []
    env_urls = os.environ.get(env_var, "")
    urls.extend(url.strip() for url in env_urls.split(",") if url.strip())
    urls.extend(_CIFAR_FALLBACK_URLS.get(dataset_cls.filename, []))
    urls.append(dataset_cls.url)
    return list(dict.fromkeys(urls))


def _download_range(url: str, start: int, end: int, path: Path) -> None:
    path.unlink(missing_ok=True)
    last_exc = None
    for _ in range(_CIFAR_DOWNLOAD_RETRIES):
        try:
            request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=_CIFAR_DOWNLOAD_TIMEOUT) as response:
                if response.status != 206:
                    raise RuntimeError("server did not honor ranged download request")
                with path.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
            if path.stat().st_size == (end - start + 1):
                return
            raise RuntimeError(f"incomplete ranged download for bytes {start}-{end}")
        except Exception as exc:
            last_exc = exc
            path.unlink(missing_ok=True)
    raise RuntimeError(f"failed to download bytes {start}-{end}: {last_exc}")


def _download_file(url: str, path: Path) -> None:
    last_exc = None
    for _ in range(_CIFAR_DOWNLOAD_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=_CIFAR_DOWNLOAD_TIMEOUT) as response:
                with path.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
            return
        except Exception as exc:
            last_exc = exc
            path.unlink(missing_ok=True)
    raise RuntimeError(f"failed to download archive: {last_exc}")


def _download_archive(url: str, archive: Path, md5: str | None) -> None:
    size, supports_ranges = _probe_download(url)
    archive.parent.mkdir(parents=True, exist_ok=True)
    tmp_archive = archive.with_name(f"{archive.name}.tmp")

    if supports_ranges and size and _CIFAR_DOWNLOAD_WORKERS > 1:
        part_dir = archive.with_name(f".{archive.name}.parts")
        part_dir.mkdir(parents=True, exist_ok=True)
        chunk_size = _CIFAR_DOWNLOAD_CHUNK_BYTES
        ranges = []
        for start in range(0, size, chunk_size):
            end = min(start + chunk_size - 1, size - 1)
            ranges.append((start, end, part_dir / f"{start}-{end}.part"))

        try:
            with ThreadPoolExecutor(max_workers=min(_CIFAR_DOWNLOAD_WORKERS, len(ranges))) as executor:
                futures = [
                    executor.submit(_download_range, url, start, end, part)
                    for start, end, part in ranges
                ]
                completed = 0
                for future in as_completed(futures):
                    future.result()
                    completed += 1
                    if completed == 1 or completed == len(ranges) or completed % 10 == 0:
                        downloaded_mb = min(completed * chunk_size, size) / (1024 * 1024)
                        total_mb = size / (1024 * 1024)
                        print(
                            f"Downloaded {completed}/{len(ranges)} chunks "
                            f"({downloaded_mb:.1f}/{total_mb:.1f} MB)"
                        )

            with tmp_archive.open("wb") as output:
                for _, _, part in ranges:
                    with part.open("rb") as handle:
                        shutil.copyfileobj(handle, output)
        finally:
            shutil.rmtree(part_dir, ignore_errors=True)
    else:
        _download_file(url, tmp_archive)

    tmp_archive.replace(archive)
    if md5 is not None and not check_integrity(str(archive), md5):
        archive.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded archive failed integrity check: {archive}")


def _check_extracted_cifar(dataset_cls, root: Path) -> bool:
    base = root / dataset_cls.base_folder
    checks = getattr(dataset_cls, "train_list", []) + getattr(dataset_cls, "test_list", [])
    return bool(checks) and all(check_integrity(str(base / name), md5) for name, md5 in checks)


def _prefetch_cifar_archive(dataset_cls, data_root: str, env_var: str) -> None:
    root = Path(data_root)
    archive = root / dataset_cls.filename
    extracted = root / dataset_cls.base_folder

    if extracted.exists() and _check_extracted_cifar(dataset_cls, root):
        return
    if check_integrity(str(archive), dataset_cls.tgz_md5):
        extract_archive(str(archive), str(root))
        return

    for url in _candidate_cifar_urls(dataset_cls, env_var):
        print(
            f"Prefetching {dataset_cls.filename} with {_CIFAR_DOWNLOAD_WORKERS} workers "
            f"from {url}"
        )
        try:
            archive_md5 = dataset_cls.tgz_md5 if url == dataset_cls.url else None
            _download_archive(url, archive, archive_md5)
            extract_archive(str(archive), str(root))
            if not _check_extracted_cifar(dataset_cls, root):
                shutil.rmtree(extracted, ignore_errors=True)
                raise RuntimeError("extracted CIFAR files failed integrity check")
            return
        except Exception as exc:
            print(f"CIFAR prefetch failed from {url}: {exc}")
    print("All CIFAR prefetch attempts failed; falling back to torchvision download.")


def _imagenet_transform(train: bool, grayscale: bool):
    steps = []
    if grayscale:
        steps.append(T.Grayscale(num_output_channels=3))
    if train:
        steps.extend(
            [
                T.RandomResizedCrop(224, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
            ]
        )
    else:
        steps.extend([T.Resize(256), T.CenterCrop(224)])
    steps.extend([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return T.Compose(steps)


def _native_transform(train: bool, mean=None, std=None):
    steps = []
    if train:
        steps.extend([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()])
    steps.append(T.ToTensor())
    if mean is not None and std is not None:
        steps.append(T.Normalize(mean, std))
    return T.Compose(steps)


def _norm_bounds(mean, std):
    # Normalization produces a different valid interval for every channel.
    # Collapsing these to global scalar extrema admits invalid RGB values.
    clamp_min = tuple((0.0 - m) / s for m, s in zip(mean, std))
    clamp_max = tuple((1.0 - m) / s for m, s in zip(mean, std))
    baseline_min, baseline_max = clamp_min, clamp_max
    return clamp_min, clamp_max, baseline_min, baseline_max


def _norm_eps(pixel_eps: float, std):
    return tuple(pixel_eps / s for s in std)


def get_dataset(
    name: str,
    data_root: str = "./data",
    input_profile: str = "native",
):
    name = name.lower()
    input_profile = input_profile.lower()
    if input_profile not in {"native", "imagenet"}:
        raise ValueError(f"Unknown input profile: {input_profile}")
    imagenet_profile = input_profile == "imagenet"
    normalization_mean = (0.0,)
    normalization_std = (1.0,)

    if name == "mnist":
        if imagenet_profile:
            normalization_mean, normalization_std = IMAGENET_MEAN, IMAGENET_STD
            train_tfm = _imagenet_transform(train=True, grayscale=True)
            test_tfm = _imagenet_transform(train=False, grayscale=True)
            clamp_min, clamp_max, baseline_min, baseline_max = _norm_bounds(
                IMAGENET_MEAN, IMAGENET_STD
            )
            eps = _norm_eps(0.3, IMAGENET_STD)
        else:
            train_tfm = test_tfm = T.Compose([T.ToTensor()])
            clamp_min, clamp_max = 0.0, 1.0
            eps = 0.3
            baseline_min, baseline_max = 0.0, 1.0
        train = torchvision.datasets.MNIST(data_root, train=True, download=True, transform=train_tfm)
        test = torchvision.datasets.MNIST(data_root, train=False, download=True, transform=test_tfm)
        num_classes = 10
        pixel_eps = 0.3

    elif name == "fashionmnist":
        if imagenet_profile:
            normalization_mean, normalization_std = IMAGENET_MEAN, IMAGENET_STD
            train_tfm = _imagenet_transform(train=True, grayscale=True)
            test_tfm = _imagenet_transform(train=False, grayscale=True)
            clamp_min, clamp_max, baseline_min, baseline_max = _norm_bounds(
                IMAGENET_MEAN, IMAGENET_STD
            )
            eps = _norm_eps(0.3, IMAGENET_STD)
        else:
            train_tfm = test_tfm = T.Compose([T.ToTensor()])
            clamp_min, clamp_max = 0.0, 1.0
            eps = 0.3
            baseline_min, baseline_max = 0.0, 1.0
        train = torchvision.datasets.FashionMNIST(data_root, train=True, download=True, transform=train_tfm)
        test = torchvision.datasets.FashionMNIST(data_root, train=False, download=True, transform=test_tfm)
        num_classes = 10
        pixel_eps = 0.3

    elif name == "cifar10":
        if imagenet_profile:
            normalization_mean, normalization_std = IMAGENET_MEAN, IMAGENET_STD
            train_tfm = _imagenet_transform(train=True, grayscale=False)
            test_tfm = _imagenet_transform(train=False, grayscale=False)
            clamp_min, clamp_max, baseline_min, baseline_max = _norm_bounds(
                IMAGENET_MEAN, IMAGENET_STD
            )
            eps = _norm_eps(8 / 255, IMAGENET_STD)
        else:
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)
            normalization_mean, normalization_std = mean, std
            train_tfm = _native_transform(train=True, mean=mean, std=std)
            test_tfm = _native_transform(train=False, mean=mean, std=std)
            clamp_min, clamp_max = -1.0, 1.0
            eps = (8 / 255) / 0.5
            baseline_min, baseline_max = -1.0, 1.0
        _prefetch_cifar_archive(torchvision.datasets.CIFAR10, data_root, "TRIGUARD_CIFAR10_URL")
        train = torchvision.datasets.CIFAR10(data_root, train=True, download=True, transform=train_tfm)
        test = torchvision.datasets.CIFAR10(data_root, train=False, download=True, transform=test_tfm)
        num_classes = 10
        pixel_eps = 8 / 255

    elif name == "cifar100":
        if imagenet_profile:
            mean = IMAGENET_MEAN
            std = IMAGENET_STD
            train_tfm = _imagenet_transform(train=True, grayscale=False)
            test_tfm = _imagenet_transform(train=False, grayscale=False)
        else:
            mean = (0.5071, 0.4867, 0.4408)
            std = (0.2675, 0.2565, 0.2761)
            train_tfm = _native_transform(train=True, mean=mean, std=std)
            test_tfm = _native_transform(train=False, mean=mean, std=std)
        normalization_mean, normalization_std = mean, std
        _prefetch_cifar_archive(torchvision.datasets.CIFAR100, data_root, "TRIGUARD_CIFAR100_URL")
        train = torchvision.datasets.CIFAR100(data_root, train=True, download=True, transform=train_tfm)
        test = torchvision.datasets.CIFAR100(data_root, train=False, download=True, transform=test_tfm)
        clamp_min, clamp_max, baseline_min, baseline_max = _norm_bounds(mean, std)
        eps = _norm_eps(8 / 255, std)
        num_classes = 100
        pixel_eps = 8 / 255

    else:
        raise ValueError(f"Unknown dataset: {name}")

    meta = {
        "num_classes": num_classes,
        "baseline_min": baseline_min,
        "baseline_max": baseline_max,
        "input_profile": input_profile,
        "pixel_eps": pixel_eps,
        "normalization_mean": normalization_mean,
        "normalization_std": normalization_std,
    }
    return train, test, clamp_min, clamp_max, eps, meta


def get_loaders(
    name: str,
    batch_size: int,
    test_batch: int = 256,
    num_workers: int | None = None,
    data_root: str = "./data",
    input_profile: str = "native",
    exclude_train_indices=None,
    seed: int = 0,
):
    train, test, clamp_min, clamp_max, eps, meta = get_dataset(
        name,
        data_root=data_root,
        input_profile=input_profile,
    )
    workers = _worker_count(num_workers)
    pin = torch.cuda.is_available()
    excluded = sorted(set(int(index) for index in (exclude_train_indices or [])))
    if excluded:
        if excluded[0] < 0 or excluded[-1] >= len(train):
            raise ValueError("Reserved reference index lies outside the training set.")
        excluded_set = set(excluded)
        train = Subset(
            train, [index for index in range(len(train)) if index not in excluded_set]
        )
    meta["excluded_reference_n"] = len(excluded)
    train_generator = torch.Generator().manual_seed(int(seed))

    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=(workers > 0),
        generator=train_generator,
    )
    test_loader = DataLoader(
        test,
        batch_size=test_batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=(workers > 0),
    )

    return train_loader, test_loader, test, clamp_min, clamp_max, eps, meta
