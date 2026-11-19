"""Download the pinned CPython 3.13 Linux x86_64 CPU PyTorch wheel in ranges.

The PyTorch CDN can terminate large single-stream downloads in constrained
build environments. Parallel byte ranges make the Docker build resumable at the
chunk level; the pinned SHA-256 verifies the official artifact before install.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import platform
import time
import urllib.request
from pathlib import Path


WHEEL_URL = (
    "https://download.pytorch.org/whl/cpu/"
    "torch-2.12.0%2Bcpu-cp313-cp313-manylinux_2_28_x86_64.whl"
)
WHEEL_SIZE = 192_271_968
WHEEL_SHA256 = "ada78018bdfa30d1c766596cd32d910dbf5b03424cd859231b6d2a00533de922"
CHUNK_SIZE = 1024 * 1024
MAX_WORKERS = 64


def _download_range(start: int, end: int) -> tuple[int, bytes]:
    last_error: Exception | None = None
    for attempt in range(5):
        request = urllib.request.Request(
            WHEEL_URL,
            headers={
                "Accept-Encoding": "identity",
                "Range": f"bytes={start}-{end}",
                "User-Agent": "BizInsight-Agent-build/1.0",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
                content_range = response.headers.get("Content-Range", "")
                expected_range = f"bytes {start}-{end}/{WHEEL_SIZE}"
                if (
                    response.status != 206
                    or content_range != expected_range
                    or len(data) != end - start + 1
                ):
                    raise RuntimeError(
                        f"Unexpected range response for {start}-{end}: "
                        f"status={response.status}, content_range={content_range!r}, "
                        f"bytes={len(data)}"
                    )
                return start, data
        except Exception as exc:  # transient CDN resets are retried per chunk
            last_error = exc
            time.sleep(min(attempt + 1, 5))
    raise RuntimeError(f"Could not download wheel range {start}-{end}: {last_error}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as wheel:
        for block in iter(lambda: wheel.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_wheel(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    ranges = [
        (start, min(start + CHUNK_SIZE, WHEEL_SIZE) - 1)
        for start in range(0, WHEEL_SIZE, CHUNK_SIZE)
    ]
    downloaded = 0
    next_report = 16 * 1024 * 1024
    started_at = time.perf_counter()

    with destination.open("wb") as wheel:
        wheel.truncate(WHEEL_SIZE)
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = [pool.submit(_download_range, start, end) for start, end in ranges]
            for future in concurrent.futures.as_completed(futures):
                start, data = future.result()
                wheel.seek(start)
                wheel.write(data)
                downloaded += len(data)
                if downloaded >= next_report or downloaded == WHEEL_SIZE:
                    elapsed = max(time.perf_counter() - started_at, 0.001)
                    rate = downloaded / elapsed / 1024 / 1024
                    print(
                        f"PyTorch CPU wheel: {downloaded}/{WHEEL_SIZE} bytes "
                        f"({rate:.2f} MiB/s)",
                        flush=True,
                    )
                    next_report = downloaded + 16 * 1024 * 1024

    actual_sha256 = _sha256(destination)
    if actual_sha256 != WHEEL_SHA256:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"PyTorch CPU wheel SHA-256 mismatch: expected {WHEEL_SHA256}, "
            f"got {actual_sha256}"
        )
    print(f"PyTorch CPU wheel verified: sha256={actual_sha256}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        raise SystemExit("This pinned PyTorch wheel supports Linux x86_64 only.")
    download_wheel(args.output)


if __name__ == "__main__":
    main()
