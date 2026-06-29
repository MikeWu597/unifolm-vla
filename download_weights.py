"""
UnifoLM-VLA 模型权重下载脚本

从 HuggingFace Hub 下载推理所需的两个模型权重:
  1. UnifoLM-VLM-Base  -- Qwen2.5-VL 基础模型 (~16GB)
  2. UnifoLM-VLA-LIBERO -- 在 LIBERO 任务上微调的 VLA checkpoint (~16GB)

使用方法:
  python download_weights.py --output-dir /path/to/models

如果网络受限, 可使用镜像:
  HF_ENDPOINT=https://hf-mirror.com python download_weights.py --output-dir /path/to/models
"""

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS = {
    "UnifoLM-VLM-Base": "unitreerobotics/UnifoLM-VLM-Base",
    "UnifoLM-VLA-LIBERO": "unitreerobotics/UnifoLM-VLA-Libero",
}


def download_with_hub(output_dir: Path, use_mirror: bool = False) -> dict[str, Path]:
    """使用 huggingface_hub 下载模型."""
    from huggingface_hub import snapshot_download

    if use_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        # Disable Xet (not supported by mirrors)
        os.environ["HF_HUB_ENABLE_HF_XET"] = "0"
        logger.info("Using HF mirror: https://hf-mirror.com")

    paths = {}
    for name, repo_id in MODELS.items():
        local_path = output_dir / name
        logger.info(f"Downloading {name} from {repo_id} -> {local_path}")

        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_path),
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=1,  # single-thread avoids mirror rate limits
        )
        paths[name] = local_path
        logger.info(f"  Done: {local_path}")

    return paths


def download_with_cli(output_dir: Path, use_mirror: bool = False) -> dict[str, Path]:
    """使用 huggingface-cli 命令行工具下载 (备选方案)."""
    import subprocess

    env = os.environ.copy()
    if use_mirror:
        env["HF_ENDPOINT"] = "https://hf-mirror.com"

    paths = {}
    for name, repo_id in MODELS.items():
        local_path = output_dir / name
        logger.info(f"Downloading {name} from {repo_id} -> {local_path}")

        cmd = [
            "huggingface-cli", "download", repo_id,
            "--local-dir", str(local_path),
            "--resume-download",
        ]
        subprocess.run(cmd, env=env, check=True)
        paths[name] = local_path
        logger.info(f"  Done: {local_path}")

    return paths


def verify_downloads(paths: dict[str, Path]) -> bool:
    """验证下载的完整性."""
    all_ok = True
    for name, path in paths.items():
        if not path.exists():
            logger.error(f"MISSING: {name} at {path}")
            all_ok = False
            continue

        # Check for key files
        total_size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        size_gb = total_size / (1024 ** 3)
        file_count = sum(1 for f in path.rglob("*") if f.is_file())
        logger.info(f"  {name}: {file_count} files, {size_gb:.1f} GB")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Download UnifoLM-VLA model weights")
    parser.add_argument(
        "--output-dir", type=str, default="./models",
        help="Directory to save model weights (default: ./models)"
    )
    parser.add_argument(
        "--use-mirror", action="store_true",
        help="Use hf-mirror.com for faster downloads in China"
    )
    parser.add_argument(
        "--use-cli", action="store_true",
        help="Use huggingface-cli instead of Python API"
    )
    parser.add_argument(
        "--skip", type=str, nargs="*", default=[],
        choices=list(MODELS.keys()),
        help="Models to skip downloading"
    )
    args = parser.parse_args()

    # Remove skipped models
    for skip_name in args.skip:
        MODELS.pop(skip_name, None)
        logger.info(f"Skipping: {skip_name}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Models to download: {list(MODELS.keys())}")
    logger.info(f"Output directory: {output_dir.resolve()}")

    try:
        if args.use_cli:
            paths = download_with_cli(output_dir, args.use_mirror)
        else:
            paths = download_with_hub(output_dir, args.use_mirror)
    except ImportError:
        logger.warning("huggingface_hub not found, falling back to CLI...")
        paths = download_with_cli(output_dir, args.use_mirror)

    logger.info("\nVerifying downloads...")
    ok = verify_downloads(paths)

    if ok:
        logger.info("\nAll models downloaded successfully!")
        logger.info("\nModel paths for run_libero_eval.sh:")
        logger.info(f"  VLM_BASE={paths.get('UnifoLM-VLM-Base', 'N/A')}")
        logger.info(f"  VLA_LIBERO={paths.get('UnifoLM-VLA-LIBERO', 'N/A')}")
        # Find the actual .pt checkpoint inside VLA-LIBERO
        vla_path = paths.get("UnifoLM-VLA-LIBERO")
        if vla_path:
            pt_files = list(vla_path.rglob("*.pt"))
            if pt_files:
                logger.info(f"  Checkpoint: {pt_files[0]}")
    else:
        logger.error("Some downloads may be incomplete. Please re-run.")
        sys.exit(1)


if __name__ == "__main__":
    main()
