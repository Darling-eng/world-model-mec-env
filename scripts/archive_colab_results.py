from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


RESULT_PATTERNS = ("metrics.jsonl", "scores.jsonl", "*.csv")


def iter_result_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in RESULT_PATTERNS:
        files.extend(path for path in root.rglob(pattern) if path.is_file())
    return sorted(set(files))


def copy_tree(src_root: Path, dst_root: Path) -> list[dict[str, str | int]]:
    manifest = []
    for src in iter_result_files(src_root):
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        manifest.append(
            {
                "relative_path": str(rel).replace("\\", "/"),
                "bytes": dst.stat().st_size,
                "source": str(src),
                "archived_to": str(dst),
            }
        )
    return manifest


def write_manifest(dst_root: Path, manifest: list[dict[str, str | int]], src_root: Path) -> Path:
    manifest_path = dst_root / "manifest.json"
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(src_root),
        "file_count": len(manifest),
        "files": manifest,
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive Colab MEC experiment result files into a persistent folder with a manifest."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("/content/world-model-mec-env/colab_results"),
        help="Colab result root containing metrics.jsonl, scores.jsonl, or CSV files.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        required=True,
        help="Persistent destination, for example /content/drive/MyDrive/mec_results/20260628.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.src.exists():
        raise FileNotFoundError(f"Result source does not exist: {args.src}")
    args.dst.mkdir(parents=True, exist_ok=True)
    manifest = copy_tree(args.src, args.dst)
    manifest_path = write_manifest(args.dst, manifest, args.src)
    print(f"archived_files={len(manifest)}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
