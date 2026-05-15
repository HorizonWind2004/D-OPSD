from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import torch
from diffusers import ZImagePipeline
from peft import PeftModel


DEFAULT_MODEL = "/mnt/hdfs/jixie/checkpoints/Z-Image/"
DEFAULT_METADATA_URL = (
    "https://raw.githubusercontent.com/HorizonWind2004/reconstruction-alignment/"
    "main/Benchmark/geneval/prompts/evaluation_metadata.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GenEval images with Z-Image baseline or LoRA finetune.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-images-per-prompt", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--lora", default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_metadata(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    urllib.request.urlretrieve(DEFAULT_METADATA_URL, tmp_path)
    tmp_path.replace(path)


def load_metadata(path: Path, limit: int | None) -> list[dict]:
    ensure_metadata(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if limit is not None:
        rows = rows[:limit]
    return rows


def resolve_lora_path(path: str | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    if (candidate / "adapter_config.json").is_file():
        return str(candidate)
    nested = candidate / "recon"
    if (nested / "adapter_config.json").is_file():
        return str(nested)
    raise FileNotFoundError(f"LoRA adapter_config.json not found under {path}")


def load_pipe(args: argparse.Namespace) -> ZImagePipeline:
    pipe = ZImagePipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    lora_path = resolve_lora_path(args.lora)
    if lora_path:
        print(f"loading LoRA: {lora_path}", flush=True)
        peft_transformer = PeftModel.from_pretrained(
            pipe.transformer,
            lora_path,
            adapter_name="recon",
            torch_dtype=torch.bfloat16,
        )
        peft_transformer.set_adapter("recon")
        pipe.transformer = peft_transformer.merge_and_unload()
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.enable_slicing()
    return pipe.to(args.device)


def generate_one(pipe: ZImagePipeline, metadata: dict, prompt_index: int, args: argparse.Namespace) -> None:
    prompt_dir = Path(args.out_dir) / f"{prompt_index:05d}"
    sample_dir = prompt_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "metadata.jsonl").write_text(json.dumps(metadata, ensure_ascii=False) + "\n", encoding="utf-8")

    missing = []
    for image_index in range(args.num_images_per_prompt):
        path = sample_dir / f"{image_index:05d}.png"
        if args.overwrite or not path.exists():
            missing.append((image_index, path))
    if not missing:
        return

    print(f"prompt={prompt_index:05d} generating images={len(missing)}", flush=True)
    for image_index, path in missing:
        generator = torch.Generator(args.device).manual_seed(args.base_seed + prompt_index * 1000 + image_index)
        result = pipe(
            prompt=metadata["prompt"],
            negative_prompt="",
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            cfg_normalization=False,
            generator=generator,
        )
        result.images[0].save(path)
        print(f"saved {path}", flush=True)


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")

    rows = load_metadata(Path(args.metadata), args.limit)
    indexed_rows = [(idx, row) for idx, row in enumerate(rows) if idx % args.num_shards == args.shard_index]
    print(
        f"shard {args.shard_index}/{args.num_shards}: prompts={len(indexed_rows)} "
        f"model={args.model} lora={args.lora} size={args.height}x{args.width} steps={args.steps}",
        flush=True,
    )

    pipe = load_pipe(args)
    done = 0
    for prompt_index, metadata in indexed_rows:
        generate_one(pipe, metadata, prompt_index, args)
        done += 1
        if done % 10 == 0:
            print(f"generated prompts={done}/{len(indexed_rows)} last={prompt_index:05d}", flush=True)
    print(f"finished shard {args.shard_index}: prompts={done}", flush=True)


if __name__ == "__main__":
    main()
