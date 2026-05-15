import argparse
import importlib.util
import json
import logging
import math
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import ZImagePipeline
from diffusers.utils.torch_utils import is_compiled_module
from peft import LoraConfig, get_peft_model
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.utils import make_grid
from tqdm.auto import tqdm

from utils import _encode_prompt
from vlm_utils import get_qwen3vl_zimage_prompt_embeds, load_matching_state_dict


os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Z-Image reconstruction LoRA from Qwen3-VL image-conditioned embeds.")
    parser.add_argument("--pretrained-model", default="/mnt/hdfs/jixie/checkpoints/Z-Image/")
    parser.add_argument("--qwen3vl-name", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--image-root", default="/mnt/hdfs/jixie/RGB_stage1/images/")
    parser.add_argument("--output-dir", default="/mnt/hdfs/jixie/checkpoints/zimage_reconstruction_train/")
    parser.add_argument("--exp-name", required=True)

    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--vl-resolution", type=int, default=224)
    parser.add_argument("--image-list", default=None)
    parser.add_argument("--image-count", type=int, default=100000)
    parser.add_argument("--image-name-template", default="{index:08d}.png")
    parser.add_argument("--max-image-load-retries", type=int, default=1000)
    parser.add_argument("--refresh-image-list", action="store_true")
    parser.add_argument("--recursive-images", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=100000)

    parser.add_argument("--max-train-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--vae-dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--enable-gc", action="store_true")
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--log-steps", type=int, default=1)
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument("--checkpoint-steps", type=int, default=500)
    parser.add_argument("--sample-num-images", type=int, default=4)
    parser.add_argument("--sample-inference-steps", type=int, default=28)
    parser.add_argument("--sample-guidance-scale", type=float, default=4.0)
    parser.add_argument(
        "--t2i-validation-prompts",
        nargs="*",
        default=[
            "A close-up portrait photo of a young woman in soft natural light, detailed skin texture, realistic colors.",
            "A small red sports car parked on a wet city street at night, cinematic reflections.",
            "A cozy wooden cabin beside a lake, mountains in the background, sunrise lighting.",
            "A bowl of fresh fruit on a kitchen table, realistic photography, detailed textures.",
        ],
    )
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--min-pixels", type=int, default=224 * 224)
    parser.add_argument("--max-pixels", type=int, default=224 * 224)
    parser.add_argument("--reca-prompts-path", default="/opt/tiger/why-reca/train/src/qflux/data/reca_prompts.py")
    parser.add_argument("--max-reca-prompts", type=int, default=360)
    parser.add_argument(
        "--reconstruction-prompt",
        default="Reconstruct the input image as faithfully as possible, preserving the main subject, layout, colors, and visual style.",
    )
    return parser.parse_args()


def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    return model._orig_mod if is_compiled_module(model) else model


def make_logger(save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(save_dir / "log.txt")],
    )
    return logging.getLogger(__name__)


def load_reca_prompts(path: str, max_prompts: int | None):
    spec = importlib.util.spec_from_file_location("reca_prompts", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load prompt file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("RECA_RECON_PROMPTS", "RECA_PROMPTS", "prompts"):
        prompts = getattr(module, name, None)
        if prompts:
            prompts = list(prompts)
            return prompts[:max_prompts] if max_prompts else prompts
    raise ValueError(f"No prompt list found in {path}")


def build_indexed_paths(image_root: Path, image_count: int, image_name_template: str):
    return [image_root / image_name_template.format(index=i) for i in range(image_count)]


def list_image_paths(image_root: Path, image_list: str | None, image_count: int, image_name_template: str, recursive: bool):
    if image_list:
        path = Path(image_list)
        return [Path(line.strip()) for line in path.read_text().splitlines() if line.strip()]
    if not recursive:
        return build_indexed_paths(image_root, image_count, image_name_template)
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    paths = []
    for pattern in patterns:
        paths.extend(image_root.rglob(pattern))
    return sorted(paths)


def center_crop_resize(image: Image.Image, size: int):
    w, h = image.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    image = image.crop((left, top, left + side, top + side))
    return image.resize((size, size), Image.Resampling.BICUBIC)


class ImageFolderReconstructionDataset(Dataset):
    def __init__(
        self,
        image_root: str,
        resolution: int,
        vl_resolution: int,
        prompts: list[str],
        max_samples: int | None,
        seed: int,
        image_list: str | None,
        image_count: int,
        image_name_template: str,
        recursive_images: bool,
        max_image_load_retries: int,
    ):
        self.image_root = Path(image_root)
        self.paths = list_image_paths(self.image_root, image_list, image_count, image_name_template, recursive_images)
        rng = random.Random(seed)
        rng.shuffle(self.paths)
        if max_samples is not None:
            self.paths = self.paths[:max_samples]
        if not self.paths:
            raise ValueError(f"No images found under {image_root}")
        self.resolution = resolution
        self.vl_resolution = vl_resolution
        self.prompts = prompts
        self.max_image_load_retries = max_image_load_retries

    def __len__(self):
        return len(self.paths)

    def _load_one(self, index: int):
        path = self.paths[index % len(self.paths)]
        image = Image.open(path).convert("RGB")
        target = center_crop_resize(image, self.resolution)
        vl_image = center_crop_resize(image, self.vl_resolution)
        pixel_values = TF.to_tensor(target) * 2 - 1
        prompt = self.prompts[index % len(self.prompts)]
        return {
            "pixel_values": pixel_values,
            "vl_image": vl_image,
            "target_pil": target,
            "prompt": prompt,
            "path": str(path),
        }

    def __getitem__(self, index: int):
        for retry in range(self.max_image_load_retries + 1):
            try:
                return self._load_one(index + retry)
            except (UnidentifiedImageError, OSError, FileNotFoundError) as exc:
                if retry >= self.max_image_load_retries:
                    raise
                print(f"[warn] skip bad image {self.paths[(index + retry) % len(self.paths)]}: {exc}", flush=True)
        raise RuntimeError("unreachable")


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "vl_images": [item["vl_image"] for item in batch],
        "target_pils": [item["target_pil"] for item in batch],
        "prompts": [item["prompt"] for item in batch],
        "paths": [item["path"] for item in batch],
    }


def decode_latents(pipeline, latents):
    latents = latents.to(device=pipeline.vae.device, dtype=pipeline.vae.dtype)
    latents = (latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    images = pipeline.vae.decode(latents, return_dict=False)[0]
    return (images / 2 + 0.5).clamp(0, 1)


def tensor_to_pil(tensor):
    tensor = tensor.detach().cpu().clamp(0, 1)
    return TF.to_pil_image(tensor)


def add_label(image: Image.Image, label: str):
    label_h = 24
    canvas = Image.new("RGB", (image.width, image.height + label_h), "white")
    canvas.paste(image, (0, label_h))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((4, 4), label, fill="black", font=font)
    return canvas


def save_image_grid(images: list[Image.Image], save_path: Path, nrow: int):
    if not images:
        return
    w, h = images[0].size
    rows = math.ceil(len(images) / nrow)
    grid = Image.new("RGB", (nrow * w, rows * h), "white")
    for i, image in enumerate(images):
        grid.paste(image, ((i % nrow) * w, (i // nrow) * h))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(save_path)


@torch.no_grad()
def sample_reconstructions(args, pipeline, vl_model, processor, batch, save_dir: Path, step: int, accelerator, dtype):
    pipeline.transformer.eval()
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        accelerator.wait_for_everyone()
        pipeline.transformer.train()
        return

    wrapped_transformer = pipeline.transformer
    pipeline.transformer = unwrap_model(wrapped_transformer, accelerator)
    images = batch["target_pils"][: args.sample_num_images]
    vl_images = batch["vl_images"][: args.sample_num_images]
    prompts = batch["prompts"][: args.sample_num_images]
    prompt_embeds = get_qwen3vl_zimage_prompt_embeds(
        vl_model=vl_model,
        processor=processor,
        prompts=prompts,
        images=vl_images,
        device=accelerator.device,
        dtype=dtype,
        max_sequence_length=args.max_sequence_length,
        hidden_state_layer=-2,
        use_system_prompt=False,
    )
    generators = [torch.Generator(accelerator.device).manual_seed(args.seed + step * 1000 + i) for i in range(len(images))]
    negative_prompt_embeds = None
    if args.sample_guidance_scale > 0:
        negative_prompt_embeds = _encode_prompt(
            pipeline.text_encoder,
            pipeline.tokenizer,
            ["" for _ in prompt_embeds],
            device=accelerator.device,
            max_sequence_length=args.max_sequence_length,
        )
    recon = pipeline(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        height=args.resolution,
        width=args.resolution,
        num_inference_steps=args.sample_inference_steps,
        guidance_scale=args.sample_guidance_scale,
        cfg_normalization=False,
        generator=generators,
    ).images

    labeled = []
    for target, rec in zip(images, recon):
        labeled.append(add_label(target, "target"))
        labeled.append(add_label(rec.resize(target.size), "recon"))
    save_image_grid(labeled, save_dir / "samples" / f"step_{step:06d}.png", nrow=2)
    (save_dir / "samples" / f"step_{step:06d}.txt").write_text("\n".join(prompts), encoding="utf-8")

    pipeline.transformer = wrapped_transformer
    accelerator.wait_for_everyone()
    pipeline.transformer.train()


@torch.no_grad()
def sample_t2i(args, pipeline, save_dir: Path, step: int, accelerator):
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        accelerator.wait_for_everyone()
        return
    wrapped_transformer = pipeline.transformer
    pipeline.transformer = unwrap_model(wrapped_transformer, accelerator)
    prompts = args.t2i_validation_prompts
    generators = [torch.Generator(accelerator.device).manual_seed(args.seed + step * 100 + i) for i in range(len(prompts))]
    images = pipeline(
        prompt=prompts,
        height=args.resolution,
        width=args.resolution,
        num_inference_steps=args.sample_inference_steps,
        guidance_scale=args.sample_guidance_scale,
        cfg_normalization=False,
        generator=generators,
    ).images
    save_image_grid([add_label(img, f"t2i {i}") for i, img in enumerate(images)], save_dir / "samples_t2i" / f"step_{step:06d}.png", nrow=2)
    (save_dir / "samples_t2i" / f"step_{step:06d}.txt").write_text("\n".join(prompts), encoding="utf-8")
    pipeline.transformer = wrapped_transformer
    accelerator.wait_for_everyone()


def main():
    args = parse_args()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=ProjectConfiguration(project_dir=args.output_dir),
    )
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    save_dir = Path(args.output_dir) / args.exp_name
    checkpoint_dir = save_dir / "checkpoints"
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
        local_logger = make_logger(save_dir)
        local_logger.info(f"Experiment directory: {save_dir}")
    accelerator.wait_for_everyone()

    inference_dtype = torch.float32
    if args.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16
    vae_dtype = torch.float32 if args.vae_dtype == "fp32" else inference_dtype

    pipeline = ZImagePipeline.from_pretrained(args.pretrained_model, low_cpu_mem_usage=False)
    num_channels_latents = pipeline.transformer.in_channels
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.transformer.requires_grad_(False)
    pipeline.vae.to(accelerator.device, dtype=vae_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.vae.enable_slicing()
    pipeline.set_progress_bar_config(disable=True)
    tokenizer = pipeline.tokenizer

    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.qwen3vl_name, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    vl_model = AutoModelForImageTextToText.from_pretrained(args.qwen3vl_name)
    load_matching_state_dict(vl_model.model.language_model, pipeline.text_encoder.state_dict(), verbose=accelerator.is_main_process)
    vl_model.requires_grad_(False)
    vl_model.to(accelerator.device, dtype=inference_dtype)
    vl_model.eval()

    target_modules = [
        "feed_forward.w1",
        "feed_forward.w2",
        "feed_forward.w3",
        "attention.to_k",
        "attention.to_q",
        "attention.to_v",
        "attention.to_out.0",
    ]
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    pipeline.transformer = get_peft_model(pipeline.transformer, lora_config, adapter_name="recon")
    pipeline.transformer.set_adapter("recon")
    pipeline.transformer.to(accelerator.device, dtype=inference_dtype)
    if args.enable_gc:
        pipeline.transformer.enable_gradient_checkpointing()

    prompts = load_reca_prompts(args.reca_prompts_path, args.max_reca_prompts)
    dataset = ImageFolderReconstructionDataset(
        image_root=args.image_root,
        resolution=args.resolution,
        vl_resolution=args.vl_resolution,
        prompts=prompts,
        max_samples=args.max_train_samples,
        seed=args.seed,
        image_list=args.image_list,
        image_count=args.image_count,
        image_name_template=args.image_name_template,
        recursive_images=args.recursive_images,
        max_image_load_retries=args.max_image_load_retries,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    trainable_params = [p for p in pipeline.transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    pipeline.transformer, optimizer, dataloader = accelerator.prepare(pipeline.transformer, optimizer, dataloader)

    if accelerator.is_main_process:
        logger.info(f"Dataset images: {len(dataset)}")
        logger.info(f"Prompts: {len(prompts)}")
        logger.info(f"Total batch size: {args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")
        logger.info(f"Learning rate: {args.learning_rate}")
        logger.info(f"Trainable params: {sum(p.numel() for p in trainable_params)}")

    global_step = 0
    progress = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process, desc="steps")
    last_batch = None
    while global_step < args.max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(pipeline.transformer):
                images = batch["pixel_values"].to(device=accelerator.device, dtype=vae_dtype)
                bsz, _, h, w = images.shape
                with torch.no_grad():
                    with accelerator.autocast():
                        latents = pipeline.vae.encode(images).latent_dist.mode()
                        latents = (latents - pipeline.vae.config.shift_factor) * pipeline.vae.config.scaling_factor
                        prompt_embeds = get_qwen3vl_zimage_prompt_embeds(
                            vl_model=vl_model,
                            processor=processor,
                            prompts=batch["prompts"],
                            images=batch["vl_images"],
                            device=accelerator.device,
                            dtype=inference_dtype,
                            max_sequence_length=args.max_sequence_length,
                            hidden_state_layer=-2,
                            use_system_prompt=False,
                        )

                noise = torch.randn_like(latents)
                t = torch.rand((bsz,), device=accelerator.device, dtype=inference_dtype)
                noisy_latents = (1 - t[:, None, None, None]) * noise + t[:, None, None, None] * latents
                target = latents - noise
                model_input = list(noisy_latents.unsqueeze(2).unbind(dim=0))

                with accelerator.autocast():
                    pred = pipeline.transformer(model_input, t, prompt_embeds, return_dict=False)[0]
                    pred = torch.stack(pred, dim=0).squeeze(2)
                    loss = F.mse_loss(pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(pipeline.transformer.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                last_batch = batch
                gathered_loss = accelerator.gather(loss.detach()).mean().item()
                progress.set_postfix(loss=f"{gathered_loss:.6f}")
                if accelerator.is_main_process and global_step % args.log_steps == 0:
                    print(f"step={global_step} loss={gathered_loss:.6f}", flush=True)

                if global_step % args.checkpoint_steps == 0 or global_step == args.max_train_steps:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        ckpt = checkpoint_dir / f"step_{global_step:06d}" / "recon"
                        ckpt.parent.mkdir(parents=True, exist_ok=True)
                        unwrap_model(pipeline.transformer, accelerator).save_pretrained(ckpt)
                        print(f"saved checkpoint {ckpt}", flush=True)

                if global_step % args.sample_steps == 0 and last_batch is not None:
                    sample_reconstructions(args, pipeline, vl_model, processor, last_batch, save_dir, global_step, accelerator, inference_dtype)
                    sample_t2i(args, pipeline, save_dir, global_step, accelerator)

                if global_step >= args.max_train_steps:
                    break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = checkpoint_dir / "final" / "recon"
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        unwrap_model(pipeline.transformer, accelerator).save_pretrained(final_dir)
        print(f"saved final checkpoint {final_dir}", flush=True)


if __name__ == "__main__":
    main()
