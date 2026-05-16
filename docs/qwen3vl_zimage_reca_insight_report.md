# Qwen3-VL Connector + Z-Image Reconstruction Alignment 实验记录

本文档记录一次围绕 “Qwen3-VL 的视觉编码器/connector 是否可以直接为 Z-Image 提供视觉条件” 的复现实验。文档面向后续论文写作，尽量保留从最初动机、实验设计、实现细节、失败路径到最终定量结果的完整链条。

## 1. 出发点与核心问题

最初的动机来自一篇关于 Reconstruction Alignment / D-OPSD 的文章及其 README 中的 insight：一个原本没有视觉输入能力的文本生成/图像生成底座，可能只需要接入同族 VLM 的视觉塔与 connector，就能在不显式重新训练完整多模态模型的情况下得到可用的视觉条件表示。更具体地说，论文中的现象可以概括为：

1. Qwen3 文本 LLM 本身没有图像理解能力。
2. Qwen3-VL 相比 Qwen3 多了视觉 ViT 与 connector，并且语言模型部分与 Qwen3/Z-Image 的 text encoder 高度同构。
3. 如果把 Qwen3-VL 的 ViT + connector 产生的多模态 hidden states 接到 Z-Image DiT，本来只吃文本 hidden states 的 Z-Image 可能会把这些 hidden states 当作一种“视觉 prompt”使用。
4. 这种结构如果成立，至少应该出现两个现象：一是 hidden states 中保留可读的视觉语义；二是它可以驱动 Z-Image 对输入图像做重建，甚至通过少量 reconstruction LoRA 训练提升生成 benchmark。

因此本轮实验从两个问题开始：

- **Zero-shot 视觉理解能力**：Qwen3 + Qwen3-VL ViT/connector 的拼接表示，是否天然带有图像语义？最初希望用 image captioning/VQA 风格的 probe 来验证。
- **重建能力**：将 Qwen3-VL image-conditioned hidden states 喂给 `/mnt/hdfs/jixie/checkpoints/Z-Image/` 的 DiT 后，是否可以重建输入图像？如果 zero-training 不够，LoRA reconstruction training 是否能改善 Z-Image 的文本生成能力和 Geneval score？

后续实验逐步从定性 probe 走向定量 Geneval：先看 caption 与可视化，再用 RGB_stage1 的 Z-Image 生成图训练 reconstruction LoRA，最后用 Geneval 对 baseline、500 step、1000 step 与不同分辨率/步数配置做对比。

## 2. 模型与实现假设

### 2.1 组件关系

本实验涉及三个核心模块：

- **Z-Image base**：路径为 `/mnt/hdfs/jixie/checkpoints/Z-Image/`。它包含 VAE、DiT/transformer、Qwen3 风格 text encoder 与 tokenizer。
- **Qwen3-VL-4B-Instruct**：作为视觉输入端，使用其视觉塔、connector 以及语言模型 forward 产生 image+text hidden states。
- **Reconstruction LoRA**：只训练 Z-Image transformer 上的 LoRA，不训练 VAE、Qwen3 text encoder、Qwen3-VL。

关键工程假设是：Z-Image 的 text encoder 与 Qwen3-VL 的 language model 结构兼容，可以把 Z-Image text encoder 的权重加载到 Qwen3-VL language model 中，使 Qwen3-VL 的视觉 connector 输出落到 Z-Image DiT 习惯的 hidden-state 空间附近。实验中 `load_matching_state_dict()` 对 Qwen3-VL language model 加载 Z-Image text encoder 参数时，出现过 `Matched keys: 398 / 398`，说明语言模型部分在参数名和 shape 上完全可匹配。

### 2.2 Hidden states 接入方式

在 probe 和训练脚本中，多模态 hidden states 的生成逻辑大致是：

1. 对每张图构造 Qwen3-VL chat template，内容包含 image 与 text prompt。
2. 使用 Qwen3-VL processor 将图文转为模型输入。
3. 前向 Qwen3-VL，取倒数第二层 hidden state。
4. 根据 attention mask 去掉 padding，得到每个样本变长 hidden state list。
5. 将该 list 作为 `prompt_embeds` 直接传入 Z-Image pipeline 或 Z-Image transformer。

训练时的 text prompt 不是随意 caption，而是使用 ReCA 的 360 条 reconstruction prompts：

```text
/opt/tiger/why-reca/train/src/qflux/data/reca_prompts.py
```

这样做的理由是：如果 image-conditioned hidden states 已经有视觉内容，那么 prompt 只需要稳定地要求模型“忠实重建输入图像”，而不是过度依赖外部 caption。

## 3. Zero-shot 视觉理解 probe

### 3.1 初始目标

最初用户明确要求“先做视觉理解测试”，而不是直接训练。我们实现并运行了一个 probe，目标是比较三种输出：

- **Original VLM caption**：Qwen3-VL 原生图文模型直接 caption。
- **Hybrid caption**：把 Z-Image/Qwen3 text encoder 权重加载进 Qwen3-VL language model 后，再用同一个图像输入生成 caption。
- **Text-only control**：不给图像，只给“描述图像”的文本指令，作为无视觉输入对照。

输出目录：

```text
/mnt/hdfs/jixie/checkpoints/qwen3vl_zimage_insight_probe/
```

典型 caption 文件包括：

```text
/mnt/hdfs/jixie/checkpoints/qwen3vl_zimage_insight_probe/caption_only_v3/captions.jsonl
/mnt/hdfs/jixie/checkpoints/qwen3vl_zimage_insight_probe/reca_recon_smoke_cfg4/captions.jsonl
/mnt/hdfs/jixie/checkpoints/qwen3vl_zimage_insight_probe/smoke/captions.jsonl
```

### 3.2 定性观察

在 `caption_only_v3` 的 4 个样例上，Original VLM caption 非常正常，能识别主体、背景与风格。例如：

- 白马 + 雪地森林 + 牵马男子，被描述为 “A man in a teal coat guides a white horse through a snowy forest, rendered in a vibrant, neon-pink and electric-blue pop-art style.”
- 手袋、甜品、咖啡杯、腊肠犬手机壳，被描述为 “A vibrant, neon-colored still life features a brown handbag with a plush dog keychain, a flan dessert, a coffee cup, and a phone with a dachshund case...”
- 哈士奇小狗，被描述为 neon/holographic style 的厨房场景。
- 旗袍女子肖像，被描述为 glitch-art 背景的 stylized portrait。

Hybrid caption 的结果更复杂：它明显获得了图像中的对象、颜色、风格信息，但语言输出经常混入 “Okay, the user wants...” 这类思维链/解释性文本，甚至重复词。这说明：

- 视觉信息确实通过 Qwen3-VL ViT + connector 进入了 hidden states。
- 但把 Z-Image text encoder 权重直接塞进 Qwen3-VL language model 后，作为语言生成器并不稳定，不适合直接当 captioning 模型使用。
- Hybrid caption 的失败主要是语言解码行为不稳，而不是视觉信息完全不存在。

Text-only control 基本只会围绕“用户要求我描述一张图片”这件事展开，无法给出具体图像内容。因此它和 hybrid caption 形成了对照：hybrid 至少能提到白马、雪地、霓虹、包、狗、甜品、哈士奇、旗袍、glitch 背景等实际视觉元素。

### 3.3 视觉理解阶段结论

这个阶段的结论不应写成“hybrid 模型具备可用的自然语言 captioning 能力”。更准确的表述是：

> Qwen3-VL ViT/connector 输出的 hidden states 在替换为 Z-Image/Qwen3 language weights 后仍然携带显著视觉语义；但是该 hybrid 语言解码路径不稳定，容易输出 reasoning 模板或重复内容。因此它更适合被视为一种视觉条件表示，而不是直接作为 VQA/captioning 模型。

这也是后续从 “captioning 是否好” 转向 “hidden states 能否驱动 Z-Image 重建和改善 generation benchmark” 的关键原因。

## 4. Reconstruction LoRA 训练设计

### 4.1 数据来源

训练数据使用 RGB_stage1 中已经由 Z-Image 生成的图像：

```text
/mnt/hdfs/jixie/RGB_stage1/images/
```

目录下约 10 万张 PNG，命名形式为：

```text
00000000.png
00000001.png
...
```

为了避免 HDFS `ls` / 递归扫描非常慢，最终训练脚本不扫描目录，而是按 index 和模板直接生成路径：

```text
--image-count 100000
--image-name-template "{index:08d}.png"
```

数据中存在个别脏图，例如 `00093601.png` 曾触发 `cannot identify image file`。训练脚本加入了容错逻辑：遇到 `UnidentifiedImageError`、`OSError`、`FileNotFoundError` 时跳过并尝试下一个样本，最多重试 `--max-image-load-retries 1000`。

### 4.2 训练目标

训练是 rectified-flow / flow matching 风格的 reconstruction LoRA：

1. 目标图像 `x0` 经过 VAE 编码为 latent。
2. 采样噪声 `noise` 与随机时间 `t`。
3. 构造 noisy latent：

```text
x_t = (1 - t) * noise + t * x0
```

4. 监督速度：

```text
v_target = x0 - noise
```

5. 使用 Qwen3-VL image+ReCA prompt 的 hidden states 作为条件，让 Z-Image transformer 预测 `v_pred`。
6. 最小化：

```text
MSE(v_pred, v_target)
```

只训练 Z-Image transformer LoRA，冻结 VAE、Z-Image text encoder、Qwen3-VL。

### 4.3 LoRA 配置

LoRA target modules：

```text
feed_forward.w1
feed_forward.w2
feed_forward.w3
attention.to_k
attention.to_q
attention.to_v
attention.to_out.0
```

主要参数：

| 参数 | 原始训练 | 低 lr 新实验 |
|---|---:|---:|
| output resolution | 256 | 256 |
| VLM input resolution | 224 | 224 |
| local batch size | 16 | 16 |
| GPUs | 4 | 4 |
| global batch size | 64 | 64 |
| gradient accumulation | 1 | 1 |
| LoRA rank | 64 | 64 |
| LoRA alpha | 128 | 128 |
| mixed precision | bf16 | bf16 |
| training prompts | ReCA 360 prompts | ReCA 360 prompts |
| max train samples | 100000 | 100000 |
| max train steps | 2000 | 1000 |
| learning rate | 1e-4 | 5e-5 |
| validation interval | 100 | 100 |
| checkpoint interval | 500, later 100 for lr5e5 | 100 |

原始训练输出：

```text
/mnt/hdfs/jixie/checkpoints/zimage_reconstruction_train/rgb_stage1_zimage_reca_style_lora/
```

低 lr 新实验输出：

```text
/mnt/hdfs/jixie/checkpoints/zimage_reconstruction_train/rgb_stage1_zimage_reca_style_lora_lr5e5_1k/
```

### 4.4 Validation 设计

每 100 step 输出两类可视化：

1. **image-conditioned reconstruction**：

```text
samples/step_000100.png
```

每个样本以 `target / recon` 的形式展示。

2. **text-to-image validation**：

```text
samples_t2i/step_000100.png
```

固定 4 条 prompt，观察 LoRA 是否破坏或改善 T2I 能力。

这一点很重要：reconstruction LoRA 不只是要看“能不能复制输入图”，还要看作为 Z-Image transformer LoRA 是否改善或伤害纯文本生成能力。

## 5. Geneval 评测设计

### 5.1 为什么用 Geneval

如果 reconstruction alignment 的确让 Z-Image 的 DiT 学到更好的语义-结构对齐，那么它可能不仅改善 image-conditioned reconstruction，也会改善 text-to-image 的组合泛化能力。Geneval 正好测试多种 compositional generation 能力：

- `color_attr`
- `single_object`
- `position`
- `counting`
- `two_object`
- `colors`

因此我们用 Geneval 作为最终定量指标。

### 5.2 固定评测口径

主对比口径：

```text
512x512 / 25 inference steps / CFG 4.0 / 4 images per prompt / 553 prompts
```

每个 prompt 生成 4 张图，总计 2212 张图。baseline 与 LoRA 使用相同 Z-Image base、相同 metadata、相同生成配置。

输出目录：

```text
/opt/tiger/why-reca/outputs/geneval_zimage_recon_ckpt1000_25step_512/
/opt/tiger/why-reca/outputs/geneval_zimage_recon_ckpt500_25step_512/
```

另有一组 Z-Image 推荐配置评测：

```text
1024x1024 / 28 inference steps / CFG 4.0
```

输出目录：

```text
/opt/tiger/why-reca/outputs/geneval_zimage_recon_ckpt1000_1024_28step_cfg4/
```

## 6. Geneval 结果

### 6.1 512x512 / 25 steps / CFG 4.0

这是最重要的同配置比较。baseline、500 step、1000 step 的总分如下：

| 模型 | Overall |
|---|---:|
| Z-Image baseline | 0.69916 |
| reconstruction LoRA step500, lr=1e-4 | 0.77268 |
| reconstruction LoRA step1000, lr=1e-4 | 0.76073 |

完整小分：

| task | baseline | step500 | step1000 |
|---|---:|---:|---:|
| color_attr | 58.00% | 65.25% | 60.50% |
| single_object | 98.75% | 99.38% | 99.38% |
| position | 31.25% | 41.75% | 44.25% |
| counting | 61.88% | 75.94% | 71.56% |
| two_object | 86.11% | 91.67% | 91.92% |
| colors | 83.51% | 89.63% | 88.83% |
| overall | 69.916% | 77.268% | 76.073% |

关键观察：

1. 两个 reconstruction LoRA checkpoint 都显著高于 baseline。
2. step500 的总分高于 step1000，说明在 `lr=1e-4` 下继续训练到 1000 step 可能开始轻微过拟合 reconstruction objective 或影响部分 text-to-image 泛化。
3. step500 主要赢在 `color_attr`、`counting`、`colors`。
4. step1000 在 `position` 和 `two_object` 上略优，但不足以抵消 step500 在其他任务上的优势。

可以写入论文的表述：

> A short reconstruction-alignment LoRA substantially improves Z-Image's compositional text-to-image performance under the same sampling budget. Interestingly, the 500-step checkpoint outperforms the 1000-step checkpoint at 512 resolution, suggesting that reconstruction alignment has an early beneficial regime and may require learning-rate or early-stopping control to avoid over-specialization.

### 6.2 1024x1024 / 28 steps / CFG 4.0

Z-Image 推荐配置下的结果：

| 模型 | Overall |
|---|---:|
| Z-Image baseline | 0.74661 |
| reconstruction LoRA step1000, lr=1e-4 | 0.79415 |

小分：

| task | baseline | step1000 |
|---|---:|---:|
| color_attr | 61.50% | 67.75% |
| single_object | 100.00% | 99.06% |
| position | 39.00% | 46.75% |
| counting | 71.88% | 77.50% |
| two_object | 88.89% | 93.94% |
| colors | 86.70% | 91.49% |
| overall | 74.661% | 79.415% |

关键观察：

1. 在更高分辨率和更推荐的采样设置下，baseline 本身更强。
2. step1000 LoRA 仍然带来约 `+0.04754` 的绝对提升。
3. 提升分布较均衡，尤其 `position`、`two_object`、`colors` 明显改善。

注意：`0.79415` 是 `1024x1024 / 28step / CFG 4.0` 的 step1000 分数，不应与 `512x512 / 25step` 的 step500 直接比较。此前曾把这两个配置混淆，后续写作必须严格标注配置。

### 6.3 低学习率 lr=5e-5 新实验状态

为验证 `1e-4` 下 500 step > 1000 step 是否来自学习率偏大或过训练，启动了新实验：

```text
exp_name = rgb_stage1_zimage_reca_style_lora_lr5e5_1k
lr = 5e-5
max_train_steps = 1000
checkpoint_steps = 100
```

该实验已完成 1000 step，并已自动评测 step500 与 step1000。训练末尾日志示例：

```text
step=985 loss=0.432645
step=1000 loss=0.408581
saved checkpoint .../checkpoints/step_001000/recon
saved final checkpoint .../checkpoints/final/recon
```

已经产生完整 checkpoint 与 validate 图：

```text
checkpoints/step_000500/recon/...
checkpoints/step_001000/recon/...
samples/step_000500.png
samples_t2i/step_000500.png
samples/step_001000.png
samples_t2i/step_001000.png
```

自动评测配置：

```text
512x512 / 25step / CFG 4.0
```

输出目录：

```text
/opt/tiger/why-reca/outputs/geneval_zimage_recon_lr5e5_1k_25step_512/
```

结果：

| 模型 | Overall |
|---|---:|
| reconstruction LoRA step500, lr=5e-5 | 0.75328 |
| reconstruction LoRA step1000, lr=5e-5 | 0.76528 |

小分：

| task | lr=5e-5 step500 | lr=5e-5 step1000 |
|---|---:|---:|
| color_attr | 64.75% | 66.00% |
| single_object | 98.75% | 99.69% |
| position | 37.00% | 40.00% |
| counting | 71.25% | 72.19% |
| two_object | 91.92% | 91.67% |
| colors | 88.30% | 89.63% |
| overall | 75.328% | 76.528% |

这说明降低学习率后，step1000 相比 step500 不再退化，支持“`1e-4` 下 step1000 轻微变差可能与学习率/过训练有关”的判断。不过在已经完成的 `512/25/CFG4` 实验中，最高分仍是 `lr=1e-4 step500` 的 `0.77268`。

在此基础上又启动了从 `lr=5e-5 step1000` 继续训练 1000 step 的实验，用于观察 1500/2000 step 是否继续改善或开始退化。继续训练仍然无 warmup、无 scheduler，固定学习率 `5e-5`。

## 7. 重要踩坑与修复

### 7.1 未及时提交导致脚本丢失

最严重的问题是早期 `train_reconstruction.py` 和启动脚本曾经只是工作区未跟踪文件，没有进入 git。后续工作区恢复/清理后文件消失，无法从 git 找回，只能从 Cursor transcript 逆向恢复。这个问题后来通过以下 commit 修复：

```text
8057cc1 Restore Z-Image reconstruction training scripts
2034f15 Cast reconstruction transformer for bf16 training
b65c3d7 Fix reconstruction validation sampling with CFG
43cc4ed Add Z-Image LoRA Geneval automation
```

远程仓库：

```text
https://github.com/HorizonWind2004/D-OPSD
```

### 7.2 CRLF 行尾导致 bash 失败

多个 shell 脚本在恢复/patch 后出现 CRLF 行尾，导致 `bash` 解析出错，例如：

```text
set -euo pipefail\r
syntax error near unexpected token `$'do\r''
```

修复方式是统一转换为 LF，并在提交前运行：

```bash
git diff --check
bash -n scripts/xxx.sh
```

### 7.3 bf16 dtype mismatch

恢复版训练脚本最初没有把 LoRA 后的 Z-Image transformer cast 到 bf16，导致 Z-Image transformer 内部 pad token 是 fp32，而输入序列在 autocast 下是 bf16：

```text
RuntimeError: Index put requires the source and destination dtypes match,
got BFloat16 for the destination and Float for the source.
```

修复：

```python
pipeline.transformer.to(accelerator.device, dtype=inference_dtype)
```

### 7.4 validate 时 prompt_embeds + CFG 需要 negative_prompt_embeds

训练在第 100 step validation 崩溃过，原因是 `sample_reconstructions()` 使用 `prompt_embeds` 直接调用 Z-Image pipeline，同时 `guidance_scale=4.0`，但没有传 `negative_prompt_embeds`。Z-Image pipeline 明确要求：

```text
When prompt_embeds is provided without prompt,
negative_prompt_embeds must also be provided for classifier-free guidance.
```

修复是在 reconstruction validation 中显式用空 prompt 编码 negative embeddings：

```python
negative_prompt_embeds = _encode_prompt(
    pipeline.text_encoder,
    pipeline.tokenizer,
    ["" for _ in prompt_embeds],
    device=accelerator.device,
    max_sequence_length=args.max_sequence_length,
)
```

### 7.5 HDFS 目录扫描慢

HDFS 上递归列目录非常慢，因此训练数据路径改为按 index 模板生成。这是能稳定大规模训练的关键工程优化。

### 7.6 训练不是越久越好

`lr=1e-4` 下，step500 在 `512/25/CFG4` 上优于 step1000。这个现象提示 reconstruction objective 可能有最佳早停点，也可能需要更低学习率。低 lr=5e-5 的 1k 实验就是为了验证这一点。

## 8. 可复现实验入口

当前 D-OPSD 仓库中与本实验相关的文件：

```text
z-image-turbo_self-distill-vlm/train_reconstruction.py
z-image-turbo_self-distill-vlm/scripts/train_reconstruction_rgb_stage1.sh
z-image-turbo_self-distill-vlm/scripts/train_reconstruction_rgb_stage1_lr5e5_1k.sh
z-image-turbo_self-distill-vlm/scripts/generate_geneval_zimage.py
z-image-turbo_self-distill-vlm/scripts/run_geneval_zimage_lora_512_25.sh
z-image-turbo_self-distill-vlm/scripts/watch_lr5e5_1k_then_geneval.sh
```

启动 lr=5e-5 训练：

```bash
cd /opt/tiger/D-OPSD/z-image-turbo_self-distill-vlm
CUDA_VISIBLE_DEVICES=0,1,2,3 MAIN_PROCESS_PORT=39517 \
  bash scripts/train_reconstruction_rgb_stage1_lr5e5_1k.sh
```

训练完成后自动评测 step500/step1000：

```bash
cd /opt/tiger/D-OPSD/z-image-turbo_self-distill-vlm
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash scripts/watch_lr5e5_1k_then_geneval.sh
```

手动评测任意 LoRA：

```bash
cd /opt/tiger/D-OPSD/z-image-turbo_self-distill-vlm
CUDA_VISIBLE_DEVICES=0,1,2,3 \
LORA=/path/to/checkpoint/recon \
PHASE=finetuned_custom \
OUT_ROOT=/opt/tiger/why-reca/outputs/geneval_custom_25step_512 \
  bash scripts/run_geneval_zimage_lora_512_25.sh
```

## 9. 对论文写作的建议性结论

可以将本实验写成三个递进层次：

1. **Connector hypothesis**：同族 VLM 的视觉塔与 connector 产生的 hidden states 可以落入文本生成模型/图像生成模型可消费的语义空间。即便 hybrid language decoding 不稳定，hidden states 中仍保留丰富视觉信息。
2. **Reconstruction alignment as a bridge**：用 image-conditioned hidden states 训练 Z-Image transformer LoRA 做 reconstruction，可以将视觉条件对齐到 DiT 的生成动态中。
3. **Generative capability transfer**：这种 reconstruction training 不仅服务于重建，还提升了纯文本生成 benchmark，尤其 Geneval 中组合属性、计数、颜色与空间关系等能力。

关键定量证据：

- `512x512 / 25step / CFG4`：baseline `0.69916`，step500 `0.77268`，step1000 `0.76073`。
- `1024x1024 / 28step / CFG4`：baseline `0.74661`，step1000 `0.79415`。

论文中应避免夸大为“zero-shot captioning 已经完全可用”。更稳妥的写法是：

> The hybrid Qwen3-VL connector representation is semantically meaningful but not a stable language-decoding interface. Its value becomes clear when used as a conditioning representation for Z-Image reconstruction alignment. A short LoRA training phase using these image-conditioned hidden states improves Geneval performance substantially, suggesting that reconstruction alignment transfers visual grounding into the text-to-image generation model.

## 10. 当前开放问题

1. `lr=5e-5` 从 step1000 继续训练到 step2000 的 Geneval 尚待完成，建议重点比较 step1500/step2000 与当前最优 `lr=1e-4 step500`。
2. 需要对 reconstruction samples 做人工筛选，观察其是否真的保留布局、颜色、主体，而不是只学习 prompt style。
3. 需要比较不同 validation CFG，例如 reconstruction sample 使用 CFG 0、1、4 对视觉保真度的影响。
4. 需要验证更小 LoRA rank 或更短训练是否已经足够，因为 `lr=1e-4 step500` 当前仍是 512/25 配置下最优。
5. 需要确认 RGB_stage1 训练图全部来自 Z-Image，这会影响论文中对 data distribution 与 self-generated reconstruction alignment 的描述。
