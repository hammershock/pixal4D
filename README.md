# Pixal4D

A minimal inference extension of Pixal3D for causal mesh sequences. It supports zero-initialized temporal residuals in all four generation flows: sparse structure, 512 geometry, 1024 geometry, and 1024 PBR material. The native per-frame geometry and material decoders remain unchanged. Mesh topology may change between frames. Maximum sequence length is 16 frames; the Python API supports frame padding masks. High-resolution temporal adapters currently support `1024_cascade` only.

Follow the upstream installation instructions below, then run from this checkout:

```bash
python inference_video.py --frames path/to/rgba_frames \
  --cameras path/to/cameras.json --output outputs/animation \
  --timestamps path/to/timestamps.json \
  --adapters path/to/temporal_adapter
```

Frames must be ordered RGBA PNGs with matching dimensions and common framing. Alpha is the foreground mask. This entry preserves the image canvas and does not crop or register individual frames. Camera metadata uses Pixal3D's projection convention: a JSON object such as `{"camera_angle_x": 0.7, "distance": 2.0, "mesh_scale": 1.0}`, or one object per frame. FOV is in radians; camera values must correspond to the supplied canvas. Camera estimation and world-space trajectory recovery are not part of this entry.

Supply actual timestamps in seconds as a JSON array, for example `[0.0, 0.04, 0.11]`. For uniformly sampled inputs, `--fps 25` can replace `--timestamps`; use the FPS of the supplied frames after subsampling. Temporal attention applies continuous rotary phases to Q/K using seconds relative to the first valid observation, so its temporal component depends on pairwise time differences. It has no learned frame-index embeddings and does not normalize by the final frame time. Fractional and irregular times are supported; missing images are not synthesized from timestamps alone. Interpolation quality and temporal consistency still require training and evaluation. The 16-frame limit is a resource policy, independent of the time encoding.

An adapter directory contains `config.json` (`max_frames`, `adapter`, and a `stages` mapping) plus stage-specific safetensors. The adapter config records `time_encoding: "relative_rope_v1"` and `time_scale: 1.0` (seconds). The mapping selects the active flows, including two-stage subsets. **Old frame-index adapter weights are incompatible and must be retrained**; they are rejected rather than silently converted. Accepted keys are `sparse_structure_flow_model`, `shape_slat_flow_model_512`, `shape_slat_flow_model_1024`, and `tex_slat_flow_model_1024`. No general-purpose trained temporal checkpoint is released yet. Omitting `--adapters` initializes all four flows with zero temporal residuals, reproducing native per-frame sampling rather than a learned motion prior. Temporal connectivity alone does not guarantee consistent geometry or materials.

Outputs are per-frame PBR GLBs and `animation.json`, in one native view-aligned coordinate convention. Use `--model` for a local Pixal3D checkpoint directory, `--image-encoder` / `--naf-source` for local DINOv3 / NAF resources, `--steps` to change sampling steps, and `--texture-size` / `--faces` to control GLB export. The upstream image-to-3D entry remains available as `inference.py`.

For an already initialized native pipeline, use `pixal4d.pipeline.Pixal4DVideoPipeline(native_pipeline).run(images, cameras, timestamps=times, valid_frames=mask)`. Valid times must be finite and strictly increasing; padding may use `None` and does not change valid observations' timing. Pass `stages=tuple(stage_names)` to the constructor to adapt a subset; otherwise all available supported flows are selected. Padding frames return `None` and do not enter temporal attention. `animation.json` preserves each input timestamp alongside its mesh. Training utilities remain private; saved predictions can be evaluated through the public entry below.

## Evaluate saved predictions

```bash
pip install -r requirements-eval.txt
python evaluate.py --csv path/to/evaluation.csv --output outputs/eval --save-previews
# Geometry only: CPU, no model checkpoints, PyTorch or renderer required
python evaluate.py --csv path/to/evaluation.csv --output outputs/geometry --metrics geometry --device cpu
# Externally rendered RGB pairs, with matching camera, lighting and canvas
python evaluate.py --csv path/to/evaluation.csv --output outputs/rgb --metrics rgb --device cpu
```

`evaluate.py` is the only evaluation entry. Defaults are `--metrics geometry pixel`, 100,000 surface samples, F-score thresholds `0.01 0.02`, seed 42, and CUDA normal rendering through the upstream nvdiffrast installation. Pixel/RGB metrics use pretrained AlexNet LPIPS v0.1; the first use may download its backbone into the standard PyTorch cache (`TORCH_HOME` can select an existing cache). `--skip-lpips` explicitly omits that metric. Evaluation does not load Pixal3D or temporal checkpoints. Use a new output directory for each run.

The CSV contains **one row per matching frame/view**, with paths relative to the CSV, or absolute paths. See [example manifest](examples/evaluation.csv); replace its placeholder mesh paths before running. No dataset-specific directory convention is assumed.

| Column | Meaning |
| --- | --- |
| `sequence_id,frame_id,timestamp` | Required sequence, frame identifier and time in seconds. Both meshes/images must represent this time; frame indices are never implicitly zipped. Multiple views may share a frame. |
| `pred_mesh,gt_mesh` | Required for geometry/pixel metrics. Static triangle meshes (e.g. GLB/PLY/OBJ), or NPZ with `vertices,faces`. Animated containers must first be exported at each timestamp. GLB scene-instance transforms are applied. |
| `reference_length` | Required for geometry: positive reference length in the common coordinate units, fixed across the sequence. Use 1 for a unit-cube-side protocol, or supply the GT sequence bounding-box diagonal. It scales distances/thresholds only; meshes are never recentered or rescaled automatically. |
| `view_id,view_type,camera` | Pixel metrics require a named view, `input` or `novel`, and a camera JSON. Input and novel views are reported separately. |
| `pred_transform,gt_transform` | Optional JSON files containing affine 4×4 matrices, applied after scene transforms. Each transform must stay fixed over a sequence. Empty means identity; no ICP or fitted registration is performed. |
| `pred_rgb,gt_rgb` | Required only with `--metrics rgb` (also requires `view_id,view_type`). Matching saved RGB/RGBA images. No resize, crop or exposure matching; alpha composites onto white. RGB evaluates the whole canvas, so keep object framing/background fixed. |

Camera JSON uses **OpenCV coordinates** (x right, y down, z forward), `K` in pixels, and a rigid `world_to_camera` transform, plus integer `width,height`; optional `near,far` default to 0.001 and 100. See [camera example](examples/evaluation_camera.json). It must describe the meshes **after** the supplied transforms, and the original image canvas. The example camera looks toward the origin from +z in native mesh coordinates. `inference_video.py` already restores native Pixal3D coordinates after O-Voxel's internal GLB axis conversion; **do not rotate its output again**. When GT is also in that native frame, leave both transform columns empty. For square inputs set `fx=fy=width/(2*tan(camera_angle_x/2))`, `cx=width/2`, `cy=height/2`, and camera z translation to `distance*mesh_scale`. Do not reuse these conventions for unrelated exporters.

| Metric family | Output and interpretation |
| --- | --- |
| Whole-surface geometry | `cd_l1` ↓: half-sum of mean bidirectional nearest-sample Euclidean distances divided by `reference_length`. `cd_l2` ↓ uses squared normalized distances. `fscore@0.01`, `fscore@0.02` ↑ use those fractions of the reference length. `normal_consistency` ↑ is symmetric mean absolute nearest-sample normal cosine, insensitive to winding flips. Independent area-uniform samples imply nonzero Chamfer even for identical surfaces. |
| Pixal3D pixel alignment | `mask_iou` ↑, `normal_psnr` ↑, `normal_ssim` ↑, `normal_lpips` ↓, mean/median/boundary normal angular error in degrees ↓, and accuracy below 11.25°/22.5°/30° ↑. `gt_coverage` reports the fraction of the GT silhouette covered by the prediction. These measure geometry, **not material appearance**. |
| Optional RGB appearance | `rgb_psnr` ↑, `rgb_ssim` ↑, `rgb_lpips` ↓ from supplied rendered pairs. PBR shading/lighting must be matched by the caller; this entry does not infer lighting or automatically render GLB materials. |

Geometry follows the surface-metric conventions in [Occupancy Networks](https://github.com/autonomousvision/occupancy_networks/blob/master/im2mesh/eval.py). Pixel metrics follow [Pixal3D §4.1 / Table 1](https://arxiv.org/html/2605.10922v1#S4.SS1), with the following **explicit local protocol**, not an exact reproduction of its published numbers: flat face-forward camera normals, no antialiasing, encoded as `(normal+1)/2`; PSNR and angular metrics on mask intersection; SSIM with an 11×11 Gaussian window only where the full window is valid; spatial LPIPS averaged on the intersection after both images receive neutral background outside it. `normal_boundary_mean_deg` uses the overlapping part of the inner GT silhouette band (`--boundary-width 3`, 4-connected erosion). Normal-map creases away from the silhouette are not included in that boundary definition. Normal conventions, sampling density, resolution, and background can change the scores. The upstream paper branch uses Direct3D-S2, whereas this fork uses the TRELLIS.2-based branch.

All scores except PSNR/angles/distances use fractions, not percentages. PSNR is capped at 120 dB. Empty overlap/window scores are `null`, never zero error; inspect IoU and coverage alongside them. Invalid/missing files, conflicting timestamps, duplicate frame/view rows, and GT entirely outside the camera fail the run. The manifest explicitly declares frame correspondence; filenames alone cannot verify it. No silent sample skipping is allowed.

Outputs: `metrics.csv` (one geometry row per frame plus rows per view), `summary.json` (views → frames → sequences → equal-weight dataset mean, valid/total counts, parameters, package versions and input hashes), `progress.json`, and optional `previews/` (GT left, prediction right, numbered by CSV row starting at zero). Geometry is counted once even with multiple views. A complete summary is written only after all rows succeed. These **per-frame quality metrics do not establish temporal consistency**; do not label their average as a flicker/motion score. A benchmark without GT geometry cannot supply CD or GT-normal metrics; use registered RGB references where available and report that limited scope.

---


<div align="center">

# Pixal3D: Pixel-Aligned 3D Generation from Images

<h3>SIGGRAPH 2026</h3>

<small>[Dong-Yang Li](https://ldyang694.github.io/)¹ · [Wang Zhao](https://thuzhaowang.github.io/)²* · [Yuxin Chen](https://orcid.org/0000-0002-7854-1072)² · [Wenbo Hu](https://wbhu.github.io/)² · [Meng-Hao Guo](https://menghaoguo.github.io/)¹ · [Fang-Lue Zhang](https://fanglue.github.io/)³ · [Ying Shan](https://www.linkedin.com/in/YingShanProfile)² · [Shi-Min Hu](https://cg.cs.tsinghua.edu.cn/shimin.htm)¹✉</small>

¹Tsinghua University (BNRist) &nbsp;&nbsp; ²Tencent ARC Lab &nbsp;&nbsp; ³Victoria University of Wellington

*Project lead &nbsp;&nbsp; ✉Corresponding author

</div>

<div align="center">
  <a href="https://ldyang694.github.io/projects/pixal3d/"><img src=https://img.shields.io/badge/Project%20Page-333399.svg?logo=googlehome height=22px></a>
  <a href="https://huggingface.co/spaces/TencentARC/Pixal3D"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Demo-276cb4.svg height=22px></a>
  <a href="https://huggingface.co/TencentARC/Pixal3D"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg height=22px></a>
  <a href="https://arxiv.org/abs/2605.10922"><img src=https://img.shields.io/badge/Arxiv-b5212f.svg?logo=arxiv height=22px></a>
  <a href="LICENSE"><img src=https://img.shields.io/badge/License-MIT-yellow.svg height=22px></a>
</div>

<div align="center">
    <img src="assets/teaser.png" alt="Teaser image of Pixal3D"/>
</div>

**Pixal3D** generates high-fidelity 3D assets from a single image. Unlike previous methods that loosely inject image features via attention, Pixal3D explicitly lifts pixel features into 3D through back-projection, establishing direct pixel-to-3D correspondences. This enables near-reconstruction-level fidelity with detailed geometry and PBR textures.

---

## ✨ News

- **Sep 2026**: Release multi-view inference code. 🎥
- **May 2026**: Release training code and data preparation toolkit. 🔧
- **May 2026**: Release the improved version based on [Trellis.2](https://github.com/microsoft/TRELLIS.2) backbone. 💪
- **May 2026**: Release inference code and online demo. 🤗
- **Apr 2026**: Our paper is accepted to SIGGRAPH 2026! 🎉

## 📌 Branches

| Branch | Description |
|--------|-------------|
| `main` | **Latest version** — improved implementation based on [Trellis.2](https://github.com/microsoft/TRELLIS.2) backbone with better performance. |
| `paper` | **Paper version** — original implementation based on [Direct3D-S2](https://github.com/DreamTechAI/Direct3D-S2), corresponding to results reported in our SIGGRAPH 2026 paper. |

> If you want to reproduce the results in our paper, please switch to the `paper` branch.

## 🎮 Try It Online

You can try Pixal3D directly in your browser without any installation via our Hugging Face Gradio demo:

👉 [**Launch Demo**](https://huggingface.co/spaces/TencentARC/Pixal3D)

## 🚀 Getting Started

### Installation

#### Step 1: Follow TRELLIS.2 Installation

Please first follow the installation guide of [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) to set up the base environment.

#### Step 2: Install Additional Dependencies

```bash
pip install -r requirements.txt
```

#### Step 3: Install natten

```bash
NATTEN_CUDA_ARCH="xx" NATTEN_N_WORKERS=xx pip install natten==0.21.0 --no-build-isolation
```

Please replace `xx` with the CUDA architecture and the number of build workers suitable for your machine.

#### Step 4: Install utils3d

```bash
pip install https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl
```

> **Note**: `requirements-hfdemo.txt` is for the Hugging Face Spaces demo (H-series GPU architecture) and may not be compatible with other architectures.

### Usage

#### Inference

Generate a GLB mesh from a single image:

```bash
python inference.py --image assets/images/0_img.png --output ./output.glb
```

**Low-VRAM mode** (reduces peak VRAM by loading models on-demand):

```bash
python inference.py --image assets/images/0_img.png --output ./output.glb --low_vram
```

By default, the pipeline resolution is **1536** (standard mode) or **1024** (low-VRAM mode). You can override this with `--resolution`:

```bash
# Force 1536 even in low-VRAM mode
python inference.py --image assets/images/0_img.png --output ./output.glb --low_vram --resolution 1536

# Force 1024 in standard mode
python inference.py --image assets/images/0_img.png --output ./output.glb --resolution 1024
```

**Tip**: If you don't have `flash_attn` installed, you can use PyTorch's built-in SDPA backend instead:
> ```bash
> ATTN_BACKEND=sdpa python inference.py --image assets/images/0_img.png --output ./output.glb --low_vram
> ```

#### Multi-View Inference

When you have several views of the same object — a few shots taken around it, or the output of a multi-view diffusion model — `inference_mv.py` conditions the cascade on all of them at once:

```bash
python inference_mv.py --views_dir assets/mv_images/example --output ./output_mv.glb
```

This path uses a separate set of multi-view weights (`ckpts/*_mv`, selected by `pipeline_mv.json`) from the same Hugging Face repo. `--low_vram`, `--resolution` and `ATTN_BACKEND` behave exactly as above, and `--num_views N` restricts the run to the first N views.

**Input format.** A directory of views plus a `transforms.json` describing each view's camera:

```
assets/mv_images/example/
├── transforms.json
├── view00_azim000.png     # alpha is taken as the object mask when present,
├── view01_azim090.png     # otherwise the view is segmented automatically
├── view02_azim180.png
└── view03_azim270.png
```

Views need not be RGBA — any view without an alpha channel is matted with the same background-removal model `inference.py` uses. They are never cropped or rescaled, so the framing you provide has to be the framing the cameras describe.

```json
{
  "camera_angle_x": 0.3490658503988659,
  "mesh_scale": 1.0,
  "frames": [
    {
      "file_path": "view00_azim000.png",
      "name": "azim000",
      "transform_matrix": [[1, 0, 0, 0], [0, 0, -1, -3.1192], [0, 1, 0, 0], [0, 0, 0, 1]]
    }
  ]
}
```

`transform_matrix` is a 4×4 camera-to-world matrix and `camera_angle_x` is the horizontal FOV in radians — the same Blender/NeRF convention the training renders use, so a dataset `transforms.json` works as-is. The world is Z-up, and each camera looks along its own −Z axis with its own +Y as up. `camera_angle_x` may also be given per frame, which overrides the top-level value.

> **No camera parameters?** If your views are the usual orbit — four images taken every 90° around the object at eye level — reuse the shipped `assets/mv_images/example/transforms.json` as-is and just point each `file_path` at your own image. It already encodes exactly that rig: azimuths 0°/90°/180°/270°, elevation 0°, 20° FOV.

> **The first frame is the main view and should be the canonical front view** (camera at `(0, -d, 0)` looking at the origin with world +Z up), since every other view is placed relative to it. `inference_mv.py` warns if it is not.

### Web Demo

We provide a Gradio web demo for Pixal3D, which allows you to generate 3D meshes from images interactively.

```bash
python app.py 
```

Low-VRAM mode is also available for the web demo. The frontend default resolution will automatically switch to 1024 in low-VRAM mode (1536 otherwise), but can be changed manually in the UI.

```bash
python app.py --low_vram
# or via environment variable:
LOW_VRAM=1 python app.py
```
## 🔧 Training

We provide the full training codebase for reproducing Pixal3D from scratch.

### Data Preparation

Prepare view-aligned O-Voxel data and rendered condition images by following the data toolkit instructions:

> 📂 **[data_toolkit/README.md](data_toolkit/README.md)**

### Overview

Pixal3D is trained as a three-stage cascade, each progressively increasing resolution:

| Stage | Model | Resolutions | Config Prefix |
|-------|-------|-------------|---------------|
| 1 | Sparse Structure | 32 → 64 | `ss_flow_img_dit_*_proj_finetune` |
| 2 | Shape | 256 → 512 → 1024 | `slat_flow_img2shape_*_proj_finetune` |
| 3 | Texture | 256 → 512 → 1024 | `slat_flow_imgshape2tex_*_proj_finetune` |

All stages use **pixel-aligned projection conditioning** and **view-aligned latents** (2 views by default). Within each stage, start from the lowest resolution and progressively fine-tune to higher resolutions by setting `finetune_ckpt` in the config.

### Quick Start

```sh
python train.py \
  --config <CONFIG_JSON> \
  --output_dir <OUTPUT_DIR> \
  --data_dir '<DATA_DIR_JSON>'
```

`--data_dir` is a JSON string describing the dataset layout. Different stages require different keys:

| Stage | Required keys |
|-------|---------------|
| Sparse Structure | `base`, `ss_latent`, `render_cond` |
| Shape | `base`, `shape_latent`, `render_cond` |
| Texture | `base`, `shape_latent`, `pbr_latent`, `render_cond` |

### Example: Training All Three Stages

Below we show the full training sequence using ObjaverseXL as an example. Each higher-resolution step requires updating `finetune_ckpt` in its config JSON to point to the previous checkpoint.

<details>
<summary><b>Stage 1: Sparse Structure (32 → 64)</b></summary>

```sh
# Resolution 32
python train.py \
  --config configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_finetune.json \
  --output_dir results/ss_32 \
  --data_dir '{"ObjaverseXL_sketchfab": {"base": "datasets/ObjaverseXL_sketchfab", "ss_latent": "datasets/ObjaverseXL_sketchfab/ss_latents/ss_enc_conv3d_16l8_fp16_64_view", "render_cond": "datasets/ObjaverseXL_sketchfab/renders_cond"}}'

# Resolution 64 (set finetune_ckpt → results/ss_32 checkpoint)
python train.py \
  --config configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_finetune_ft64.json \
  --output_dir results/ss_ft64 \
  --data_dir '{"ObjaverseXL_sketchfab": {"base": "datasets/ObjaverseXL_sketchfab", "ss_latent": "datasets/ObjaverseXL_sketchfab/ss_latents/ss_enc_conv3d_16l8_fp16_64_view", "render_cond": "datasets/ObjaverseXL_sketchfab/renders_cond"}}'
```
</details>

<details>
<summary><b>Stage 2: Shape (256 → 512 → 1024)</b></summary>

```sh
# Resolution 256
python train.py \
  --config configs/gen/slat_flow_img2shape_dit_1_3B_256_bf16_proj_finetune.json \
  --output_dir results/shape_256 \
  --data_dir '{"ObjaverseXL_sketchfab": {"base": "datasets/ObjaverseXL_sketchfab", "shape_latent": "datasets/ObjaverseXL_sketchfab/shape_latents/shape_enc_next_dc_f16c32_fp16_256_view", "render_cond": "datasets/ObjaverseXL_sketchfab/renders_cond"}}'

# Resolution 512
python train.py \
  --config configs/gen/slat_flow_img2shape_dit_1_3B_256_bf16_proj_finetune_ft512.json \
  --output_dir results/shape_ft512 \
  --data_dir '{"ObjaverseXL_sketchfab": {"base": "datasets/ObjaverseXL_sketchfab", "shape_latent": "datasets/ObjaverseXL_sketchfab/shape_latents/shape_enc_next_dc_f16c32_fp16_512_view", "render_cond": "datasets/ObjaverseXL_sketchfab/renders_cond"}}'

# Resolution 1024
python train.py \
  --config configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_proj_finetune_ft1024.json \
  --output_dir results/shape_ft1024 \
  --data_dir '{"ObjaverseXL_sketchfab": {"base": "datasets/ObjaverseXL_sketchfab", "shape_latent": "datasets/ObjaverseXL_sketchfab/shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view", "render_cond": "datasets/ObjaverseXL_sketchfab/renders_cond"}}'
```
</details>

<details>
<summary><b>Stage 3: Texture (256 → 512 → 1024)</b></summary>

```sh
# Resolution 256
python train.py \
  --config configs/gen/slat_flow_imgshape2tex_dit_1_3B_256_bf16_proj_finetune.json \
  --output_dir results/tex_256 \
  --data_dir '{"ObjaverseXL_sketchfab": {"base": "datasets/ObjaverseXL_sketchfab", "shape_latent": "datasets/ObjaverseXL_sketchfab/shape_latents/shape_enc_next_dc_f16c32_fp16_256_view", "pbr_latent": "datasets/ObjaverseXL_sketchfab/pbr_latents/tex_enc_next_dc_f16c32_fp16_256_view", "render_cond": "datasets/ObjaverseXL_sketchfab/renders_cond"}}'

# Resolution 512
python train.py \
  --config configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_finetune.json \
  --output_dir results/tex_512 \
  --data_dir '{"ObjaverseXL_sketchfab": {"base": "datasets/ObjaverseXL_sketchfab", "shape_latent": "datasets/ObjaverseXL_sketchfab/shape_latents/shape_enc_next_dc_f16c32_fp16_512_view", "pbr_latent": "datasets/ObjaverseXL_sketchfab/pbr_latents/tex_enc_next_dc_f16c32_fp16_512_view", "render_cond": "datasets/ObjaverseXL_sketchfab/renders_cond"}}'

# Resolution 1024
python train.py \
  --config configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_finetune_ft1024.json \
  --output_dir results/tex_ft1024 \
  --data_dir '{"ObjaverseXL_sketchfab": {"base": "datasets/ObjaverseXL_sketchfab", "shape_latent": "datasets/ObjaverseXL_sketchfab/shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view", "pbr_latent": "datasets/ObjaverseXL_sketchfab/pbr_latents/tex_enc_next_dc_f16c32_fp16_1024_view", "render_cond": "datasets/ObjaverseXL_sketchfab/renders_cond"}}'
```
</details>

### Additional Options

<details>
<summary><b>All command-line arguments</b></summary>

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Config JSON path | *required* |
| `--output_dir` | Output directory | *required* |
| `--data_dir` | Dataset JSON string | `./data/` |
| `--load_dir` | Checkpoint load directory | `output_dir` |
| `--ckpt` | Resume from step | `latest` |
| `--auto_retry` | Retries on failure | `3` |
| `--tryrun` | Dry run | `false` |
| `--profile` | Profiling | `false` |
| `--num_nodes` | Number of nodes | `1` |
| `--node_rank` | Current node rank | `0` |
| `--num_gpus` | GPUs per node | all |
| `--master_addr` | Master address | `localhost` |
| `--master_port` | Master port | `12666` |
| `--use_wandb` | Enable W&B logging | `false` |
| `--wandb_project` | W&B project | `trellis2-training` |
| `--wandb_name` | W&B run name | basename of `output_dir` |
| `--wandb_id` | W&B run ID (resume) | — |

</details>

## 🌐 Community Projects

We thank the community for building extensions and deployment guides for Pixal3D!

- [Pixal3D-ComfyUI](https://github.com/Saganaki22/Pixal3D-ComfyUI) — ComfyUI integration with deployment guides for Windows, WSL, and more.

## 🤗 Acknowledgements

This project is heavily built upon [Trellis.2](https://github.com/microsoft/TRELLIS.2) and [Direct3D-S2](https://github.com/DreamTechAI/Direct3D-S2). We sincerely thank the authors for their outstanding work on scalable 3D generation , which serves as the foundation of our codebase and model architecture.

We also thank the following repos for their great contributions:

- [Direct3D-S2](https://github.com/DreamTechAI/Direct3D-S2)
- [Trellis](https://github.com/microsoft/TRELLIS)
- [Trellis.2](https://github.com/microsoft/TRELLIS.2)

## 📄 Citation

If you find this work useful, please consider citing:

```bibtex
@article{li2026pixal3d,
    title={Pixal3D: Pixel-Aligned 3D Generation from Images},
    author={Li, Dong-Yang and Zhao, Wang and Chen, Yuxin and Hu, Wenbo and Guo, Meng-Hao and Zhang, Fang-Lue and Shan, Ying and Hu, Shi-Min},
    journal={arXiv preprint arXiv:2605.10922},
    year={2026}
}
```

## 📜 License

This project is released under the [MIT License](LICENSE). The third-party components included in this project remain licensed under their respective original terms; see [NOTICE](NOTICE) for the full list of dependencies and their licenses.

