# Channel-Diff

**Physics-guided Diffusion Models for Multi-scale Prediction of Reference Signal Received Power in Wireless Networks**

Xiaoqian Qi, Haoye Chai, Yue Wang, Zhaocheng Wang, and Yong Li

[Paper & citation](#citation) · [Model](models_with_mask_scale.py) · [Training](train.py) · [Data loader](DataLoader.py)

Channel-Diff predicts Reference Signal Received Power (RSRP) sequences along user trajectories from network parameters and urban environmental information. It combines propagation physics with conditional diffusion to capture both large-scale (LS) attenuation and the conditional statistics of small-scale (SS) fading.

The method uses a teacher stage to learn propagation-model-derived RSRP, followed by a student stage trained on measured RSRP. Occlusion-aware prior guidance carries physical information into generation, while reflection embeddings and a Microenvironment Feature Extraction Network (MFEN) provide complementary descriptions of the surrounding environment.

## Overview

Given a trajectory of length $T$, Channel-Diff generates an aligned RSRP sequence $x\in\mathbb{R}^{T}$ conditioned on network and environmental features. The paper studies this as conditional sequence generation; a measured RSRP prefix is not required for full-sequence generation.

The framework contains three layers:

1. **Input layer:** network parameters describing the base station, user equipment, and their geometry, together with ground-altitude and building-height maps.
2. **Physics representation layer:** Large-scale Propagation Modelling (LPM), Occlusion & Shadow Modelling (OSM), and Multipath Propagation Modelling (MPM) produce a theoretical RSRP baseline, an occlusion factor, and reflection embeddings.
3. **Prediction layer:** a conditional diffusion model integrates these representations with trajectory-aligned microenvironment features through two-stage training and prior guidance.

<!-- FIGURE 1: Insert the overall framework from paper Fig. 2 here.
Suggested file: figures/framework.png
![Overall framework of Channel-Diff](figures/framework.png)
-->
> **[Insert Figure 1 here]**

*Figure 1. Overall framework of Channel-Diff. Network parameters and multi-attribute urban maps feed the physics representation layer, which models large-scale propagation, occlusion and shadowing, and multipath geometry. The prediction layer combines these representations with microenvironment features to generate RSRP sequences through physics-guided conditional diffusion. Adapted from Fig. 2 of the paper.*

## Method

### Physical representations

| Component | Purpose | Input to this implementation |
| --- | --- | --- |
| LPM | Estimate the large-scale received-power trend using propagation models such as Hata and WINNER II. | Precomputed `calc` sequences. |
| OSM | Describe blockage using Fresnel-zone geometry and diffraction-based attenuation. | Occlusion information in `cond`; the last condition channel supplies the prior weight. |
| MPM | Represent dominant propagation paths using reflection geometry. | Precomputed `ref_emb` features at each trajectory point. |
| MFEN | Learn environmental features associated with local fading. | Ground-altitude and building-height maps (`mag`) and trajectory coordinates (`traj`). |

The training entry point consumes the precomputed physical representations in a dataset archive. It does not run propagation calculations, building polygonization, or reflection-path enumeration on raw maps.

### Denoising network and microenvironment features

The denoising backbone is a Transformer with diffusion-step conditioning through adaptive layer normalization. Signal, observation, mask, and conditional features are embedded before entering the Transformer blocks.

The default visual encoder, named `visual_DETR` in [visual_layer.py](visual_layer.py), implements the MFEN pipeline:

1. Enhance map edges using a Hessian-based residual term.
2. Extract spatial features using a modified ResNet18 backbone.
3. Add spatial positional embeddings and apply a Transformer encoder.
4. Sample local regions around trajectory points using the microenvironment serializer.
5. Pool and project the local features into a sequence of microenvironment embeddings.

In the student stage, the 20-dimensional reflection embedding is projected into the signal embedding space and added to the signal, observation, and mask embeddings. Network conditions and MFEN features are then concatenated with these embeddings. The teacher stage bypasses the visual feature computation.

<!-- FIGURE 2: Insert the model architecture from paper Fig. 3 here.
Suggested file: figures/architecture.png
![Two-stage diffusion and MFEN architecture](figures/architecture.png)
-->
> **[Insert Figure 2 here]**

*Figure 2. Physics-guided conditional diffusion architecture. Teacher-stage training learns large-scale propagation from calculated RSRP; student-stage training incorporates measured RSRP, reflection embeddings, and trajectory-aligned microenvironment features. Prior guidance connects the physical baseline to denoising. The MFEN combines edge enhancement, spatial encoding, and local feature serialization. Adapted from Fig. 3 of the paper.*

### Two-stage training and prior guidance

**Teacher stage.** Train the denoising model on `calc` using network conditions, learning the large-scale propagation baseline.

**Student stage.** Continue training the same model on measured `data`, enabling reflection and microenvironment features. In a complete training run, the student inherits the model at the end of teacher training. The separate `stu-train` mode instead loads a saved teacher checkpoint.

**Prior guidance in the current code.** The paper introduces this mechanism through noise prediction. This implementation defaults to `predict_xstart=True`. Let $H$ denote normalized `calc`, and let $\phi$ be the last channel of the normalized condition tensor. With `use_prior=True`, the student target becomes

$$
r_0 = x_0 - \phi H.
$$

At each reverse step, the network's residual estimate is converted back to an estimate of the complete signal by adding $\phi H$. In the sampling implementation, the residual estimate is clipped before this addition. Forward diffusion still adds noise to the complete $x_0$ sequence. The default `gamma=0` disables the additional physics-loss penalty; prior guidance remains enabled.

## Repository layout

```text
<project-root>/
├── datasets/
│   └── RSRP-image-3.npz
└── Channel-Diff/
    ├── main.py                         # Experiment configuration and entry point
    ├── DataLoader.py                   # NPZ loading, normalization, and splits
    ├── models_with_mask_scale.py       # Conditional DiT denoising network
    ├── Embed.py                        # Signal, mask, and positional embeddings
    ├── visual_layer.py                 # MFEN and alternative CNN/MLP encoders
    ├── train.py                        # Training, masking, and sample export
    ├── diffusion/                      # Diffusion objectives and samplers
    ├── utils.py                        # Argument parsing helpers
    ├── visualize_examples-gene.ipynb    # Experimental visualization notebook
    ├── load.py                         # Standalone exploratory script
    └── experiments/                    # Checkpoints and outputs, created for runs
```

The repository contains the `Channel-Diff/` code shown above. Prepare `datasets/` as a sibling directory; datasets, trained checkpoints, and the manuscript PDF are not included in this repository.

`load.py` is not part of the training pipeline. The model does not require a local `vit_model/` directory; the default visual encoder uses torchvision's ResNet18 weights.

## Setup

Use a Python environment with a compatible CUDA-enabled PyTorch/torchvision pair. The current training and sampling code contains explicit `.cuda()` calls and assumes an NVIDIA GPU. A pinned dependency environment is not provided in this directory.

After installing PyTorch and torchvision for your CUDA environment, install the remaining runtime dependencies:

```bash
python -m pip install numpy scikit-learn timm tensorboard setproctitle tqdm
```

For the visualization notebook, also install:

```bash
python -m pip install matplotlib pandas scipy jupyterlab
```

The default MFEN initializes `torchvision.models.resnet18(pretrained=True)`. Its weights must be available in the torchvision cache or downloadable when the model is first constructed.

Run commands from this directory because data and experiment paths are relative to the working directory:

```bash
cd Channel-Diff
mkdir -p experiments
```

## Data preparation

Place the processed dataset at `../datasets/<dataset_list>.npz`. The default configuration expects `../datasets/RSRP-image-3.npz`.

The supplied local archive has the following schema, verified from its array headers:

| Key | Shape | Description |
| --- | --- | --- |
| `data` | `(3600, 64)` | Measured RSRP sequences. |
| `calc` | `(3600, 64)` | Corresponding propagation-model estimates. |
| `cond` | `(3600, 10, 64)` | Conditions selected by `ray_condition='OF'`. |
| `cond_Nb` | `(3600, 10, 64)` | Alternative conditions selected by `ray_condition='Nb'`. |
| `mag` | `(3600, 2, 64, 64)` | Ground altitude in channel 0 and building height in channel 1. |
| `traj` | `(3600, 64, 2)` | Trajectory coordinates in the map grid, in the coordinate order expected by the serializer. |
| `ref_emb` | `(3600, 64, 20)` | Reflection features aligned with each trajectory. |

For a custom dataset, replace the sample count with $N$ and provide at least `time_length` aligned positions. The condition-channel count is inferred from the data; the reflection projection expects exactly 20 features. Trajectory coordinates must be grid coordinates suitable for map sampling, rather than unconverted geographic coordinates.

The loader clips RSRP values, fits an RSRP scaler on the training subset, normalizes conditions, and clips/normalizes map channels. It currently requires `ref_emb` whenever maps are enabled, even if `use_ref_emb=False`.

The paper evaluates RSRP-CPGMCM and RSRP-Image. RSRP-CPGMCM has no urban maps, while the paper constructs 3,600 RSRP-Image trajectories from 180 radio maps. These dataset descriptions and the original dataset references are provided in Section VI-A and references [68]–[69] of the paper. The quick start below targets the processed RSRP-Image configuration in this directory.

## Training

Edit the configuration block at the top of [main.py](main.py):

```python
dataset_list = 'RSRP-image-3'
model_size = 'DiT-S/8'
time_length = 64
task = 'generation'
prompt_state = 'train'

use_BH_map = True
use_AL_map = True
visual_model = 'DETR'
use_Hessian = True
use_prior = True
ray_condition = 'OF'
use_ref_emb = True
gamma = 0

teacher_epoches = 50
student_epoches = 320
```

Set the `CUDA_VISIBLE_DEVICES` assignment in `main.py` to the physical GPU you intend to use; the checked-in value is `'4'`. Leave `device_id='0'` to select the first visible GPU. An environment variable set only in the shell is overwritten by this assignment.

Start the two-stage run:

```bash
python main.py
```

The checked-in `prompt_state` is `'test'`, so change it to `'train'` before starting a new training run. Settings such as `prompt_state`, `save_folder`, `model_size`, and `time_length` are Python configuration variables, not command-line options. Other settings defined in `create_argparser()` can be overridden, for example:

```bash
python main.py --lr 0.0001 --log_interval 20
```

Keep dataset selection consistent by editing `dataset_list` directly: it is also used to name checkpoints and experiment folders. Keep the batch size consistent when loading checkpoints because the MFEN spatial positional parameter includes a batch dimension.

Each run writes to `experiments/<dataset>-t<teacher_epochs>-s<student_epochs>_P_<timestamp>/` with the default prior enabled. Outputs include:

| Output | Purpose |
| --- | --- |
| `set.txt` | Selected experiment settings. |
| `model_save/model_best_teacher_<dataset>.pkl` | Best teacher model state according to the validation objective. |
| `model_save/model_best_student_<dataset>.pkl` | Best student model state according to the validation objective. |
| `model_save/scaler_<dataset>.pk` | RSRP normalization parameters. |
| `model_save/idx_<dataset>.pk` | Train, validation, and test indices. |
| `model_save/loss_curve_teacher.npy`, `loss_curve_student.npy` | Saved training-loss histories. |
| TensorBoard event files and text logs | Training progress and validation objectives. |

```bash
tensorboard --logdir experiments
```

To train a student from an existing teacher checkpoint, set `prompt_state='stu-train'` and `save_folder` to its experiment folder name. Checkpoints store model parameters; this mode starts a new optimizer rather than resuming its previous state.

## Inference and visualization

Set the following values in `main.py`, using an existing checkpoint trained with matching model and data settings:

```python
prompt_state = 'test'
task = 'generation'
save_folder = '<existing-experiment-folder-name>'
```

The checkpoint must be located at:

```text
experiments/<save_folder>/model_save/model_best_student_<dataset_list>.pkl
```

Then run:

```bash
python main.py
```

Inference runs the 400-step reverse diffusion sampler and exports results into a **new** experiment folder under `data_save_student/`:

| Archive | Array key | Contents |
| --- | --- | --- |
| `generate.npz` | `gen_traffic` | Generated batches. For full generation, values are in the model's normalized space. |
| `target.npz` | `tar_traffic` | Reference RSRP batches in the original scale, after loader clipping. |
| `mask.npz` | `mask` | Target-position masks: 1 means generated, 0 means observed. |
| `cond.npz` | `cond` | Normalized conditioning features. |
| `datatype.npz` | `datatype` | Dataset identifiers for the exported batches. |

Use the saved scaler to inverse-transform generated values before comparing them with reference RSRP. The loader still expects a dataset containing `data` and `calc` in test mode; this entry point is an evaluation workflow, not a standalone API accepting only conditions.

The code also contains random-position and trailing-segment masks for imputation and forecasting experiments. The full-sequence generation setting above is the primary workflow documented here.

The [visualization notebook](visualize_examples-gene.ipynb) provides plotting and metric utilities. Update its `path` and `stage` (`'_student'`) before use. Its current postprocessing additionally rescales each generated sequence to the target's minimum and maximum; remove that target-dependent step when evaluating raw model predictions.

<!-- FIGURE 3: Insert the prediction comparison from paper Fig. 4 here.
Suggested file: figures/prediction_examples.png
![RSRP prediction examples](figures/prediction_examples.png)
-->
> **[Insert Figure 3 here]**

*Figure 3. Measured and predicted RSRP sequences from Channel-Diff and representative baselines. The examples compare the recovery of large-scale trends and local fluctuations on RSRP-CPGMCM and RSRP-Image. Adapted from Fig. 4 of the paper, where panels (a)–(d) show RSRP-CPGMCM and panels (e)–(h) show RSRP-Image.*

## Results reported in the paper

The following values are transcribed from Table II of the paper; they are not results from a new run of this code snapshot. Lower values are better for all three metrics.

| Dataset | JSD ↓ | NRMSE ↓ | MAE (dB) ↓ | Average relative improvement over the second-best model |
| --- | --- | --- | --- | --- |
| RSRP-CPGMCM | 0.1191 | 0.2029 | 5.681 | 37.19% over GBDT+DNN |
| RSRP-Image | 0.1401 | 0.2675 | 5.520 | 25.15% over EFEM |

The improvement column averages the relative improvements across JSD, NRMSE, and MAE. The paper further evaluates prior guidance, MFEN design, reflection embeddings, cross-dataset transfer, field of view, sequence length, positioning errors, and convergence efficiency.

<!-- FIGURE 4: Insert the architectural ablations from paper Fig. 7 here.
Suggested file: figures/ablation.png
![Ablation of MFEN and prior guidance](figures/ablation.png)
-->
> **[Insert Figure 4 here]**

*Figure 4. Ablation results on RSRP-Image for MFEN and Noise Prior Guidance (NPG), including removal of Hessian enhancement and replacement of MFEN with CNN or MLP encoders. The map variants isolate ground-altitude and building-height information and their combination. Adapted from Fig. 7 of the paper.*

<!-- FIGURE 5: Insert the training curves from paper Fig. 13 here.
Suggested file: figures/training_curves.png
![Training curves with different teacher-stage durations](figures/training_curves.png)
-->
> **[Insert Figure 5 here]**

*Figure 5. Training-loss curves on RSRP-CPGMCM with 0, 50, and 100 teacher epochs. The transition to measured-data supervision changes the loss scale, while physical pretraining improves the starting point and subsequent convergence of student training. Adapted from Fig. 13 of the paper.*

## Implementation and reproduction notes

The instructions above describe the current source snapshot. Several details should be aligned before using it to reproduce the paper's complete experiments:

- **Architecture settings:** `DiT-S/8` currently selects 6 Transformer blocks, hidden size 256, and 8 attention heads, with `t_patch_size=1`. Table III of the paper lists 12 blocks, 6 heads, and patch size 4. The code uses an $x_0$ prediction head with fixed diffusion variance. Do not infer the architecture or parameter count from the preset name alone.
- **Data protocol:** the loader caps most datasets at 2,000 samples (4,000 for `RSRP-mix`), uses the first 80% for training for RSRP-Image, and reuses the remaining subset for both validation and test. Test mode recomputes preprocessing instead of loading the training scaler and indices. These choices need alignment with the intended evaluation protocol.
- **Prior conventions:** `cond` is normalized to approximately $[-1,1]$, and its first channel is overwritten before normalization. The prior uses its final normalized channel directly; this differs from directly using the paper's physical occlusion factor in $[0,1]$. The default $x_0$ path is documented above. The optional noise-prediction branch multiplies the prior weight twice during sampling and requires correction before use.
- **Ablation and transfer modes:** `use_prior` and `use_ref_emb` are not fully independent in the data plumbing, and the sampling loop does not forward a disabled prior flag. Map-free loading also contains a NumPy-array comparison with `None` that needs correction when reflection arrays are present. Validate these paths before running prior-removal, map-free, zero-shot, or few-shot experiments.
- **Metrics:** training-time logs named RMSE summarize the normalized denoising objective rather than a complete generated-sequence benchmark. The notebook's `compute_JSD` returns SciPy's Jensen–Shannon distance, the square root of the divergence. For partial masks, the current export mixes normalized predictions with observations in the original scale. Align units, masks, preprocessing, and metric definitions before reporting results.
- **Randomness:** the current seed helper does not seed Python, NumPy, and PyTorch comprehensively, and the loader resets the NumPy seed from system entropy. Save the full configuration and control all random sources for repeatable experiments.

## Citation

Please cite the accompanying paper when using Channel-Diff:

> Xiaoqian Qi, Haoye Chai, Yue Wang, Zhaocheng Wang, and Yong Li. “Physics-guided Diffusion Models for Multi-scale Prediction of Reference Signal Received Power in Wireless Networks.” *IEEE Transactions on Mobile Computing*.

The supplied PDF does not provide a publication year, volume, issue, page range, or DOI for this article. Add the final publication metadata to the citation when available.
