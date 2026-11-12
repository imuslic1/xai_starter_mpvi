# Explainable knowledge distillation: images + five-star reviews

Two independent, runnable **Python script** experiments for Windows 11 using either an **Intel Arc GPU (XPU)** or an **NVIDIA GPU (CUDA)**. Each experiment has a **teacher**, a **standard supervised student**, and a **custom knowledge-distilled student**.

## Setup on Windows 11 (PowerShell)

### 1. Common prerequisites (both hardware configurations)

Install **Python 3.11 or 3.12** and current drivers for your **GPU**. Use a separate virtual environment per PyTorch backend; do **not** install CUDA and XPU wheels on top of each other in the same environment.

Open PowerShell in the project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If your terminal already shows `(.venv)`, you only need to activate the environment; do not recreate it. If Windows prevents PowerShell from activating scripts, you can use ` .\.venv\Scripts\python.exe ` in place of `python`, or temporarily set the execution policy for this terminal with `Set-ExecutionPolicy -Scope Process Bypass`. **Administrator rights are not required** for the Python installation steps.

### 2A. Intel CPU + Intel Arc GPU (native XPU; e.g. Core i5-14600K + Arc B580)

Install an up-to-date **Intel Arc graphics driver**, then install the PyTorch XPU build:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu
python -m pip install -r requirements.txt
python check_xpu.py
```

Expected result: `XPU available: True`, the Intel GPU name, and a successful matrix-multiplication check. Verify the same backend manually if needed:

```powershell
python -c "import torch; print('torch:', torch.__version__); print('XPU:', torch.xpu.is_available())"
```

Official documentation: [PyTorch on Intel GPUs](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html).

**Do not install CUDA wheels or the old `intel-extension-for-pytorch` for this native XPU workflow.** If XPU is unavailable, update drivers and check that your PyTorch build includes XPU. For a CPU-only fallback, use `--device cpu` (significantly slower).

### 2B. Intel CPU + NVIDIA GPU (CUDA; e.g. Core i5 + GeForce RTX)

CUDA requires a **compatible NVIDIA GPU and current NVIDIA driver**. The Intel CPU needs no special ML runtime. Check whether the NVIDIA driver detects your GPU:

```powershell
nvidia-smi
```

For example, the official **CUDA 12.6 PyTorch wheel** is a conservative Windows option when supported by your NVIDIA driver and GPU:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable')"
```

Expected result: `CUDA: True` and the name of your NVIDIA GPU. If this installation is incompatible with your driver or GPU, **choose the CUDA command recommended for your machine** using the official [PyTorch installation selector](https://pytorch.org/get-started/locally/) or [version-specific wheel list](https://pytorch.org/get-started/previous-versions/); do not assume that the newest CUDA wheel supports every older GPU.

You typically **do not need to install the full CUDA Toolkit** separately to run prebuilt PyTorch CUDA wheels; a compatible NVIDIA driver is required. The toolkit is relevant if you later compile CUDA extensions. `check_xpu.py` is an **Intel-only** check and is not appropriate for NVIDIA validation.

### 3. Selecting a backend and running the scripts

The shared `common.py` device selection already supports both stacks:

| `--device` | Behavior |
|---|---|
| `auto` (default) | Choose available **XPU** first, then **CUDA**, then CPU |
| `xpu` | Force native Intel GPU backend (requires XPU wheels and Intel GPU) |
| `cuda` | Force NVIDIA CUDA backend (requires CUDA-enabled PyTorch and supported NVIDIA GPU) |
| `cpu` | Run on CPU only |

**Intel Arc example:**

```powershell
python images.py --device xpu --epochs 4 --kd-epochs 8 --batch 16
python text_reviews.py --device xpu --train-size 3000 --epochs 2 --batch 8
```

**NVIDIA example:**

```powershell
python images.py --device cuda --epochs 4 --kd-epochs 8 --batch 16
python text_reviews.py --device cuda --train-size 3000 --epochs 2 --batch 8
```

**Notes:**

- On a PC with both GPUs, `--device auto` prefers Intel XPU if that backend is visible. Pass `--device cuda` explicitly to select NVIDIA.
- Switching from an XPU environment to CUDA (or vice versa) is best done in a **new virtual environment**, installing only the relevant torch/torchvision wheels before `requirements.txt`.
- Both scripts use their selected backend for training and evaluation. **Grad-CAM in `images.py` runs on the CPU intentionally**, for consistent backward-hook handling, regardless of which GPU trained the models.
- Initial pretrained model and dataset downloads require internet access; checkpoints and generated results are stored under `outputs/`.
- On Windows, Hugging Face's symlink-cache warning is normally harmless. Windows Developer Mode can enable symlinks without launching the terminal as Administrator. Optional `hf_xet` may speed some downloads but is not required.

## Quick start: Windows batch launchers (no commands to remember)

After following **Setup 1 + either 2A (Intel Arc) or 2B (NVIDIA CUDA)** above **once**, you can launch either experiment by double-clicking a `.bat` file in the project folder. The `.bat` files **do not install PyTorch or switch between XPU and CUDA wheels**; they run the already-configured Python environment for the intended GPU. No Administrator privileges are needed.

| Windows launcher | Hardware | Experiment | Default arguments |
|---|---|---|---|
| `run_images_intel.bat` | Intel Arc GPU (XPU) | Bean disease / Grad-CAM | `--epochs 4 --kd-epochs 8 --batch 16` |
| `run_text_intel.bat` | Intel Arc GPU (XPU) | Yelp sentiment | `--train-size 3000 --epochs 2 --batch 8` |
| `run_images_cuda.bat` | NVIDIA GPU (CUDA) | Bean disease / Grad-CAM | `--epochs 4 --kd-epochs 8 --batch 16` |
| `run_text_cuda.bat` | NVIDIA GPU (CUDA) | Yelp sentiment | `--train-size 3000 --epochs 2 --batch 8` |

**How to use them:**

1. Create/install the correct virtual environment once, using the manual setup instructions above.
2. Double-click the matching `.bat` file (or run it from `cmd.exe` or PowerShell). A console window shows device verification, training epochs, evaluation and output paths; it remains open when the run finishes or fails.
3. The launcher checks that the requested GPU backend is available **before** starting. If it fails, follow the manual environment setup for that GPU. It will **not silently fall back to CPU**.

The launchers look for Python in this order: an already activated `VIRTUAL_ENV`, `.venv` in the project folder, then `.venv` in the parent folder. **If you maintain separate CUDA and XPU environments on the same PC, activate the appropriate environment before launching**; the scripts do not automatically pick a separate environment based on the GPU. For example, in PowerShell:

```powershell
# From the project folder, after installing CUDA wheels in a separate environment:
..\.venv-cuda\Scripts\Activate.ps1
.\run_images_cuda.bat
```

Optional CLI arguments are forwarded to the Python script. For example, in PowerShell:

```powershell
.\run_images_intel.bat --skip-train
.\run_text_cuda.bat --train-size 5000 --epochs 3 --batch 4
```

The launchers explicitly set `--device xpu` or `--device cuda`. Python's argument parser uses the later value for duplicate options, so additional epoch/batch arguments override the launcher defaults. Existing saved checkpoints are automatically reused; to retrain from scratch, move or delete the relevant checkpoints first.

The **manual `python ... --device ...` commands remain fully supported**; use them for advanced configurations, CPU fallback or debugging.

## Experiment 1: bean leaf disease and Grad-CAM

```powershell
python images.py --epochs 4 --kd-epochs 8 --batch 16
```

Dataset: https://huggingface.co/datasets/AI-Lab-Makerere/beans (smartphone images of bean leaves, labeled angular leaf spot / bean rust / healthy, 1,034 train / 133 validation / 128 test).

* Teacher: ImageNet-pretrained ResNet-50, fine-tuned on beans.
* Conventional student: ImageNet-pretrained MobileNetV3-small, fine-tuned with true labels.
* Self-made distilled student: ImageNet-pretrained ResNet-18 architecture, further fine-tuned on ground truth plus teacher soft targets using temperature=4, alpha=0.5.
* XAI: class-targeted Grad-CAM (same teacher-predicted class across all three). Save image overlays, top-20% heatmap IoU, Spearman heatmap correlation, and a 20%-pixel deletion faithfulness score.

Note: initializing a student from ImageNet weights **does not disqualify it as self-distilled**: its task training is performed with your own knowledge distillation. To attempt more difficult full-from-scratch training, change `make_model('distilled',pretrained=True)` to `pretrained=False` in `images.py` and increase epochs.

Outputs: `outputs/images/metrics.json`, `outputs/images/explainability.csv`, `outputs/images/gradcam_*.png`, `.pt` weights.

Rerun models without training:

```powershell
python images.py --skip-train
```

## Experiment 2: Yelp rating vs review language

```powershell
python text_reviews.py --train-size 3000 --epochs 2 --batch 8
```

Dataset: https://huggingface.co/datasets/Yelp/yelp_review_full (1–5 stars; labels stored 0–4; 650k train, 50k test). Loading the dataset downloads the published dataset first, then takes a **reproducibly shuffled subset** for rapid experimentation.

* Teacher (already five-star task-trained): `nlptown/bert-base-multilingual-uncased-sentiment` (note its prior domain is not Yelp; domain shift must be evaluated).
* Conventional student: `distilbert/distilbert-base-uncased`, five-star classifier fine-tuned on Yelp labels (pretrained language model, but classifier head is new).
* Self-distilled student: `google/bert_uncased_L-4_H-256_A-4`, compact pretrained BERT fine-tuned with your own ground-truth + teacher soft-label KD loss.
* XAI: aligned whitespace word **leave-one-out occlusion** on the first 40 words. Signed scores reveal positive or negative evidence for the **same teacher-selected target star class**. Save three aligned attribution charts per review, teacher-student Spearman correlation and a 20% top-word deletion test. Prioritize examples where the teacher rating differs from human rating by at least two stars.

This is **word occlusion**, not SHAP or Integrated Gradients. It is intentionally easier to debug, requires no gradients and works across distinct tokenizers and on XPU. For a thesis/project, add SHAP or Integrated Gradients after the baseline runs.

Outputs: `outputs/text/metrics.json`, `outputs/text/explainability.csv`, `outputs/text/word_attributions_*.png`, `outputs/text/examples.json`, saved student checkpoints.

Rerun without training:

```powershell
python text_reviews.py --skip-train
```

## Interpreting results and keeping the evaluation rigorous

* Accuracy, macro-F1, and star mean absolute error assess model quality. Parameters assess compression, but real inference timing and saved model bytes should also be measured before making efficiency claims.
* Explanation similarity is **not explanation correctness**. High similarity can mean inherited shortcut learning, and low similarity can come from different valid cues. The deletion test adds partial evidence of faithfulness but does not establish causal reasoning.
* Image overlays use the **same predicted class** so they remain comparable, even when student argmax predictions differ.
* Word deletion can introduce out-of-distribution inputs; token counts may change after removing words. Treat this as a methodological limitation.
* A five-star mismatch is **not** necessarily a model failure. The example "waiter was nice, but the meal was cold" contains mixed aspects. For a future upgrade, add aspect-based sentiment labels or manually annotate a small review subset.
* **Research-grade protocol:** do not tune against test data; use validation only for hyperparameters; evaluate held-out test once. Run 3–5 independent seeds, compute confidence intervals and review attributions with human qualitative checks. Current scripts are runnable **starter experiments**, not validated research results.

## Performance and compatibility notes

* The scripts use eager PyTorch (no `torch.compile`), FP32, small batches, zero DataLoader workers; these conservative choices avoid many Windows GPU-specific issues.
* For unsupported GPU kernels or memory errors, reduce `--batch` to 4 or 8 or try `--device cpu`. CPU may be substantially slower.
* Model downloads and Hugging Face dataset downloads require internet access the first time.
* `images.py` keeps explainability on CPU for reliable autograd and visualization, even if training used XPU; `text_reviews.py` uses the selected training backend for occlusion when possible.
* The same models are reused on subsequent runs from saved checkpoints. Delete output subfolders if you intentionally want to retrain from scratch.

## Suggested academic presentation

Question: **Does knowledge distillation preserve evidence used for decisions, as well as predictive performance?**

Show three columns (teacher, conventional student, distilled student), one performance table, one attribution visual, one explanation agreement plot and one deletion-faithfulness plot. Discuss target class alignment, uncertainty, shortcut learning, dataset limitations and model domain shift.
