# Explainable knowledge distillation: images + five-star reviews

Two independent, runnable **Python script** experiments for Windows 11 and Intel Arc B580.
Each has a **teacher**, a **standard supervised student**, and a **custom knowledge-distilled student**.

## Setup (PowerShell)

Install the latest Intel Arc graphics driver first. Python 3.11/3.12, clean virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu
python -m pip install -r requirements.txt
python check_xpu.py
```

Official PyTorch XPU instructions: https://docs.pytorch.org/get-started/additional-platforms/

**Do not install CUDA wheels or old `intel-extension-for-pytorch`** for this native XPU workflow. If `check_xpu.py` says XPU is unavailable, check Windows 11, graphics drivers, and whether the PyTorch version supports the B580. Both experiments can run with `--device cpu` at reduced speed.

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

* The scripts use eager PyTorch (no `torch.compile`), FP32, small batches, zero DataLoader workers; these conservative choices avoid many Windows/XPU-specific issues.
* For incompatible XPU backward kernels or memory errors, try `--device cpu` or reduce `--batch` to 4 or 8. CPU may be substantially slower.
* Model downloads and Hugging Face dataset downloads require internet access the first time.
* `images.py` keeps explainability on CPU for reliable autograd and visualization, even if training used XPU; `text_reviews.py` uses XPU inference for occlusion by default.
* The same models are reused on subsequent runs from saved checkpoints. Delete output subfolders if you intentionally want to retrain from scratch.

## Suggested academic presentation

Question: **Does knowledge distillation preserve evidence used for decisions, as well as predictive performance?**

Show three columns (teacher, conventional student, distilled student), one performance table, one attribution visual, one explanation agreement plot and one deletion-faithfulness plot. Discuss target class alignment, uncertainty, shortcut learning, dataset limitations and model domain shift.
