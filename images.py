
"""Bean leaf disease classification, knowledge distillation and Grad-CAM.

Teacher: ResNet-50
Baseline: MobileNetV3-Small
Distilled student: ResNet-18

Run:
    python images.py --epochs 4 --kd-epochs 8 --batch 16
    python images.py --skip-train --device xpu

Output:
    outputs/images/dashboard.html
"""

import argparse
import base64
import csv
import json
import os
import webbrowser
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import spearmanr
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from common import device_from_arg, seed_all, params_m, train_step


CLASSES = ["angular_leaf_spot", "bean_rust", "healthy"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

MODEL_LABELS = {
    "teacher": "Teacher (ResNet-50)",
    "baseline": "Baseline (MobileNetV3-Small)",
    "distilled": "Distilled (ResNet-18)",
}


# ============================================================
# TERMINAL FORMATTING
# ============================================================

def section(title):
    print("\n" + "=" * 78, flush=True)
    print(f"  {title}", flush=True)
    print("=" * 78, flush=True)


def info(message):
    print(f"  > {message}", flush=True)


def print_results(results):
    section("FINAL TEST-SET COMPARISON")

    print(
        f"  {'Model':<32} "
        f"{'Accuracy':>10} "
        f"{'CE loss':>10} "
        f"{'Params (M)':>12}"
    )
    print("  " + "-" * 66)

    for kind, stats in results.items():
        print(
            f"  {MODEL_LABELS[kind]:<32} "
            f"{stats['accuracy'] * 100:>9.2f}% "
            f"{stats['cross_entropy']:>10.4f} "
            f"{stats['parameters_million']:>12.3f}"
        )

    info("Cross-entropy: lower is better.")
    info("Accuracy: higher is better.")


# ============================================================
# DATASET
# ============================================================

class Beans(Dataset):
    def __init__(self, split, transform):
        self.data = load_dataset(
            "AI-Lab-Makerere/beans",
            split=split
        )
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        image = row["image"].convert("RGB")
        return self.transform(image), int(row["labels"])


# ============================================================
# MODELS
# ============================================================

def make_model(kind, pretrained=True):
    if kind == "teacher":
        model = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
            if pretrained else None
        )
        model.fc = nn.Linear(model.fc.in_features, 3)

    elif kind == "baseline":
        model = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
            if pretrained else None
        )
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features, 3
        )

    else:
        # ImageNet initialization makes small-data KD practical.
        model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
            if pretrained else None
        )
        model.fc = nn.Linear(model.fc.in_features, 3)

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader, device):
    model.eval()
    n = 0
    correct = 0
    total_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            total_loss += float(
                F.cross_entropy(
                    logits, y, reduction="sum"
                ).cpu()
            )

            correct += int(
                (logits.argmax(1) == y).sum().cpu()
            )

            n += len(y)

    return {
        "accuracy": round(correct / n, 4),
        "cross_entropy": round(total_loss / n, 4),
        "n": n
    }


# ============================================================
# TRAINING + HISTORY
# ============================================================

def train(
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    lr,
    teacher=None
):
    model.to(device)

    if teacher is not None:
        teacher.to(device).eval()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.01
    )

    best_accuracy = -1
    best_state = None
    history = []

    for epoch in range(epochs):
        model.train()
        losses = []

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            if teacher is None:
                logits = model(x)
                loss = F.cross_entropy(logits, y)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                losses.append(float(loss.detach().cpu()))
            else:
                losses.append(
                    train_step(
                        model, teacher, x, y, optimizer
                    )
                )

        stats = evaluate(model, val_loader, device)
        mean_loss = float(np.mean(losses))

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(mean_loss, 5),
            "val_accuracy": stats["accuracy"],
            "val_cross_entropy": stats["cross_entropy"]
        })

        improved = stats["accuracy"] > best_accuracy

        print(
            f"  Epoch {epoch + 1:>2}/{epochs:<2} | "
            f"train loss {mean_loss:.4f} | "
            f"val accuracy {stats['accuracy'] * 100:6.2f}% | "
            f"val CE {stats['cross_entropy']:.4f} | "
            f"{'BEST' if improved else '----'}",
            flush=True
        )

        if improved:
            best_accuracy = stats["accuracy"]
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    model.load_state_dict(best_state)

    info(
        "Restored best validation checkpoint "
        f"({best_accuracy * 100:.2f}% accuracy)."
    )

    return history


# ============================================================
# GRAD-CAM
# ============================================================

def cam(model, x, kind, target):
    """Compute Grad-CAM using CPU hooks for portability."""

    model.eval().cpu()

    if kind == "baseline":
        layer = model.features[-1]
    else:
        layer = model.layer4[-1]

    saved = {}

    def hook(_, __, output):
        saved["activation"] = output
        output.register_hook(
            lambda gradient: saved.update(
                gradient=gradient
            )
        )

    handle = layer.register_forward_hook(hook)

    try:
        output = model(x.cpu())
        model.zero_grad(set_to_none=True)

        output[0, target].backward()

        activations = saved["activation"].detach()
        gradients = saved["gradient"].detach()

        weights = gradients.mean(
            (2, 3), keepdim=True
        )

        heat = F.relu(
            (activations * weights).sum(
                1, keepdim=True
            )
        )

        heat = F.interpolate(
            heat,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False
        )[0, 0].numpy()

        heat = (
            (heat - heat.min()) /
            (heat.max() - heat.min() + 1e-8)
        )

        return heat

    finally:
        handle.remove()


def top_iou(a, b, fraction=0.2):
    mask_a = a >= np.quantile(a, 1 - fraction)
    mask_b = b >= np.quantile(b, 1 - fraction)

    intersection = np.logical_and(
        mask_a, mask_b
    ).sum()

    union = np.logical_or(
        mask_a, mask_b
    ).sum()

    return float(intersection / max(union, 1))


def deletion_drop(
    model,
    x,
    heat,
    target,
    fraction=0.2
):
    """Measure probability change after masking salient pixels."""

    model.eval().cpu()

    with torch.no_grad():
        before = F.softmax(
            model(x), dim=-1
        )[0, target].item()

        mask = torch.tensor(
            heat >= np.quantile(
                heat, 1 - fraction
            )
        )

        altered = x.clone()
        altered[0, :, mask] = 0

        after = F.softmax(
            model(altered), dim=-1
        )[0, target].item()

    return round(before - after, 4)


def select_balanced_examples(data, examples_per_class=5, seed=42):
    """Select reproducible test samples from every class."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(data.data["labels"])
    selected = []

    for class_id, class_name in enumerate(CLASSES):
        candidates = np.flatnonzero(labels == class_id)

        n = min(examples_per_class, len(candidates))
        indices = rng.choice(
            candidates, size=n, replace=False
        ).tolist()

        selected.append(indices)
        info(f"{class_name}: selected {n} images")

    # Interleave classes: 0, 1, 2, 0, 1, 2, ...
    return [
        group[i]
        for i in range(max(map(len, selected)))
        for group in selected
        if i < len(group)
    ]


# ============================================================
# EXPLAINABILITY OUTPUTS
# ============================================================


def explain(models_dict, data, out, examples_per_class=5):
    """Generate balanced Grad-CAM examples and save all model predictions."""

    names = list(models_dict)
    rows = []
    prediction_records = {}

    selected_indices = select_balanced_examples(
        data,
        examples_per_class=examples_per_class
    )

    count = len(selected_indices)

    info(f"Generating Grad-CAM explanations for {count} test images.")

    for i, dataset_index in enumerate(selected_indices):
        x, y = data[dataset_index]
        x = x.unsqueeze(0)

        # Predict the class independently with each model.
        with torch.no_grad():
            predictions = {
                name: int(
                    model.cpu().eval()(x).argmax(1).item()
                )
                for name, model in models_dict.items()
            }

        teacher_pred = predictions["teacher"]

        # Store predictions using readable class names.
        prediction_records[str(i)] = {
            name: CLASSES[pred]
            for name, pred in predictions.items()
        }

        # All Grad-CAM maps use the teacher's target class
        # so the explanations remain directly comparable.
        heatmaps = {
            kind: cam(model, x, kind, teacher_pred)
            for kind, model in models_dict.items()
        }

        image = (
            x[0].permute(1, 2, 0).numpy()
            * np.array(STD)
            + np.array(MEAN)
        )
        image = np.clip(image, 0, 1)

        fig, axes = plt.subplots(
            1, 4, figsize=(13, 3.5)
        )

        axes[0].imshow(image)
        axes[0].set_title(f"True: {CLASSES[y]}")

        for axis, (name, heat) in zip(
            axes[1:], heatmaps.items()
        ):
            axis.imshow(image)
            axis.imshow(
                heat,
                cmap="jet",
                alpha=0.45,
                vmin=0,
                vmax=1
            )
            axis.set_title(name)

        for axis in axes:
            axis.axis("off")

        fig.suptitle(
            f"Grad-CAM | Teacher target: {CLASSES[teacher_pred]}"
        )
        fig.tight_layout()

        image_path = out / f"gradcam_{i:02d}.png"

        fig.savefig(image_path, dpi=150)
        plt.close(fig)

        info(
            f"Image {i + 1:>2}/{count} | "
            f"True: {CLASSES[y]} | "
            f"Teacher: {CLASSES[predictions['teacher']]} | "
            f"Baseline: {CLASSES[predictions['baseline']]} | "
            f"Distilled: {CLASSES[predictions['distilled']]}"
        )

        # Calculate XAI metrics for all three models.
        for name in names:
            heat = heatmaps[name]

            correlation = spearmanr(
                heatmaps["teacher"].ravel(),
                heat.ravel()
            ).statistic

            rows.append({
                "sample": i,
                "true": CLASSES[y],
                "teacher_target": CLASSES[teacher_pred],
                "model": name,
                "iou_top20_vs_teacher": round(
                    top_iou(heatmaps["teacher"], heat), 4
                ),
                "spearman_vs_teacher": round(
                    float(correlation)
                    if np.isfinite(correlation)
                    else 0.0,
                    4
                ),
                "deletion_drop_top20": deletion_drop(
                    models_dict[name],
                    x,
                    heat,
                    teacher_pred
                )
            })

    # Save explainability measurements.
    with (out / "explainability.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)

    # Save predictions separately from the XAI metrics.
    (out / "predictions.json").write_text(
        json.dumps(prediction_records, indent=2),
        encoding="utf-8"
    )

    info("Saved explainability.csv and predictions.json")


# ============================================================
# HTML DASHBOARD TEMPLATE
# ============================================================

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport"
      content="width=device-width, initial-scale=1">

<title>LeafLens | XAI Dashboard</title>

<style>
:root {
    color-scheme: dark;
    --bg: #09101e;
    --panel: #152238;
    --border: #2c405e;
    --muted: #a1b2ca;
    --ink: #f2f7ff;
    --a: #4ce1d2;
    --b: #84aaff;
    --c: #c9a4ff;
}

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font: 15px/1.55 system-ui, Segoe UI, sans-serif;
    color: var(--ink);
    background:
        radial-gradient(
            ellipse at 90% 0%,
            #233b62,
            transparent 55%
        ),
        var(--bg);
}

main {
    max-width: 1180px;
    margin: auto;
    padding: 30px 24px 80px;
}

header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border);
    padding-bottom: 22px;
}

header b {
    letter-spacing: .12em;
}

header span,
.muted {
    color: var(--muted);
}

h1 {
    font-size: clamp(36px, 6vw, 65px);
    letter-spacing: -.06em;
    line-height: 1.08;
    margin: 50px 0 15px;
}

h1 em {
    color: var(--a);
    font-style: normal;
}

h2 {
    font-size: 23px;
    margin: 44px 0 16px;
}

p {
    color: var(--muted);
}

.grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
}

.two {
    grid-template-columns: 1fr 1fr;
}

.card {
    background: linear-gradient(
        145deg,
        #1a2b45,
        var(--panel)
    );
    border: 1px solid var(--border);
    border-radius: 19px;
    padding: 23px;
    box-shadow: 0 12px 30px #0002;
    min-width: 0;
}

.big {
    font-size: 36px;
    font-weight: 800;
    letter-spacing: -.04em;
    margin: 12px 0;
}

.small {
    font-size: 12px;
    color: var(--muted);
}

.bar {
    height: 13px;
    border-radius: 9px;
    background: #293b53;
    overflow: hidden;
    margin: 9px 0 20px;
}

.fill {
    height: 100%;
    width: 0;
    border-radius: 9px;
    transition:
        width 1.4s cubic-bezier(.2, .8, .2, 1);
}

.line {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    font-size: 13px;
}

.line strong {
    font-variant-numeric: tabular-nums;
}

.legend {
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
    font-size: 13px;
    color: var(--muted);
}

.legend i {
    display: inline-block;
    height: 9px;
    width: 9px;
    border-radius: 50%;
    margin-right: 6px;
}

svg {
    width: 100%;
    height: auto;
    display: block;
}

.heat {
    width: 100%;
    border-radius: 12px;
    border: 1px solid var(--border);
}

select,
button {
    background: #21334f;
    color: white;
    border: 1px solid #46607b;
    border-radius: 10px;
    padding: 10px 14px;
    cursor: pointer;
    font: inherit;
}

button:hover,
button.active {
    background: #1d535b;
    border-color: var(--a);
}

.tabs {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}

.metrics {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin: 18px 0;
}

.mini {
    background: #0c192c;
    padding: 15px;
    border-radius: 12px;
}

.mini strong {
    display: block;
    font-size: 24px;
}

.report {
    border-left: 3px solid var(--a);
    padding: 4px 18px;
    margin: 16px 0;
}

.report h3 {
    margin: 4px 0;
}

.tablewrap {
    overflow-x: auto;
}

table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}

th,
td {
    padding: 12px;
    border-bottom: 1px solid var(--border);
    text-align: left;
}

th {
    color: var(--muted);
}

.fade {
    animation: rise .8s both;
}

@keyframes rise {
    from {
        opacity: 0;
        transform: translateY(16px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@media (max-width: 800px) {
    .grid,
    .two {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 450px) {
    .metrics {
        grid-template-columns: 1fr;
    }
    main {
        padding: 18px;
    }
}

@media (prefers-reduced-motion: reduce) {
    *,
    *:before,
    *:after {
        animation: none !important;
        transition: none !important;
    }
}
</style>
</head>

<body>
<main>

<header>
    <b>◈ LEAFLENS <span>/ XAI LAB</span></b>
    <span>Offline experiment report</span>
</header>

<section class="fade">
    <h1>
        Beyond prediction.<br>
        <em>Into explanation.</em>
    </h1>
    <p>
        Knowledge distillation and Grad-CAM ·
        Bean leaf disease classification ·
        Three-model comparison
    </p>
</section>

<div id="cards" class="grid fade"></div>

<h2>01 / Performance comparison</h2>

<div class="grid two">
    <div class="card">
        <h3>Test accuracy</h3>
        <div id="accuracy"></div>
    </div>
    <div class="card">
        <h3>Parameter count (millions)</h3>
        <div id="size"></div>
    </div>
</div>

<h2>02 / Training history</h2>

<div class="card">
    <div id="curves"></div>
    <p class="small" id="history-note"></p>
</div>

<h2>03 / Grad-CAM explorer</h2>

<div class="card">
    <label for="sample">Select test image: </label>
    <select id="sample"></select>

    <p id="sample-label"></p>

    <img
        id="heatmap"
        class="heat"
        alt="Original image and Grad-CAM overlays"
    >

    <p class="small">
        Left to right: original, teacher, supervised
        baseline, distilled student. Every map targets
        the teacher-predicted class.
    </p>

    <div id="tabs" class="tabs"></div>
    <div class="metrics" id="xai-metrics"></div>
    <p id="xai-note" class="small"></p>
</div>

<h2>04 / Interpretation</h2>

<div class="card" id="interpret"></div>

<h2>05 / Detailed explanation measurements</h2>

<div class="card tablewrap">
    <button id="csv">Export CSV</button>

    <table>
        <thead>
            <tr>
                <th>Image</th>
                <th>True class</th>
                <th>Teacher target</th>
                <th>Model</th>
                <th>Top-20% IoU</th>
                <th>Spearman ρ</th>
                <th>Deletion Δp</th>
            </tr>
        </thead>
        <tbody id="table"></tbody>
    </table>
</div>

<p class="small">
    Important: Similar heatmaps are not evidence of
    identical reasoning. Deletion Δp is a single
    perturbation test, not a complete insertion/deletion
    curve. Only eight images are explained; conclusions
    about explanation agreement are exploratory.
</p>

</main>

<script id="data" type="application/json">__DATA__</script>

<script>
(() => {
    'use strict';

    const D = JSON.parse(
        document.getElementById('data').textContent
    );

    const $ = id => document.getElementById(id);

    const names = [
        'teacher',
        'baseline',
        'distilled'
    ];

    const colors = {
        teacher: '#4ce1d2',
        baseline: '#84aaff',
        distilled: '#c9a4ff'
    };

    const short = {
        teacher: 'Teacher',
        baseline: 'Baseline',
        distilled: 'Distilled'
    };

    const fmt = (x, n = 3) => Number(x).toFixed(n);
    const pct = x => (100 * x).toFixed(2) + '%';

    const esc = x => String(x).replace(
        /[&<>"']/g,
        c => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#39;'
        }[c])
    );

    const metric = D.metrics;

    const best = names.reduce(
        (a, b) => metric[b].accuracy > metric[a].accuracy
            ? b : a
    );

    // Summary cards
    $('cards').innerHTML = names.map(n => `
        <div class="card">
            <span class="small">${esc(D.labels[n])}</span>
            <div class="big" style="color:${colors[n]}">
                ${pct(metric[n].accuracy)}
            </div>
            <div class="small">
                Test accuracy ·
                ${fmt(metric[n].parameters_million, 2)}M
                parameters ·
                CE ${fmt(metric[n].cross_entropy, 4)}
            </div>
        </div>
    `).join('');

    // Animated bars
    function bars(id, key, scale, suffix) {
        const maximum = Math.max(
            ...names.map(n => metric[n][key])
        );

        $(id).innerHTML = names.map(n => `
            <div class="line">
                <span>${short[n]}</span>
                <strong>
                    ${fmt(metric[n][key] * scale, 2)}
                    ${suffix}
                </strong>
            </div>
            <div class="bar">
                <div
                    class="fill"
                    data-width="${
                        maximum
                            ? metric[n][key] / maximum * 100
                            : 0
                    }"
                    style="background:${colors[n]}"
                ></div>
            </div>
        `).join('');
    }

    bars('accuracy', 'accuracy', 100, '%');
    bars('size', 'parameters_million', 1, 'M');

    requestAnimationFrame(() => {
        setTimeout(() => {
            document.querySelectorAll('.fill').forEach(
                e => e.style.width = e.dataset.width + '%'
            );
        }, 80);
    });

    // Training history
    const histories = D.history || {};

    const available = names.filter(
        n => Array.isArray(histories[n]) &&
             histories[n].length
    );

    if (!available.length) {
        $('curves').textContent =
            'No saved training history. Run training once ' +
            'with the updated script to record epoch values.';

        $('history-note').textContent =
            'Existing model checkpoints do not contain ' +
            'past epoch histories.';
    } else {
        const series = available.flatMap(
            n => histories[n]
        );

        const maxEpoch = Math.max(
            2, ...series.map(r => r.epoch)
        );

        const lo = Math.max(
            0,
            Math.floor(
                (
                    Math.min(
                        ...series.map(
                            r => r.val_accuracy * 100
                        )
                    ) - 5
                ) / 10
            ) * 10
        );

        const hi = Math.min(
            100,
            Math.max(
                lo + 10,
                Math.ceil(
                    (
                        Math.max(
                            ...series.map(
                                r => r.val_accuracy * 100
                            )
                        ) + 3
                    ) / 10
                ) * 10
            )
        );

        const X = e =>
            50 + (e - 1) / (maxEpoch - 1) * 810;

        const Y = v =>
            200 - (v - lo) / (hi - lo) * 170;

        let svg = `
            <svg viewBox="0 0 900 240"
                 aria-label="Validation accuracy by epoch"
                 role="img">
        `;

        for (let i = 0; i <= 4; i++) {
            const value = lo + (hi - lo) * i / 4;

            svg += `
                <line
                    x1="50" y1="${Y(value)}"
                    x2="860" y2="${Y(value)}"
                    stroke="#344962"
                    stroke-dasharray="4 5"
                />
                <text
                    x="44" y="${Y(value) + 4}"
                    font-size="12"
                    fill="#a1b2ca"
                    text-anchor="end"
                >${fmt(value, 0)}%</text>
            `;
        }

        for (const n of available) {
            const points = histories[n].map(
                r => `${X(r.epoch)},${Y(r.val_accuracy * 100)}`
            ).join(' ');

            svg += `
                <polyline
                    class="trace"
                    points="${points}"
                    fill="none"
                    stroke="${colors[n]}"
                    stroke-width="3.5"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                />
            `;
        }

        svg += '</svg>';

        $('curves').innerHTML =
            '<div class="legend">' +
            available.map(n => `
                <span>
                    <i style="background:${colors[n]}"></i>
                    ${short[n]}
                </span>
            `).join('') +
            '</div>' + svg;

        $('history-note').textContent =
            'Validation accuracy is recorded after every ' +
            'training epoch. Models are evaluated using ' +
            'their best validation checkpoints.';

        document.querySelectorAll('.trace').forEach(
            e => {
                const length = e.getTotalLength();
                e.style.strokeDasharray = length;
                e.style.strokeDashoffset = length;
                e.style.transition =
                    'stroke-dashoffset 1.6s ease';

                requestAnimationFrame(
                    () => e.style.strokeDashoffset = 0
                );
            }
        );
    }

    // Grad-CAM gallery
    let sample = D.samples[0]?.id ?? 0;
    let model = 'distilled';

    $('sample').innerHTML = D.samples.map(s => `
        <option value="${s.id}">
            Image ${s.id + 1} -
            ${esc(s.true.replaceAll('_', ' '))}
        </option>
    `).join('');

    $('tabs').innerHTML = names.map(n => `
        <button type="button" data-model="${n}">
            ${short[n]}
        </button>
    `).join('');

    
function render() {
    const s = D.samples.find(s => s.id === sample);
    if (!s) return;

    // Display the selected Grad-CAM image.
    $('heatmap').src = s.image;

    const readable = name =>
        esc(name.replaceAll('_', ' '));

    // Prediction row with correctness indicator.
    const predictionRow = (label, prediction) => {
        const correct = prediction === s.true;

        const color = correct ? '#4ce1d2' : '#ff7497';
        const symbol = correct ? '✓' : '✗';

        return `
            <div style="
                display:flex;
                justify-content:space-between;
                align-items:center;
                gap:12px;
                padding:10px 0;
                border-bottom:1px solid #2c405e;
            ">
                <span>${label}</span>

                <strong style="color:${color}">
                    ${readable(prediction)}
                    &nbsp;${symbol}
                </strong>
            </div>
        `;
    };

    // Display actual class and every model's prediction.
    $('sample-label').innerHTML = `
        <div style="
            background:#0c192c;
            padding:18px;
            border-radius:12px;
            margin:18px 0;
        ">
            <div style="margin-bottom:12px">
                <span style="color:#a1b2ca">
                    ACTUAL CLASS
                </span>

                <div style="
                    font-size:20px;
                    font-weight:700;
                    margin-top:5px;
                ">
                    ${readable(s.true)}
                </div>
            </div>

            ${predictionRow(
                'Teacher (ResNet-50)',
                s.predictions.teacher
            )}

            ${predictionRow(
                'Baseline (MobileNetV3)',
                s.predictions.baseline
            )}

            ${predictionRow(
                'Distilled (ResNet-18)',
                s.predictions.distilled
            )}
        </div>
    `;

    // Highlight the currently selected model tab.
    $('tabs').querySelectorAll('button').forEach(
        b => b.classList.toggle(
            'active',
            b.dataset.model === model
        )
    );

    // Display Grad-CAM metrics for selected model.
    const r = s.rows[model];
    if (!r) return;

    $('xai-metrics').innerHTML = [
        ['Top-20% IoU', r.iou_top20_vs_teacher],
        ['Spearman ρ', r.spearman_vs_teacher],
        ['Deletion Δp', r.deletion_drop_top20]
    ].map(([key, value]) => `
        <div class="mini">
            <span class="small">${key}</span>
            <strong>${fmt(value)}</strong>
        </div>
    `).join('');

    $('xai-note').textContent =
        model === 'teacher'
            ? 'The teacher is compared to itself for ' +
              'IoU and correlation.'
            : 'IoU and rank correlation measure ' +
              'similarity to the teacher Grad-CAM ' +
              'for the same target class.';
}


    $('sample').addEventListener(
        'change',
        e => {
            sample = Number(e.target.value);
            render();
        }
    );

    $('tabs').addEventListener(
        'click',
        e => {
            if (e.target.dataset.model) {
                model = e.target.dataset.model;
                render();
            }
        }
    );

    render();

    // Detailed table
    const rows = D.samples.flatMap(
        s => names.map(n => s.rows[n]).filter(Boolean)
    );

    $('table').innerHTML = rows.map(r => `
        <tr>
            <td>${r.sample + 1}</td>
            <td>${esc(r.true)}</td>
            <td>${esc(r.teacher_target)}</td>
            <td>${short[r.model]}</td>
            <td>${fmt(r.iou_top20_vs_teacher)}</td>
            <td>${fmt(r.spearman_vs_teacher)}</td>
            <td>${fmt(r.deletion_drop_top20)}</td>
        </tr>
    `).join('');

    // Interpretation from actual measured data
    const avg = (name, key) => {
        const values = rows
            .filter(r => r.model === name)
            .map(r => r[key]);

        return values.length
            ? values.reduce((a, b) => a + b, 0)
              / values.length
            : NaN;
    };

    const gap = (
        metric.distilled.accuracy -
        metric.teacher.accuracy
    ) * 100;

    const reduction = 100 * (
        1 -
        metric.distilled.parameters_million /
        metric.teacher.parameters_million
    );

    const insights = [
        [
            'Predictive performance',
            `${short[best]} reached the highest measured ` +
            `test accuracy (${pct(metric[best].accuracy)}). ` +
            `The distilled student is ` +
            `${Math.abs(gap).toFixed(2)} percentage points ` +
            `${gap >= 0 ? 'above' : 'below'} the teacher.`
        ],
        [
            'Model size',
            `The distilled model contains ` +
            `${fmt(metric.distilled.parameters_million, 2)} ` +
            `million parameters versus ` +
            `${fmt(metric.teacher.parameters_million, 2)} ` +
            `million for the teacher ` +
            `(${fmt(reduction, 1)}% fewer). ` +
            `MobileNetV3 is smaller still.`
        ],
        [
            'Explanation agreement',
            `On ${D.samples.length} displayed images, ` +
            `distilled Grad-CAM average top-20% IoU ` +
            `with teacher is ` +
            `${fmt(avg('distilled',
                       'iou_top20_vs_teacher'))}, ` +
            `with mean Spearman correlation ` +
            `${fmt(avg('distilled',
                       'spearman_vs_teacher'))}. ` +
            `The supervised baseline IoU is ` +
            `${fmt(avg('baseline',
                       'iou_top20_vs_teacher'))}.`
        ],
        [
            'Deletion test',
            `Average target-class probability change ` +
            `after masking the top 20% salient pixels: ` +
            `teacher ` +
            `${fmt(avg('teacher',
                       'deletion_drop_top20'))}, ` +
            `baseline ` +
            `${fmt(avg('baseline',
                       'deletion_drop_top20'))}, ` +
            `distilled ` +
            `${fmt(avg('distilled',
                       'deletion_drop_top20'))}. ` +
            `Negative values can occur; deletion is ` +
            `not proof of causality.`
        ]
    ];

    $('interpret').innerHTML = insights.map(
        ([title, body]) => `
            <div class="report">
                <h3>${esc(title)}</h3>
                <p>${esc(body)}</p>
            </div>
        `
    ).join('');

    // CSV export
    $('csv').addEventListener('click', () => {
        const keys = [
            'sample',
            'true',
            'teacher_target',
            'model',
            'iou_top20_vs_teacher',
            'spearman_vs_teacher',
            'deletion_drop_top20'
        ];

        const quote = v =>
            '"' + String(v ?? '').replaceAll('"', '""') + '"';

        const text = [
            keys.join(','),
            ...rows.map(
                r => keys.map(
                    k => quote(r[k])
                ).join(',')
            )
        ].join('\r\n');

        const url = URL.createObjectURL(
            new Blob(
                [text],
                {type: 'text/csv'}
            )
        );

        const a = document.createElement('a');
        a.href = url;
        a.download = 'explainability.csv';
        a.click();

        setTimeout(
            () => URL.revokeObjectURL(url),
            1000
        );
    });

})();
</script>
</body>
</html>
"""


# ============================================================
# HTML DASHBOARD GENERATOR
# ============================================================


def build_dashboard(out, results, history):
    """Generate a self-contained offline HTML dashboard."""

    with (out / "explainability.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))

    for row in rows:
        row["sample"] = int(row["sample"])

        for key in (
            "iou_top20_vs_teacher",
            "spearman_vs_teacher",
            "deletion_drop_top20"
        ):
            row[key] = float(row[key])

    # Load independently calculated model predictions.
    predictions_path = out / "predictions.json"

    predictions = json.loads(
        predictions_path.read_text(encoding="utf-8")
    )

    samples = []

    for index in sorted({r["sample"] for r in rows}):
        image_path = out / f"gradcam_{index:02d}.png"

        if not image_path.exists():
            continue

        current = [
            r for r in rows
            if r["sample"] == index
        ]

        samples.append({
            "id": index,
            "true": current[0]["true"],
            "target": current[0]["teacher_target"],

            # Predictions from all three models.
            "predictions": predictions[str(index)],

            "image": (
                "data:image/png;base64," +
                base64.b64encode(
                    image_path.read_bytes()
                ).decode("ascii")
            ),

            "rows": {
                r["model"]: r
                for r in current
            }
        })

    data = {
        "metrics": results,
        "history": history,
        "samples": samples,
        "labels": MODEL_LABELS
    }

    payload = json.dumps(
        data,
        ensure_ascii=True,
        allow_nan=False
    )

    # Safely embed JSON into the HTML document.
    payload = (
        payload
        .replace("<", r"\u003c")
        .replace(">", r"\u003e")
        .replace("&", r"\u0026")
    )

    dashboard_path = out / "dashboard.html"

    dashboard_path.write_text(
        HTML_TEMPLATE.replace("__DATA__", payload),
        encoding="utf-8"
    )

    return dashboard_path


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=4
    )

    parser.add_argument(
        "--kd-epochs",
        type=int,
        default=8
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=16
    )

    parser.add_argument(
        "--device",
        default="auto"
    )

    parser.add_argument(
        "--out",
        default="outputs/images"
    )

    parser.add_argument(
        "--skip-train",
        action="store_true"
    )

    parser.add_argument(
        "--examples-per-class",
        type=int,
        default=5,
        help="Number of Grad-CAM examples per class"
    )

    args = parser.parse_args()

    seed_all()
    device = device_from_arg(args.device)

    section(
        "BEAN LEAF DISEASE | "
        "KNOWLEDGE DISTILLATION + GRAD-CAM"
    )

    info(f"Compute device: {device}")

    info(
        f"Training epochs: {args.epochs} | "
        f"KD epochs: {args.kd_epochs} | "
        f"Batch size: {args.batch}"
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    # Dataset
    section("1 / 5  DATASET")

    train_ds = Beans("train", transform)
    val_ds = Beans("validation", transform)
    test_ds = Beans("test", transform)

    info(
        f"Beans: {len(train_ds)} train / "
        f"{len(val_ds)} validation / "
        f"{len(test_ds)} test samples"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        num_workers=0
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch,
        num_workers=0
    )

    # Training history
    history_file = out / "training_history.json"

    history = (
        json.loads(
            history_file.read_text(encoding="utf-8")
        )
        if history_file.exists()
        else {}
    )

    models_dict = {}

    # Models
    section("2 / 5  TRAINING & CHECKPOINTS")

    for kind in (
        "teacher",
        "baseline",
        "distilled"
    ):
        checkpoint = out / f"{kind}.pt"

        info(MODEL_LABELS[kind])

        model = make_model(
            kind,
            pretrained=not (
                args.skip_train and checkpoint.exists()
            )
        )

        if checkpoint.exists():
            model.load_state_dict(
                torch.load(
                    checkpoint,
                    map_location="cpu",
                    weights_only=True
                )
            )

            info(
                f"Loaded existing checkpoint: "
                f"{checkpoint}"
            )

        elif args.skip_train:
            raise FileNotFoundError(checkpoint)

        else:
            if kind == "distilled":
                info("Starting knowledge distillation...")
            else:
                info("Starting supervised training...")

            history[kind] = train(
                model,
                train_loader,
                val_loader,
                device,
                args.kd_epochs
                if kind == "distilled"
                else args.epochs,
                1e-4
                if kind == "distilled"
                else 2e-4,
                teacher=(
                    models_dict["teacher"]
                    if kind == "distilled"
                    else None
                )
            )

            history_file.write_text(
                json.dumps(history, indent=2),
                encoding="utf-8"
            )

            torch.save(
                model.cpu().state_dict(),
                checkpoint
            )

            info(
                f"Saved best model: {checkpoint}"
            )

        models_dict[kind] = model

    # Test evaluation
    section("3 / 5  TEST EVALUATION")

    results = {}

    for kind, model in models_dict.items():
        info(
            f"Evaluating {MODEL_LABELS[kind]}..."
        )

        results[kind] = {
            **evaluate(
                model.to(device),
                test_loader,
                device
            ),
            "parameters_million": params_m(model)
        }

        model.cpu()

    (out / "metrics.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8"
    )

    print_results(results)

    # Grad-CAM
    section("4 / 5  EXPLAINABILITY (GRAD-CAM)")

    explain(
        models_dict,
        test_ds,
        out,
        examples_per_class=args.examples_per_class
    )

    # HTML dashboard
    section("5 / 5  GENERATING HTML DASHBOARD")

    dashboard = build_dashboard(
        out,
        results,
        history
    )

    info(
        f"Interactive dashboard: "
        f"{dashboard.resolve()}"
    )

    section("COMPLETE")

    info(f"Results: {out.resolve()}")

    info(
        "Files: dashboard.html, metrics.json, "
        "training_history.json (if trained), "
        "explainability.csv, gradcam_*.png"
    )

    webbrowser.open(dashboard.resolve().as_uri())


if __name__ == "__main__":
    main()
