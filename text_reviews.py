
"""Yelp 1-5 star review classification and explainable knowledge distillation.

Teacher: nlptown multilingual BERT (already trained for star ratings)
Baseline: supervised DistilBERT
Distilled: 4-layer BERT trained with teacher logits

Examples:
    python text_reviews.py --train-size 3000 --epochs 2 --batch 8
    python text_reviews.py --device xpu --skip-train

Dashboard: outputs/text/dashboard.html (self-contained and offline)
"""
import os
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

import argparse
import base64
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from common import device_from_arg, seed_all, params_m

TEACHER = 'nlptown/bert-base-multilingual-uncased-sentiment'
BASELINE = 'distilbert/distilbert-base-uncased'
SMALL = 'google/bert_uncased_L-4_H-256_A-4'
NAMES = ('teacher', 'baseline', 'distilled')
LABELS = {'teacher': 'Teacher · BERT', 'baseline': 'Baseline · DistilBERT',
          'distilled': 'Distilled · BERT-4'}


def section(title):
    print('\n' + '=' * 77 + '\n  ' + title + '\n' + '=' * 77, flush=True)


def info(message):
    print('  > ' + message, flush=True)


class Reviews(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate(rows):
    return [r['text'] for r in rows], torch.tensor([r['label'] for r in rows])


def logits(model, tokenizer, texts, max_length):
    """Move tokens to the model's actual device (avoids CPU/XPU mismatches)."""
    device = next(model.parameters()).device
    tokens = tokenizer(texts, return_tensors='pt', truncation=True,
                       padding=True, max_length=max_length)
    return model(**{k: v.to(device) for k, v in tokens.items()}).logits


def evaluate(model, tokenizer, rows, device, batch, max_length):
    model.to(device).eval()
    gold, pred = [], []
    loader = DataLoader(Reviews(rows), batch_size=batch, collate_fn=collate)
    with torch.no_grad():
        for texts, labels in loader:
            guesses = logits(model, tokenizer, texts, max_length).argmax(-1)
            pred.extend(guesses.cpu().tolist())
            gold.extend(labels.tolist())
    return {'accuracy': round(accuracy_score(gold, pred), 4),
            'macro_f1': round(f1_score(gold, pred, average='macro', zero_division=0), 4),
            'star_mae': round(mean_absolute_error(gold, pred), 4), 'n': len(gold)}


def fit(model, tokenizer, train_rows, valid_rows, device, epochs, batch, lr,
        max_length, teacher=None, teacher_tokenizer=None):
    model.to(device)
    if teacher is not None:
        teacher.to(device).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
    loader = DataLoader(Reviews(train_rows), batch_size=batch,
                        shuffle=True, collate_fn=collate)
    history = []
    best_f1, best_weights = -1., None
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for texts, labels in loader:
            labels = labels.to(device)
            student_logits = logits(model, tokenizer, texts, max_length)
            hard_loss = F.cross_entropy(student_logits, labels)
            if teacher is None:
                loss = hard_loss
            else:
                with torch.no_grad():
                    teacher_logits = logits(teacher, teacher_tokenizer, texts, max_length)
                temperature = 4.
                soft_loss = F.kl_div(
                    F.log_softmax(student_logits / temperature, dim=-1),
                    F.softmax(teacher_logits / temperature, dim=-1),
                    reduction='batchmean') * temperature ** 2
                loss = .5 * hard_loss + .5 * soft_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        stats = evaluate(model, tokenizer, valid_rows, device, batch, max_length)
        entry = {'epoch': epoch, 'train_loss': round(float(np.mean(losses)), 5),
                 'val_accuracy': stats['accuracy'], 'val_macro_f1': stats['macro_f1'],
                 'val_star_mae': stats['star_mae']}
        history.append(entry)
        improved = stats['macro_f1'] > best_f1
        if improved:
            best_f1 = stats['macro_f1']
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"  Epoch {epoch:>2}/{epochs:<2} | loss {entry['train_loss']:.4f} | "
              f"val acc {stats['accuracy']:6.2%} | F1 {stats['macro_f1']:.4f} | "
              f"MAE {stats['star_mae']:.3f} | {'BEST' if improved else '----'}", flush=True)
    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.cpu()
    info(f'Restored best validation checkpoint (macro F1 {best_f1:.4f}).')
    return history


def occlusion(model, tokenizer, text, max_length, target, max_words=40):
    """Signed importance: P(target | original) - P(target | word removed)."""
    words = text.split()[:max_words]
    if not words:
        return [], np.array([]), 0.
    variants = [' '.join(words)] + [
        ' '.join(words[:i] + words[i + 1:]) for i in range(len(words))]
    probabilities = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(variants), 8):
            values = F.softmax(logits(model, tokenizer, variants[start:start + 8],
                                      max_length), dim=-1)[:, target]
            probabilities.extend(values.cpu().tolist())
    return words, np.array([probabilities[0] - p for p in probabilities[1:]]), probabilities[0]


def deletion_curve(model, tokenizer, words, weights, max_length, target):
    fractions = [0., .1, .2, .3, .5]
    order = np.argsort(-np.abs(weights))
    variants = []
    for fraction in fractions:
        removed = set(order[:int(round(len(words) * fraction))])
        variants.append(' '.join(w for i, w in enumerate(words) if i not in removed))
    with torch.no_grad():
        p = F.softmax(logits(model, tokenizer, variants, max_length),
                      dim=-1)[:, target].cpu().tolist()
    return fractions, p


def explain(models, tokenizers, examples, out, device, max_length, n, max_words):
    if n < 1:
        raise ValueError('--explain-n must be at least 1')
    rows, details = [], []
    selected = examples[:n]
    if not selected:
        raise ValueError('No reviews to explain')
    info(f'Creating word-occlusion explanations for {len(selected)} reviews.')
    for i, row in enumerate(selected):
        text = ' '.join(row['text'].split()[:max_words])
        true_rating = int(row['label']) + 1
        with torch.no_grad():
            target = int(logits(models['teacher'], tokenizers['teacher'],
                                [text], max_length).argmax(-1).item())
        weights_by_model, plot_prob = {}, {}
        detail = {'sample': i, 'rating': true_rating, 'teacher_pred': target + 1,
                  'text': text, 'models': {}}
        for name in NAMES:
            model = models[name]
            model.to(device).eval()
            words, weights, prob = occlusion(model, tokenizers[name], text,
                                             max_length, target, max_words=max_words)
            fractions, curve = deletion_curve(model, tokenizers[name], words,
                                               weights, max_length, target)
            weights_by_model[name], plot_prob[name] = weights, prob
            if name == 'teacher':
                correlation = 1.
            else:
                correlation = spearmanr(weights_by_model['teacher'], weights).statistic
                correlation = float(correlation) if np.isfinite(correlation) else 0.
            row_result = {'sample': i, 'model': name, 'rating': true_rating,
                          'teacher_prediction': target + 1,
                          'teacher_rating_mismatch': abs(true_rating - (target + 1)) >= 2,
                          'teacher_target_probability': round(prob, 4),
                          'attribution_spearman': round(correlation, 4),
                          'deletion_prob_drop_20pct': round(curve[0] - curve[2], 4)}
            rows.append(row_result)
            detail['models'][name] = {
                'words': words, 'weights': [round(float(w), 6) for w in weights],
                'target_probability': round(prob, 6),
                'attribution_spearman': row_result['attribution_spearman'],
                'deletion_fractions': fractions,
                'deletion_probabilities': [round(float(p), 6) for p in curve],
                'deletion_prob_drop_20pct': row_result['deletion_prob_drop_20pct']}
            model.cpu()  # Save GPU memory, and logits() will follow device next time.
        fig, axes = plt.subplots(3, 1, figsize=(max(10, len(words) * .24), 6), sharex=True)
        for ax, name in zip(axes, NAMES):
            ax.bar(range(len(words)), weights_by_model[name])
            ax.axhline(0, color='gray', linewidth=.7)
            ax.set_ylabel(name)
            ax.set_title(f'{name}: P(teacher target)={plot_prob[name]:.3f}')
        axes[-1].set_xticks(range(len(words)), words, rotation=65, ha='right', fontsize=8)
        fig.suptitle(f'Review {i + 1}: given {true_rating} stars, teacher predicts {target + 1}')
        fig.tight_layout()
        fig.savefig(out / f'word_attributions_{i:02d}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        details.append(detail)
        info(f'Review {i + 1:>2}/{len(selected)} | actual {true_rating} stars | '
             f'teacher {target + 1} stars | saved attribution plot')
    with (out / 'explainability.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (out / 'examples.json').write_text(json.dumps(details, indent=2, ensure_ascii=False),
                                       encoding='utf-8')
    return details


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SentimentScope · Explainable AI</title>
<style>
:root{color-scheme:dark;--bg:#09111e;--panel:#15243b;--edge:#2b405f;--ink:#f0f6ff;--sub:#a5b7cf;--t:#43e0cb;--b:#82a9ff;--d:#c8a0fa}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at 85% 0%,#203b5d,transparent 55%),var(--bg);color:var(--ink);font:15px/1.6 system-ui,Segoe UI,sans-serif}
main{max-width:1160px;margin:auto;padding:28px 22px 85px}header{display:flex;justify-content:space-between;gap:10px;padding-bottom:18px;border-bottom:1px solid var(--edge);font-size:13px;letter-spacing:.07em}h1{font-size:clamp(36px,6vw,63px);line-height:1.08;letter-spacing:-.055em;margin:55px 0 14px}h1 em{font-style:normal;color:var(--t)}h2{margin:45px 0 16px;font-size:23px}h3{margin:4px 0 12px}p,.small{color:var(--sub)}.small{font-size:12px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}.two{grid-template-columns:repeat(2,minmax(0,1fr))}.card{min-width:0;background:linear-gradient(145deg,#192c46,var(--panel));border:1px solid var(--edge);border-radius:19px;padding:23px;box-shadow:0 12px 35px #0002}.big{font-size:36px;font-weight:800;letter-spacing:-.05em}.bar{height:12px;background:#2b405d;border-radius:12px;overflow:hidden;margin:8px 0 21px}.fill{height:100%;width:0;border-radius:12px;transition:width 1.2s cubic-bezier(.2,.9,.2,1)}.line{display:flex;justify-content:space-between;gap:10px;font-size:13px}.line strong{font-variant-numeric:tabular-nums}.legend{display:flex;flex-wrap:wrap;gap:15px;color:var(--sub);font-size:13px;margin-bottom:12px}.dot{width:9px;height:9px;display:inline-block;border-radius:50%;margin-right:6px}button,select{border:1px solid #526b88;border-radius:10px;background:#203550;color:white;padding:10px 13px;font:inherit;cursor:pointer}button.active,button:hover{border-color:var(--t);background:#20525c}.tabs{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}.review{font-size:19px;line-height:2.2;padding:15px;background:#0c192c;border-radius:13px}.word{display:inline-block;border-radius:4px;padding:0 3px;margin:1px;transition:transform .15s}.word:hover{transform:translateY(-3px)}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:14px}.stat{background:#0d1a2d;padding:13px;border-radius:12px}.stat b{font-size:23px;display:block}.report{border-left:3px solid var(--t);padding:2px 18px;margin:16px 0}.report p{margin:7px 0 14px}.tablebox{overflow-x:auto}table{width:100%;border-collapse:collapse;white-space:nowrap;font-size:13px}td,th{text-align:left;padding:11px;border-bottom:1px solid var(--edge)}th{color:var(--sub)}svg{width:100%;height:auto;display:block}.fade{animation:up .8s both}@keyframes up{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}.tag{padding:4px 9px;border-radius:12px;border:1px solid var(--edge);font-size:12px;display:inline-block;margin-left:5px}.chart{min-height:80px}.plot{width:100%;border-radius:12px;margin-top:15px}a{color:var(--t)}@media(max-width:800px){.grid,.two{grid-template-columns:1fr}}@media(max-width:480px){.stats{grid-template-columns:1fr}main{padding:18px}.review{font-size:16px}}@media(prefers-reduced-motion:reduce){*,*:before,*:after{animation:none!important;transition:none!important}}
</style></head><body><main>
<header><b>◈ SENTIMENTSCOPE / XAI LAB</b><span>Offline research dashboard</span></header>
<section class="fade"><h1>Do smaller models<br><em>notice the same words?</em></h1><p>Five-star Yelp reviews · Teacher vs. supervised and distilled students · Word occlusion</p></section>
<div id="summary" class="grid"></div>
<h2>01 / Model performance</h2><div class="grid two"><div class="card"><h3>Test accuracy</h3><div id="accuracy"></div></div><div class="card"><h3>Number of parameters</h3><div id="parameters"></div></div></div>
<h2>02 / Validation history</h2><div class="card"><div id="history"></div><p class="small" id="historyhint"></p></div>
<h2>03 / Review explanation explorer</h2><div class="card"><label for="sample">Select review: </label><select id="sample"></select><p id="rating"></p><div class="tabs" id="tabs"></div><div id="words" class="review"></div><p class="small">Teal = word supports the teacher's predicted star class for the selected model. Pink = word suppresses it. Opacity indicates relative strength; removal is a perturbation, not proof of causality.</p><div class="stats" id="stats"></div><h3 style="margin-top:24px">Deletion curve</h3><div id="deletion"></div><p class="small">Words are removed in descending absolute importance. The vertical axis is the probability of the teacher's target class for the selected model.</p><details><summary>Original saved attribution plot</summary><img id="plot" class="plot" alt="Attribution chart for the three models"></details></div>
<h2>04 / Interpretation from measured results</h2><div class="card" id="interpret"></div>
<h2>05 / Explainability measurements</h2><div class="card tablebox"><button id="export">Export CSV</button><table><thead><tr><th>Review</th><th>Model</th><th>Actual rating</th><th>Teacher prediction</th><th>Mismatch ≥2</th><th>Target probability</th><th>Spearman ρ</th><th>Deletion Δp (20%)</th></tr></thead><tbody id="table"></tbody></table></div>
<p class="small">Methodological notes: Star ratings are subjective; a mismatch is not automatically a model error. Explanations use only the first 40 whitespace-separated words of each review. The teacher target class is shared across models. A small selection of reviews cannot support statistical claims about all Yelp reviews. Metrics and attribution curves come from the actual run.</p>
</main><script type="application/json" id="payload">__DATA__</script><script>
(()=>{'use strict';
const D=JSON.parse(document.getElementById('payload').textContent),$=id=>document.getElementById(id);
const names=['teacher','baseline','distilled'],colors={teacher:'#43e0cb',baseline:'#82a9ff',distilled:'#c8a0fa'},short={teacher:'Teacher',baseline:'Baseline',distilled:'Distilled'};
const fmt=(v,d=3)=>Number(v).toFixed(d),pct=v=>fmt(v*100,2)+'%',esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const m=D.metrics;
$('summary').innerHTML=names.map(n=>`<div class="card"><span class="small">${esc(D.labels[n])}</span><div class="big" style="color:${colors[n]}">${pct(m[n].accuracy)}</div><div class="small">Macro F1: ${fmt(m[n].macro_f1,4)} · Rating MAE: ${fmt(m[n].star_mae,3)}<br>${fmt(m[n].parameters_million,2)}M parameters</div></div>`).join('');
function bars(id,key,scale,suffix){const max=Math.max(...names.map(n=>m[n][key]));$(id).innerHTML=names.map(n=>`<div class="line"><span>${short[n]}</span><strong>${fmt(m[n][key]*scale,2)}${suffix}</strong></div><div class="bar"><div class="fill" data-w="${max?m[n][key]/max*100:0}" style="background:${colors[n]}"></div></div>`).join('')}
bars('accuracy','accuracy',100,'%');bars('parameters','parameters_million',1,'M');requestAnimationFrame(()=>setTimeout(()=>document.querySelectorAll('.fill').forEach(e=>e.style.width=e.dataset.w+'%'),80));
// Validation F1 curves, if saved after training
const hist=D.history||{},available=names.filter(n=>Array.isArray(hist[n])&&hist[n].length);
function linePlot(series,yLabel){let all=series.flatMap(s=>s.data.map(p=>p[1]));if(!all.length)return '<p>No data available.</p>';let lo=Math.min(...all),hi=Math.max(...all);if(Math.abs(hi-lo)<.001){lo-=.05;hi+=.05}else{lo=Math.max(0,lo-(hi-lo)*.15);hi=Math.min(1,hi+(hi-lo)*.15)}let n=Math.max(2,...series.flatMap(s=>s.data.map(p=>p[0]))),X=x=>54+(x-1)/(n-1)*810,Y=y=>208-(y-lo)/(hi-lo)*170;let svg='<svg viewBox="0 0 900 245" role="img" aria-label="'+esc(yLabel)+'">';for(let i=0;i<=4;i++){const v=lo+(hi-lo)*i/4;svg+=`<line x1="54" x2="865" y1="${Y(v)}" y2="${Y(v)}" stroke="#344b66" stroke-dasharray="4 5"/><text x="48" y="${Y(v)+4}" text-anchor="end" font-size="12" fill="#a5b7cf">${fmt(v,2)}</text>`}for(const s of series){let pts=s.data.map(p=>`${X(p[0])},${Y(p[1])}`).join(' ');svg+=`<polyline class="trace" points="${pts}" fill="none" stroke="${colors[s.name]}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>`}return svg+'</svg>'}
if(available.length){$('history').innerHTML='<div class="legend">'+available.map(n=>`<span><i class="dot" style="background:${colors[n]}"></i>${short[n]}</span>`).join('')+'</div>'+linePlot(available.map(n=>({name:n,data:hist[n].map(x=>[x.epoch,x.val_macro_f1])})),'Validation macro F1 by epoch');$('historyhint').textContent='Curves represent macro F1 after each epoch. Pretrained teacher is not fine-tuned here.';document.querySelectorAll('.trace').forEach(e=>{let l=e.getTotalLength();e.style.strokeDasharray=l;e.style.strokeDashoffset=l;e.style.transition='stroke-dashoffset 1.5s ease';requestAnimationFrame(()=>e.style.strokeDashoffset=0)})}else{$('history').innerHTML='<p>Training history was not saved with these existing checkpoints.</p>';$('historyhint').textContent='Train with this updated script to record future epoch metrics.'}
// Interactive review explorer
let activeSample=D.examples[0]?.sample??0,activeModel='distilled';
$('sample').innerHTML=D.examples.map(e=>`<option value="${e.sample}">Review ${e.sample+1}: ${e.rating}★ rated / ${e.teacher_pred}★ teacher</option>`).join('');
$('tabs').innerHTML=names.map(n=>`<button type="button" data-name="${n}">${short[n]}</button>`).join('');
function curveChart(e){const series=names.map(n=>({name:n,data:e.models[n].deletion_probabilities.map((p,i)=>[i,p])})),fractions=e.models.teacher.deletion_fractions;let svg='<svg viewBox="0 0 900 260" role="img" aria-label="Deletion probability by removed word fraction">',X=i=>52+i/(fractions.length-1)*810,Y=v=>220-v*184;for(let p=0;p<=1.001;p+=.25){svg+=`<line x1="52" x2="862" y1="${Y(p)}" y2="${Y(p)}" stroke="#344b66" stroke-dasharray="4 5"/><text x="46" y="${Y(p)+4}" text-anchor="end" font-size="12" fill="#a5b7cf">${fmt(p,2)}</text>`}for(const s of series){svg+=`<polyline points="${s.data.map(p=>`${X(p[0])},${Y(p[1])}`).join(' ')}" fill="none" stroke="${colors[s.name]}" stroke-width="${s.name===activeModel?4:2}" stroke-opacity="${s.name===activeModel?1:.55}"/>`}fractions.forEach((v,i)=>svg+=`<text x="${X(i)}" y="244" text-anchor="middle" fill="#a5b7cf" font-size="12">${Math.round(v*100)}%</text>`);return '<div class="legend">'+names.map(n=>`<span><i class="dot" style="background:${colors[n]}"></i>${short[n]}</span>`).join('')+'</div>'+svg+'</svg>'}
function render(){const e=D.examples.find(r=>r.sample===activeSample);if(!e)return;const d=e.models[activeModel],weights=d.weights,max=Math.max(.00001,...weights.map(Math.abs));$('rating').innerHTML=`Actual: <b>${e.rating} stars</b> · Teacher predicts: <b>${e.teacher_pred} stars</b> ${Math.abs(e.rating-e.teacher_pred)>=2?'<span class="tag">Rating disagreement ≥2</span>':''}`;$('tabs').querySelectorAll('button').forEach(b=>b.classList.toggle('active',b.dataset.name===activeModel));$('words').innerHTML=d.words.map((word,i)=>{let w=weights[i],a=Math.min(.72,.12+Math.abs(w)/max*.6),color=w>=0?`rgba(67,224,203,${a})`:`rgba(255,116,151,${a})`;return `<span class="word" style="background:${color}" title="Occlusion importance: ${fmt(w,5)}">${esc(word)}</span>`}).join(' ');$('stats').innerHTML=[['Teacher-target P',fmt(d.target_probability)],['Spearman vs teacher',fmt(d.attribution_spearman)],['Deletion drop (20%)',fmt(d.deletion_prob_drop_20pct)]].map(([key,val])=>`<div class="stat"><div class="small">${key}</div><b>${val}</b></div>`).join('');$('deletion').innerHTML=curveChart(e);$('plot').src=e.plot}
$('sample').addEventListener('change',ev=>{activeSample=Number(ev.target.value);render()});$('tabs').addEventListener('click',ev=>{const n=ev.target.dataset.name;if(n){activeModel=n;render()}});render();
const rows=D.rows;$('table').innerHTML=rows.map(r=>`<tr><td>${r.sample+1}</td><td>${short[r.model]}</td><td>${r.rating}★</td><td>${r.teacher_prediction}★</td><td>${r.teacher_rating_mismatch?'Yes':'No'}</td><td>${fmt(r.teacher_target_probability)}</td><td>${fmt(r.attribution_spearman)}</td><td>${fmt(r.deletion_prob_drop_20pct)}</td></tr>`).join('');
function avg(n,key){const v=rows.filter(r=>r.model===n).map(r=>r[key]);return v.length?v.reduce((a,b)=>a+b,0)/v.length:NaN}
const best=names.reduce((a,b)=>m[b].accuracy>m[a].accuracy?b:a),gap=(m.distilled.accuracy-m.teacher.accuracy)*100,reduction=(1-m.distilled.parameters_million/m.teacher.parameters_million)*100,disagreements=D.examples.filter(e=>Math.abs(e.rating-e.teacher_pred)>=2).length;
const insights=[['Predictive performance',`${short[best]} has the highest measured test accuracy (${pct(m[best].accuracy)}). The distilled model is ${fmt(Math.abs(gap),2)} percentage points ${gap>=0?'above':'below'} the teacher. It is not valid to infer statistical significance from this one run.`],['Compression tradeoff',`Distilled BERT-4 has ${fmt(m.distilled.parameters_million,2)}M parameters versus ${fmt(m.teacher.parameters_million,2)}M for the teacher (${fmt(reduction,1)}% fewer). Parameter count is not the same as measured inference speed.`],['Word importance agreement',`Across ${D.examples.length} selected review excerpts, distilled-vs-teacher mean Spearman ρ is ${fmt(avg('distilled','attribution_spearman'))}; baseline-vs-teacher is ${fmt(avg('baseline','attribution_spearman'))}. These are exploratory descriptive averages, not population estimates.`],['Rating versus language',`${disagreements} of ${D.examples.length} displayed reviews differ from the teacher prediction by at least two stars. These reviews were deliberately prioritized, so this is NOT a random-sample mismatch rate and a mismatch is not automatically an error.`],['Faithfulness probe',`Mean reduction in teacher-target probability after removing the top 20% of influential words: teacher ${fmt(avg('teacher','deletion_prob_drop_20pct'))}; baseline ${fmt(avg('baseline','deletion_prob_drop_20pct'))}; distilled ${fmt(avg('distilled','deletion_prob_drop_20pct'))}. Removal can change context and tokenization.`]];
$('interpret').innerHTML=insights.map(([title,body])=>`<div class="report"><h3>${esc(title)}</h3><p>${esc(body)}</p></div>`).join('');
$('export').addEventListener('click',()=>{const keys=['sample','model','rating','teacher_prediction','teacher_rating_mismatch','teacher_target_probability','attribution_spearman','deletion_prob_drop_20pct'];const q=v=>'"'+String(v??'').replaceAll('"','""')+'"';const csv=[keys.join(','),...rows.map(r=>keys.map(k=>q(r[k])).join(','))].join('\r\n');const url=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));const link=document.createElement('a');link.href=url;link.download='explainability.csv';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000)});
})();
</script></body></html>'''


def build_dashboard(out, results, history, examples):
    with (out / 'explainability.csv').open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row['sample'] = int(row['sample'])
        row['rating'] = int(row['rating'])
        row['teacher_prediction'] = int(row['teacher_prediction'])
        row['teacher_rating_mismatch'] = row['teacher_rating_mismatch'].lower() == 'true'
        for key in ('teacher_target_probability', 'attribution_spearman',
                    'deletion_prob_drop_20pct'):
            row[key] = float(row[key])
    for example in examples:
        image = out / f"word_attributions_{example['sample']:02d}.png"
        example['plot'] = ('data:image/png;base64,' +
                           base64.b64encode(image.read_bytes()).decode('ascii'))
    payload = json.dumps({'metrics': results, 'history': history,
                          'examples': examples, 'rows': rows, 'labels': LABELS},
                         ensure_ascii=True, allow_nan=False)
    payload = payload.replace('<', r'\u003c').replace('>', r'\u003e').replace('&', r'\u0026')
    dashboard = out / 'dashboard.html'
    dashboard.write_text(HTML.replace('__DATA__', payload), encoding='utf-8')
    return dashboard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-size', type=int, default=3000)
    parser.add_argument('--val-size', type=int, default=500)
    parser.add_argument('--test-size', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--explain-n', type=int, default=6)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--out', default='outputs/text')
    parser.add_argument('--skip-train', action='store_true')
    parser.add_argument(
    '--explain-words',
    type=int,
    default=100,
    help='Maximum number of words per review used for XAI'
    )
    args = parser.parse_args()
    if min(args.train_size, args.val_size, args.test_size, args.batch, args.explain_n) < 1:
        parser.error('Sizes, batch, and --explain-n must be positive')
    seed_all()
    device = device_from_arg(args.device)
    section('YELP 5-STAR REVIEWS | KNOWLEDGE DISTILLATION + XAI')
    info(f'Device: {device} | Train: {args.train_size} | Epochs: {args.epochs} | Batch: {args.batch}')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    section('1 / 5  DATASET')
    train_all = load_dataset('Yelp/yelp_review_full', split='train')
    test_all = load_dataset('Yelp/yelp_review_full', split='test')
    pool = train_all.shuffle(seed=42).select(range(min(args.train_size + args.val_size, len(train_all))))
    train_rows = [pool[i] for i in range(min(args.train_size, len(pool)))]
    val_rows = [pool[i] for i in range(len(train_rows), len(pool))]
    test_pool = test_all.shuffle(seed=42).select(range(min(args.test_size, len(test_all))))
    test_rows = [test_pool[i] for i in range(len(test_pool))]
    if not val_rows:
        raise ValueError('No validation data: increase --val-size')
    info(f'{len(train_rows)} train / {len(val_rows)} validation / {len(test_rows)} test reviews')

    sources = {'teacher': TEACHER, 'baseline': BASELINE, 'distilled': SMALL}
    models, tokenizers = {}, {}
    history_path = out / 'training_history.json'
    history = json.loads(history_path.read_text(encoding='utf-8')) if history_path.exists() else {}

    section('2 / 5  MODELS & TRAINING')
    for name in NAMES:
        checkpoint = out / name
        tokenizer = AutoTokenizer.from_pretrained(checkpoint if checkpoint.exists() else sources[name])
        if checkpoint.exists():
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        elif name == 'teacher':
            model = AutoModelForSequenceClassification.from_pretrained(sources[name])
        else:
            model = AutoModelForSequenceClassification.from_pretrained(sources[name], num_labels=5)
        models[name], tokenizers[name] = model, tokenizer
        info(f'{LABELS[name]} | {params_m(model):.2f}M parameters | '
             + ('checkpoint loaded' if checkpoint.exists() else 'pretrained weights loaded'))
        if name != 'teacher' and not checkpoint.exists():
            if args.skip_train:
                raise FileNotFoundError(checkpoint)
            info('Starting ' + ('teacher-guided knowledge distillation' if name == 'distilled'
                                else 'supervised fine-tuning'))
            history[name] = fit(model, tokenizer, train_rows, val_rows, device, args.epochs,
                                args.batch, 7e-5 if name == 'distilled' else 3e-5,
                                args.max_length, teacher=models['teacher'] if name == 'distilled' else None,
                                teacher_tokenizer=tokenizers['teacher'] if name == 'distilled' else None)
            history_path.write_text(json.dumps(history, indent=2), encoding='utf-8')
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
            info('Saved checkpoint: ' + str(checkpoint))
        model.cpu()

    section('3 / 5  HELD-OUT TEST EVALUATION')
    results = {}
    for name in NAMES:
        info(f'Evaluating {LABELS[name]} ...')
        results[name] = {**evaluate(models[name], tokenizers[name], test_rows, device,
                                    args.batch, args.max_length),
                         'parameters_million': params_m(models[name])}
        models[name].cpu()
    (out / 'metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f"  {'Model':<27} {'Accuracy':>9} {'Macro F1':>10} {'Star MAE':>10} {'Params':>10}")
    print('  ' + '-' * 70)
    for name in NAMES:
        r = results[name]
        print(f"  {LABELS[name]:<27} {r['accuracy']:>8.2%} {r['macro_f1']:>10.4f} "
              f"{r['star_mae']:>10.3f} {r['parameters_million']:>9.2f}M")

    section('4 / 5  WORD OCCLUSION & MISMATCH EXAMPLES')
    mismatches, others = [], []
    models['teacher'].to(device).eval()
    for row in test_rows[:min(100, len(test_rows))]:
        with torch.no_grad():
            predicted = int(logits(models['teacher'], tokenizers['teacher'],
                                   [row['text']], args.max_length).argmax(-1).item())
        (mismatches if abs(predicted - int(row['label'])) >= 2 else others).append(row)
    details = explain(models, tokenizers, mismatches + others, out, device,
                      args.max_length, args.explain_n, args.explain_words)
    info(f'Found {len(mismatches)} teacher/rating disagreements in the first '
         f'{min(100, len(test_rows))} test reviews (not a random sample of dashboard examples).')

    section('5 / 5  INTERACTIVE HTML DASHBOARD')
    dashboard = build_dashboard(out, results, history, details)
    info('Dashboard: ' + str(dashboard.resolve()))
    section('COMPLETE')
    info('Files: dashboard.html, metrics.json, examples.json, explainability.csv, '
         'word_attributions_*.png')


if __name__ == '__main__':
    main()
