"""Yelp 1-5 stars: ready teacher, supervised DistilBERT, KD 4-layer BERT.
Run: python text_reviews.py --train-size 3000 --epochs 2
Explanation: word deletion (occlusion) under shared teacher target, no gradient/XPU dependency.
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import argparse, csv, json, re
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from common import device_from_arg,seed_all,params_m
import matplotlib.pyplot as plt

TEACHER='nlptown/bert-base-multilingual-uncased-sentiment'
BASELINE='distilbert/distilbert-base-uncased'
SMALL='google/bert_uncased_L-4_H-256_A-4'

class Reviews(Dataset):
    def __init__(self,rows): self.rows=rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return self.rows[i]

def collate(rows):return [r['text'] for r in rows],torch.tensor([r['label'] for r in rows])

def logits(model,tok,texts,device,max_length):
    # Input tensors must always follow the actual model device.
    # In particular, evaluate() may have left a model on CPU.
    model_device=next(model.parameters()).device
    tokens=tok(texts,return_tensors='pt',truncation=True,padding=True,max_length=max_length)
    tokens={k:v.to(model_device) for k,v in tokens.items()}
    return model(**tokens).logits

def fit(model,tokenizer,train,valid,device,epochs,batch,lr,max_length,teacher=None,teacher_tok=None):
    model.to(device)
    if teacher is not None:teacher.to(device).eval()
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
    loader=DataLoader(Reviews(train),batch_size=batch,shuffle=True,collate_fn=collate)
    for epoch in range(epochs):
        model.train(); losses=[]
        for texts,labels in loader:
            labels=labels.to(device)
            student=logits(model,tokenizer,texts,device,max_length)
            hard=F.cross_entropy(student,labels)
            if teacher is None: loss=hard
            else:
                with torch.no_grad(): t=logits(teacher,teacher_tok,texts,device,max_length)
                T=4.
                soft=F.kl_div(F.log_softmax(student/T,dim=-1),F.softmax(t/T,dim=-1),reduction='batchmean')*T*T
                loss=0.5*hard+0.5*soft
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            losses.append(float(loss.detach().cpu()))
        stats=evaluate(model,tokenizer,valid,device,batch,max_length)
        print(f'  Epoch {epoch+1:>2}/{epochs:<2} | Loss {np.mean(losses):.4f} | Val accuracy {stats["accuracy"]:.2%} | F1 {stats["macro_f1"]:.4f} | MAE {stats["star_mae"]:.3f}',flush=True)
    model.cpu()

def evaluate(m,tok,rows,device,batch,max_length):
    m.to(device).eval();gold=[];pred=[]
    loader=DataLoader(Reviews(rows),batch_size=batch,collate_fn=collate)
    with torch.no_grad():
        for texts,y in loader:
            yhat=logits(m,tok,texts,device,max_length).argmax(-1).cpu().tolist()
            pred+=yhat;gold+=y.tolist()
    return {'accuracy':round(accuracy_score(gold,pred),4),
            'macro_f1':round(f1_score(gold,pred,average='macro'),4),
            'star_mae':round(mean_absolute_error(gold,pred),4),'n':len(gold)}

def word_occlusion(m,tok,text,device,max_length,target,max_words=40):
    """Signed per-word reduction in P(target) when the word is removed.
    Whitespace tokens are shared between models; perturbations can affect tokenization/context.
    """
    words=text.split()[:max_words]
    if not words:return [],np.array([]),0.
    excerpt=' '.join(words)
    variants=[excerpt]+[' '.join(words[:i]+words[i+1:]) for i in range(len(words))]
    m.to(device).eval()
    probs=[]
    with torch.no_grad():
        for i in range(0,len(variants),8):
            vals=F.softmax(logits(m,tok,variants[i:i+8],device,max_length),dim=-1)[:,target]
            probs.extend(vals.cpu().tolist())
    return words,np.array([probs[0]-v for v in probs[1:]]),probs[0]

def deletion_curve(m,tok,words,weights,device,max_length,target):
    order=np.argsort(-np.abs(weights)); fractions=[0,.1,.2,.3,.5]
    variants=[]
    for f in fractions:
        drop=set(order[:int(round(len(words)*f))])
        variants.append(' '.join(w for i,w in enumerate(words) if i not in drop))
    with torch.no_grad():
        p=F.softmax(logits(m,tok,variants,device,max_length),dim=-1)[:,target].cpu().tolist()
    return fractions,p

def explain(models,tokens,examples,out,device,max_length,n):
    if n < 1: raise ValueError('--explain-n must be at least 1')
    rows=[]; details=[]
    for i,row in enumerate(examples[:n]):
        # Work on the same first <= 40 words across all models for fair alignment.
        text=' '.join(row['text'].split()[:40]);true=int(row['label'])
        with torch.no_grad():
            pred=int(logits(models['teacher'],tokens['teacher'],[text],device,max_length).argmax(-1).item())
        maps={}; probs={}
        for name in models:
            words,weights,p=word_occlusion(models[name],tokens[name],text,device,max_length,pred)
            maps[name]=weights; probs[name]=p
            fractions,curve=deletion_curve(models[name],tokens[name],words,weights,device,max_length,pred)
            corr=spearmanr(maps['teacher'],weights).statistic if name!='teacher' else 1.
            rows.append({'sample':i,'model':name,'rating':true+1,'teacher_prediction':pred+1,
                         'teacher_rating_mismatch':abs(true-pred)>=2,
                         'teacher_target_probability':round(p,4),
                         'attribution_spearman':round(float(corr) if np.isfinite(corr) else 0.,4),
                         'deletion_prob_drop_20pct':round(curve[0]-curve[2],4)})
        # Explanations can be visually compared on exactly aligned whitespace word positions.
        fig,axes=plt.subplots(3,1,figsize=(max(10,len(words)*.22),5.5),sharex=True)
        for ax,(name,values) in zip(axes,maps.items()):
            ax.bar(range(len(words)),values)
            ax.axhline(0,color='gray',linewidth=.7)
            ax.set_ylabel(name); ax.set_title(f'{name}: P(teacher class)={probs[name]:.3f}')
        axes[-1].set_xticks(range(len(words)),words,rotation=65,ha='right',fontsize=8)
        fig.suptitle(f'Example {i}: given {true+1} stars, teacher predicts {pred+1} stars')
        fig.tight_layout();fig.savefig(out/f'word_attributions_{i:02d}.png',dpi=150,bbox_inches='tight');plt.close(fig)
        details.append({'sample':i,'rating':true+1,'teacher_pred':pred+1,'text':text})
    with (out/'explainability.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    (out/'examples.json').write_text(json.dumps(details,indent=2,ensure_ascii=False),encoding='utf-8')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train-size',type=int,default=3000);p.add_argument('--val-size',type=int,default=500)
    p.add_argument('--test-size',type=int,default=500);p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--batch',type=int,default=8);p.add_argument('--max-length',type=int,default=128)
    p.add_argument('--explain-n',type=int,default=6);p.add_argument('--device',default='auto')
    p.add_argument('--out',default='outputs/text');p.add_argument('--skip-train',action='store_true')
    args=p.parse_args();seed_all();device=device_from_arg(args.device)
    print('\n'+'='*68)
    print('  XAI / KNOWLEDGE DISTILLATION — YELP 5-STAR REVIEWS')
    print('='*68)
    print(f'  Device: {device} | Train: {args.train_size} | Epochs: {args.epochs} | Batch: {args.batch}')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    print('\n[1/4] Loading Yelp reviews (5 star classes) ...')
    train_all=load_dataset('Yelp/yelp_review_full',split='train')
    test_all=load_dataset('Yelp/yelp_review_full',split='test')
    pool=train_all.shuffle(seed=42).select(range(min(args.train_size+args.val_size,len(train_all))))
    train=[pool[i] for i in range(min(args.train_size,len(pool)))]
    val=[pool[i] for i in range(len(train),len(pool))]
    test_pool=test_all.shuffle(seed=42).select(range(min(args.test_size,len(test_all))))
    test=[test_pool[i] for i in range(len(test_pool))]
    if not val:raise ValueError('Need validation rows; set --val-size > 0')
    sources={'teacher':TEACHER,'baseline':BASELINE,'distilled':SMALL}
    models={};tokenizers={}
    print('\n[2/4] Preparing teacher and students ...')
    for name,source in sources.items():
        checkpoint=out/name
        tok=AutoTokenizer.from_pretrained(checkpoint if checkpoint.exists() else source)
        if checkpoint.exists(): m=AutoModelForSequenceClassification.from_pretrained(checkpoint)
        elif name=='teacher':m=AutoModelForSequenceClassification.from_pretrained(source)
        else:m=AutoModelForSequenceClassification.from_pretrained(source,num_labels=5)
        models[name]=m;tokenizers[name]=tok
        print(f'  {name.capitalize():<10} | {params_m(m):>7.2f}M parameters | '+('loaded checkpoint' if checkpoint.exists() else 'loaded pretrained weights'),flush=True)
        if name!='teacher' and not checkpoint.exists():
            if args.skip_train:raise FileNotFoundError(checkpoint)
            print(f'\n  Training {name}: '+('teacher-guided distillation' if name=='distilled' else 'supervised learning'),flush=True)
            fit(m,tok,train,val,device,args.epochs,args.batch,
                7e-5 if name=='distilled' else 3e-5,args.max_length,
                teacher=models['teacher'] if name=='distilled' else None,
                teacher_tok=tokenizers['teacher'] if name=='distilled' else None)
            m.save_pretrained(checkpoint);tok.save_pretrained(checkpoint)
        m.cpu()
    print('\n[3/4] Evaluating on held-out test examples ...')
    results={}
    for name in models:
        results[name]={**evaluate(models[name],tokenizers[name],test,device,args.batch,args.max_length),
                       'parameters_million':params_m(models[name])}
        models[name].cpu()
    (out/'metrics.json').write_text(json.dumps(results,indent=2))
    print(f'  {"Model":<12} {"Accuracy":>10} {"Macro F1":>10} {"Star MAE":>10} {"Params":>10}')
    print('  '+'-'*56)
    for name,metric in results.items():
        print(f'  {name:<12} {metric["accuracy"]:>9.2%} {metric["macro_f1"]:>10.4f} {metric["star_mae"]:>10.3f} {metric["parameters_million"]:>8.2f}M')
    # Prefer naturally occurring disagreement cases without assuming that rating mismatch is an error.
    print('\n[4/4] Generating word-occlusion explanations ...',flush=True)
    mismatches=[];others=[]
    models['teacher'].to(device).eval()
    for row in test[:min(100,len(test))]:
        with torch.no_grad():
            pred=int(logits(models['teacher'],tokenizers['teacher'],[row['text']],device,args.max_length).argmax(-1).item())
        (mismatches if abs(pred-int(row['label']))>=2 else others).append(row)
    explain(models,tokenizers,(mismatches+others),out,device,args.max_length,args.explain_n)
    print(f'\n  Selected {min(args.explain_n,len(mismatches+others))} reviews ({len(mismatches)} disagreement cases in first {min(100,len(test))} tested).')
    print('  Outputs:',out.resolve())
    print('  Files: metrics.json, explainability.csv, examples.json, word_attributions_*.png')
    print('='*68)
if __name__=='__main__':main()
