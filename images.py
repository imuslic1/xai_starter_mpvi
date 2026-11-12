"""Beans disease: teacher ResNet50, supervised MobileNetV3, KD ResNet18, Grad-CAM.
Run: python images.py --epochs 4 --kd-epochs 8
"""
import argparse, json, csv, os
from pathlib import Path

# Suppress routine Windows cache/Xet notices; retain meaningful ML warnings.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from datasets import load_dataset
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from common import device_from_arg, seed_all, params_m, train_step

CLASSES=['angular_leaf_spot','bean_rust','healthy']
MEAN=[0.485,0.456,0.406]; STD=[0.229,0.224,0.225]

MODEL_LABELS = {
    "teacher": "Teacher (ResNet-50)",
    "baseline": "Baseline (MobileNetV3-Small)",
    "distilled": "Distilled (ResNet-18)",
}


def section(title):
    print("\n" + "=" * 78, flush=True)
    print(f"  {title}", flush=True)
    print("=" * 78, flush=True)


def info(message):
    print(f"  > {message}", flush=True)


def print_results(results):
    section("FINAL TEST-SET COMPARISON")
    print(f"  {'Model':<32} {'Accuracy':>10} {'CE loss':>10} {'Params (M)':>12}")
    print("  " + "-" * 66)
    for kind, stats in results.items():
        print(f"  {MODEL_LABELS[kind]:<32} {stats['accuracy']*100:>9.2f}% "
              f"{stats['cross_entropy']:>10.4f} {stats['parameters_million']:>12.3f}")
    info("Cross-entropy (CE): lower is better; accuracy: higher is better.")

class Beans(Dataset):
    def __init__(self, split, transform):
        self.data=load_dataset('AI-Lab-Makerere/beans', split=split)
        self.transform=transform
    def __len__(self): return len(self.data)
    def __getitem__(self,i):
        row=self.data[i]
        return self.transform(row['image'].convert('RGB')),int(row['labels'])

def make_model(kind, pretrained=True):
    if kind=='teacher':
        m=models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        m.fc=nn.Linear(m.fc.in_features,3)
    elif kind=='baseline':
        m=models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        m.classifier[-1]=nn.Linear(m.classifier[-1].in_features,3)
    else:
        # Scratch architecture initialized with ImageNet features to make small-data KD practical.
        m=models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        m.fc=nn.Linear(m.fc.in_features,3)
    return m

def evaluate(m,loader,device):
    m.eval(); n=0; correct=0; ls=0.
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            logits=m(x); ls+=float(F.cross_entropy(logits,y,reduction='sum').cpu())
            correct+=int((logits.argmax(1)==y).sum().cpu()); n+=len(y)
    return {'accuracy':round(correct/n,4),'cross_entropy':round(ls/n,4),'n':n}

def train(m,train_loader,val_loader,device,epochs,lr,teacher=None):
    m.to(device)
    if teacher is not None: teacher.to(device).eval()
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=0.01)
    best=-1; best_state=None
    for epoch in range(epochs):
        m.train(); losses=[]
        for x,y in train_loader:
            x,y=x.to(device),y.to(device)
            if teacher is None:
                logits=m(x); loss=F.cross_entropy(logits,y)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                losses.append(float(loss.detach().cpu()))
            else:
                losses.append(train_step(m,teacher,x,y,opt))
        stats=evaluate(m,val_loader,device)
        improved = stats['accuracy'] > best
        print(f"  Epoch {epoch+1:>2}/{epochs:<2} | train loss {np.mean(losses):.4f} | "
              f"val accuracy {stats['accuracy']*100:6.2f}% | val CE {stats['cross_entropy']:.4f}"
              f" | {'BEST' if improved else '----'}", flush=True)
        if improved:
            best=stats['accuracy']; best_state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    m.load_state_dict(best_state)
    info(f"Restored best validation checkpoint (accuracy {best*100:.2f}%).")

def cam(m,x,kind,target):
    """Grad-CAM on CPU for portable backward hooks; x is normalized [1,3,H,W]."""
    m.eval().cpu()
    layer=m.features[-1] if kind=='baseline' else m.layer4[-1]
    saved={}
    def hook(_, __, out):
        saved['activation']=out
        out.register_hook(lambda g: saved.update(gradient=g))
    h=layer.register_forward_hook(hook)
    try:
        out=m(x.cpu()); m.zero_grad(set_to_none=True)
        out[0,target].backward()
        a=saved['activation'].detach(); g=saved['gradient'].detach()
        weights=g.mean((2,3),keepdim=True)
        heat=F.relu((a*weights).sum(1,keepdim=True))
        heat=F.interpolate(heat,size=x.shape[-2:],mode='bilinear',align_corners=False)[0,0].numpy()
        heat=(heat-heat.min())/(heat.max()-heat.min()+1e-8)
        return heat
    finally:
        h.remove()

def top_iou(a,b,fraction=0.2):
    aa=a>=np.quantile(a,1-fraction); bb=b>=np.quantile(b,1-fraction)
    return float(np.logical_and(aa,bb).sum()/max(np.logical_or(aa,bb).sum(),1))

def deletion_drop(m,x,heat,target,fraction=0.2):
    """Zero normalized pixels = ImageNet mean-color replacement; change in target probability."""
    m.eval().cpu(); before=F.softmax(m(x),dim=-1)[0,target].item()
    mask=torch.tensor(heat>=np.quantile(heat,1-fraction))
    altered=x.clone(); altered[0,:,mask]=0
    with torch.no_grad(): after=F.softmax(m(altered),dim=-1)[0,target].item()
    return round(before-after,4)

def explain(models_dict,data,out):
    names=list(models_dict); rows=[]
    count=min(8,len(data))
    info(f"Generating Grad-CAM plots and explanations for {count} test images.")
    for i in range(count):
        x,y=data[i]; x=x.unsqueeze(0)
        with torch.no_grad(): teacher_pred=int(models_dict['teacher'](x).argmax(1))
        heatmaps={k:cam(m,x,k,teacher_pred) for k,m in models_dict.items()}
        img=x[0].permute(1,2,0).numpy()*np.array(STD)+np.array(MEAN)
        img=np.clip(img,0,1)
        fig,ax=plt.subplots(1,4,figsize=(13,3.5)); ax[0].imshow(img);ax[0].set_title(f'true: {CLASSES[y]}')
        for a,(name,hm) in zip(ax[1:],heatmaps.items()):
            a.imshow(img);a.imshow(hm,cmap='jet',alpha=0.45,vmin=0,vmax=1)
            a.set_title(name)
        for a in ax:a.axis('off')
        fig.suptitle(f'Grad-CAM for TEACHER target: {CLASSES[teacher_pred]}');fig.tight_layout()
        fig.savefig(out/f'gradcam_{i:02d}.png',dpi=150);plt.close(fig)
        info(f"Image {i+1:>2}/{count}: true={CLASSES[y]:<18} teacher={CLASSES[teacher_pred]:<18} saved gradcam_{i:02d}.png")
        for name in names:
            hm=heatmaps[name]
            corr=spearmanr(heatmaps['teacher'].ravel(),hm.ravel()).statistic
            rows.append({'sample':i,'true':CLASSES[y],'teacher_target':CLASSES[teacher_pred],
                         'model':name,'iou_top20_vs_teacher':round(top_iou(heatmaps['teacher'],hm),4),
                         'spearman_vs_teacher':round(float(corr) if np.isfinite(corr) else 0.,4),
                         'deletion_drop_top20':deletion_drop(models_dict[name],x,hm,teacher_pred)})
    with (out/'explainability.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--kd-epochs',type=int,default=8);p.add_argument('--batch',type=int,default=16)
    p.add_argument('--device',default='auto');p.add_argument('--out',default='outputs/images')
    p.add_argument('--skip-train',action='store_true'); args=p.parse_args()
    seed_all();device=device_from_arg(args.device)
    section("BEAN LEAF DISEASE | KNOWLEDGE DISTILLATION + GRAD-CAM")
    info(f"Compute device: {device}")
    info(f"Training epochs: {args.epochs} | KD epochs: {args.kd_epochs} | Batch size: {args.batch}")
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    # Deterministic transforms: model-comparable Grad-CAM; no train-only random augmentation.
    t=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    section('1 / 4  DATASET')
    train_ds=Beans('train',t); val_ds=Beans('validation',t); test_ds=Beans('test',t)
    info(f'Beans: {len(train_ds)} train / {len(val_ds)} validation / {len(test_ds)} test samples')
    train_loader=DataLoader(train_ds,batch_size=args.batch,shuffle=True,num_workers=0)
    val_loader=DataLoader(val_ds,batch_size=args.batch,num_workers=0)
    test_loader=DataLoader(test_ds,batch_size=args.batch,num_workers=0)
    ms={}
    section('2 / 4  TRAINING & CHECKPOINTS')
    for kind in ('teacher','baseline','distilled'):
        path=out/f'{kind}.pt'
        info(MODEL_LABELS[kind])
        m=make_model(kind,pretrained=not(args.skip_train and path.exists()))
        if path.exists():
            m.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
            info(f'Loaded existing checkpoint: {path}')
        elif args.skip_train:
            raise FileNotFoundError(path)
        else:
            info('Starting knowledge distillation...' if kind == 'distilled' else 'Starting supervised training...')
            train(m,train_loader,val_loader,device,args.kd_epochs if kind=='distilled' else args.epochs,
                  1e-4 if kind=='distilled' else 2e-4, teacher=ms['teacher'] if kind=='distilled' else None)
            torch.save(m.cpu().state_dict(),path)
            info(f'Saved best model: {path}')
        ms[kind]=m
    section('3 / 4  TEST EVALUATION')
    results={}
    for kind,m in ms.items():
        info(f'Evaluating {MODEL_LABELS[kind]}...')
        results[kind]={**evaluate(m.to(device),test_loader,device),'parameters_million':params_m(m)}
        m.cpu()
    (out/'metrics.json').write_text(json.dumps(results,indent=2))
    print_results(results)

    section('4 / 4  EXPLAINABILITY (GRAD-CAM)')
    explain(ms,test_ds,out)
    section('COMPLETE')
    info(f'Results: {out.resolve()}')
    info('Files: metrics.json, explainability.csv, gradcam_00.png ...')
if __name__=='__main__':main()
