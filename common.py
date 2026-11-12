import random
import numpy as np
import torch


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device_from_arg(choice='auto'):
    if choice != 'auto':
        return torch.device(choice)
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return torch.device('xpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def train_step(student, teacher, inputs, labels, optimizer, alpha=0.5, temperature=4.0):
    import torch.nn.functional as F
    student.train()
    teacher.eval()
    with torch.no_grad():
        tl = teacher(**inputs).logits if isinstance(inputs, dict) else teacher(inputs)
    sl = student(**inputs).logits if isinstance(inputs, dict) else student(inputs)
    hard = F.cross_entropy(sl, labels)
    soft = F.kl_div(F.log_softmax(sl / temperature, dim=-1),
                    F.softmax(tl / temperature, dim=-1), reduction='batchmean') * temperature**2
    loss = alpha * hard + (1-alpha) * soft
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def params_m(model):
    return round(sum(p.numel() for p in model.parameters())/1e6, 3)
