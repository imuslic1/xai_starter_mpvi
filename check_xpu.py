import torch
print('PyTorch:', torch.__version__)
print('XPU available:', hasattr(torch, 'xpu') and torch.xpu.is_available())
if hasattr(torch, 'xpu') and torch.xpu.is_available():
    print('XPU devices:', torch.xpu.device_count())
    print('GPU:', torch.xpu.get_device_name(0))
    a=torch.randn((256,256),device='xpu'); print('Matrix multiplication:',float((a@a).mean().cpu()))
else:
    print('GPU unavailable; upgrade Intel drivers and install official PyTorch XPU wheels.')
