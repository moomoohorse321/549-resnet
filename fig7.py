# figure7_resnet_paper_exact.py
import os, math, torch, torch.nn as nn, torch.nn.functional as F, torchvision as tv, matplotlib.pyplot as plt

# ---------- minimal CIFAR models ----------
def conv3x3(ic, oc, s=1): return nn.Conv2d(ic, oc, 3, stride=s, padding=1, bias=False)

class OptionA(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__(); self.s=stride; self.pad_c=out_c - in_c; assert self.pad_c>=0
    def forward(self,x):
        if self.s==2: x = x[:, :, ::2, ::2]
        if self.pad_c: x = torch.cat([x, torch.zeros(x.size(0), self.pad_c, x.size(2), x.size(3), device=x.device, dtype=x.dtype)], 1)
        return x

class RBlock(nn.Module):
    def __init__(self, ic, oc, s=1):
        super().__init__()
        self.c1,self.b1 = conv3x3(ic,oc,s), nn.BatchNorm2d(oc)
        self.c2,self.b2 = conv3x3(oc,oc,1), nn.BatchNorm2d(oc)
        self.short = nn.Identity() if (s==1 and ic==oc) else OptionA(ic,oc,s)
    def forward(self,x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))         # <-- hook here (post-BN, pre-add/ReLU)
        y = y + self.short(x)
        return F.relu(y)

class PBlock(nn.Module):  # plain block (no residual add)
    def __init__(self, ic, oc, s=1):
        super().__init__()
        self.c1,self.b1 = conv3x3(ic,oc,s), nn.BatchNorm2d(oc)
        self.c2,self.b2 = conv3x3(oc,oc,1), nn.BatchNorm2d(oc)
    def forward(self,x):
        x = F.relu(self.b1(self.c1(x)))
        x = self.b2(self.c2(x))         # <-- hook here (post-BN, pre-ReLU)
        return F.relu(x)

def make_layer(B, ic, oc, n, s):
    return nn.Sequential(B(ic,oc,s), *[B(oc,oc,1) for _ in range(n-1)])

class CIFARNet(nn.Module):
    # depth = 6n+2  → n in {3,9,18} for {20,56,110}
    def __init__(self, n, residual=True, num_classes=10):
        super().__init__()
        B = RBlock if residual else PBlock
        self.conv1 = nn.Conv2d(3,16,3,padding=1,bias=False); self.bn1=nn.BatchNorm2d(16)
        self.s1 = make_layer(B,16,16,n,1)
        self.s2 = make_layer(B,16,32,n,2)
        self.s3 = make_layer(B,32,64,n,2)
        self.fc = nn.Linear(64,num_classes)
    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.s1(x); x = self.s2(x); x = self.s3(x)
        x = F.adaptive_avg_pool2d(x,1).flatten(1)
        return self.fc(x)

# ---------- collect "post-BN pre-nonlinearity" responses ----------
@torch.no_grad()
def layer_response_stds(model, loader, device):
    model.eval().to(device)
    stats, hooks = [], []
    def hook_factory():
        idx=len(stats); stats.append([0.0,0.0,0])
        def hook(_m,_i,y):
            y=y.detach()
            s,ss,n=stats[idx]
            s+=y.sum().item(); ss+=(y*y).sum().item(); n+=y.numel()
            stats[idx]=[s,ss,n]
        return hook
    # register on every BN that immediately follows a 3×3 conv
    prev_is_3x3=False
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.kernel_size==(3,3): prev_is_3x3=True
        elif isinstance(m, nn.BatchNorm2d) and prev_is_3x3:
            hooks.append(m.register_forward_hook(hook_factory())); prev_is_3x3=False
        elif isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
            prev_is_3x3 = isinstance(m, nn.Conv2d) and m.kernel_size==(3,3)
        else:
            prev_is_3x3=False
    for x,_ in loader:
        model(x.to(device))
    for h in hooks: h.remove()
    return [math.sqrt(ss/n - (s/n)**2) for s,ss,n in stats]

# ---------- data: use CIFAR-10 training set ----------
tf = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
])
trainset = tv.datasets.CIFAR10('./data', train=True, download=True, transform=tf)
loader   = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

# ---------- build models & load checkpoints ----------
cfg = {
    'plain-20':   CIFARNet(3,  residual=False),
    'plain-56':   CIFARNet(9,  residual=False),
    'ResNet-20':  CIFARNet(3,  residual=True),
    'ResNet-56':  CIFARNet(9,  residual=True),
    'ResNet-110': CIFARNet(18, residual=True),
}
for name,m in cfg.items():
    ckpt=f'checkpoints/{name}.pth'
    if os.path.isfile(ckpt):
        m.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=False)

# ---------- measure ----------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
results = {name: layer_response_stds(m, loader, device) for name,m in cfg.items()}

# ---------- plot (Figure 7: original order + sorted) ----------
plt.figure(figsize=(8,3.2))
plt.subplot(2,1,1); [plt.plot(v,label=k) for k,v in results.items()]
plt.ylabel('std'); plt.title('Std of layer responses (after BN, before nonlinearity)')
plt.legend(fontsize=8)
plt.subplot(2,1,2); [plt.plot(sorted(v, reverse=True)) for v in results.values()]
plt.xlabel('layer index'); plt.ylabel('std'); plt.tight_layout(); plt.show()
