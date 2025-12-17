import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
from train import get_loaders

def get_activation_layer(name):
    if name == 'relu': return nn.ReLU()
    if name == 'gelu': return nn.GELU()
    if name == 'silu': return nn.SiLU()
    if name == 'tanh': return nn.Tanh()
    raise ValueError(f"Unknown activation: {name}")

class OptionAShortcut(nn.Module):
    def __init__(self, stride, in_c, out_c):
        super().__init__()
        self.stride = stride
        self.delta = out_c - in_c
    
    def forward(self, x):
        if self.stride == 2:
            x = x[:, :, ::2, ::2]
        if self.delta > 0:
            pad = torch.zeros(x.size(0), self.delta, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        return x

class DynamicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, act_name='relu'):
        super().__init__()
        self.c1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(out_c)
        self.c2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(out_c)
        self.act = get_activation_layer(act_name)

        if stride != 1 or in_c != out_c:
            self.shortcut = OptionAShortcut(stride, in_c, out_c)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        y = self.act(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        y = y + self.shortcut(x)
        return self.act(y)

class DynamicResNet(nn.Module):
    def __init__(self, n=3, num_classes=10, act_name='relu'):
        super().__init__()
        self.act_name = act_name
        self.act = get_activation_layer(act_name)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.stage1 = self._make_layer(16, 16, n, 1)
        self.stage2 = self._make_layer(16, 32, n, 2)
        self.stage3 = self._make_layer(32, 64, n, 2)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, in_c, out_c, n, stride):
        layers = [DynamicBlock(in_c, out_c, stride, self.act_name)]
        for _ in range(n - 1):
            layers.append(DynamicBlock(out_c, out_c, 1, self.act_name))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)

def evaluate_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    print("  Evaluating...", end="", flush=True)
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    print(" Done.")
    return 100. * correct / total

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, testloader = get_loaders(batch_size=128, workers=0) 

    models_config = [
        ("ReLU", "relu", r"checkpoints/ResNet-20_relu.pth"),
        ("GeLU", "gelu", r"checkpoints/ResNet-20_gelu.pth"),
        ("SiLU", "silu", r"checkpoints/ResNet-20_silu.pth"),
        ("Tanh", "tanh", r"checkpoints/ResNet-20_tanh.pth"),
    ]

    results = {}
    colors = ['gray', 'tab:blue', 'tab:green', 'tab:red']
    print(f"\n{'Activation':<10} | {'Test Accuracy':<15}")

    for name, act_type, path in models_config:
        model = DynamicResNet(n=3, act_name=act_type).to(device)
        if os.path.exists(path):
            try:
                state_dict = torch.load(path, map_location=device)
                if list(state_dict.keys())[0].startswith('module.'):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict, strict=False) 
            except Exception as e:
                print(f"Error loading {name}: {e}")
                continue
        else:
            print(f"File not found: {path} (Skipping)")
            continue

        acc = evaluate_accuracy(model, testloader, device)
        results[name] = acc
        print(f"{name:<10} | {acc:.2f}%")

    if len(results) > 0:
        plt.figure(figsize=(10, 6))
        names = list(results.keys())
        accs = list(results.values())
        bars = plt.bar(names, accs, color=colors[:len(names)], alpha=0.8, edgecolor='black')
        if len(accs) > 0:
            min_acc = min(accs)
            plt.ylim(87.5, 93) 
        
        plt.ylabel("Test Accuracy (%)")
        plt.title("ResNet-20 Performance with Different Activations")
        plt.grid(axis='y', alpha=0.3)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 1.0, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')

        save_path = "plots/activation_comparison.png"
        os.makedirs("plots", exist_ok=True)
        plt.savefig(save_path)
        print(f"\n[Success] Plot saved to {save_path}")
    else:
        print("\nNo models were loaded successfully.")