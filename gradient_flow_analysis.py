import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from train import CIFARNet  

def get_gradient_stats(model, device):
    model.train()
    model.to(device)
    
    # backward hooks
    grad_norms = {}
    hooks = []
    
    def get_hook(name):
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                # calculate the L2-Norm
                grad_norm = grad_output[0].norm().item()
                grad_norms[name] = grad_norm
        return hook

    # Register the Backward Hook
    layer_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            h = module.register_full_backward_hook(get_hook(f"{layer_idx}_{name}"))
            hooks.append(h)
            layer_idx += 1

    # Dummy Input (Batch=128)
    dummy_input = torch.randn(128, 3, 32, 32).to(device)
    dummy_target = torch.randint(0, 10, (128,)).to(device)
    criterion = nn.CrossEntropyLoss()

    # Forward & Backward
    model.zero_grad()
    output = model(dummy_input)
    loss = criterion(output, dummy_target)
    loss.backward()

    # Rmove the Hook
    for h in hooks:
        h.remove()

    return grad_norms

def plot_gradient_flow(models_dict, device):
    plt.figure(figsize=(10, 6))
    
    for model_name, model in models_dict.items():
        print(f"Analyzing {model_name} on {device}...")
        grads = get_gradient_stats(model, device)
        
        # (Input -> Output)
        sorted_keys = sorted(grads.keys(), key=lambda x: int(x.split('_')[0]))
        values = [grads[k] for k in sorted_keys]
        
        # Plot
        plt.plot(values, label=model_name, alpha=0.7, linewidth=2, marker='o', markersize=4)
        
    plt.xlabel("Layer Index (Input -> Output)")
    plt.ylabel("Gradient L2 Norm (Log Scale)")
    plt.title("Gradient Flow: Plain vs ResNet (Initialization)")
    plt.yscale('log') 
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    
    save_path = "gradient_flow_analysis.png"
    plt.savefig(save_path, dpi=300)
    print(f"Done! Saved plot to {save_path}")
    plt.show()

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    models = {
        'Plain-56': CIFARNet(n=9, residual=False),
        'ResNet-56': CIFARNet(n=9, residual=True),
        'Plain-110': CIFARNet(n=18, residual=False), 
        'ResNet-110': CIFARNet(n=18, residual=True)
    }
    
    plot_gradient_flow(models, device=device)