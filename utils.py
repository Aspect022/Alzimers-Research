"""
KnoAD-Net Utility Functions
===========================
Helper functions for local execution
"""

import torch
import numpy as np
import random
import time
from pathlib import Path
# Add to utils.py (e.g., under other imports / helper functions)
import json
import importlib.util
from pathlib import Path
import os

def load_config(path_or_none=None):
    """
    Load configuration from a JSON file path, or if not found, fall back to config.py
    Returns: a plain dict compatible with notebook expectations:
      - config['random_seed']
      - config['directories'] -> dict with keys like 'data_processed','data_splits','models','checkpoints','results','visualizations'
    """
    # If a path provided and the file exists and is JSON, load it
    if path_or_none:
        p = Path(path_or_none)
        if p.exists() and p.suffix.lower() == '.json':
            with open(p, 'r') as f:
                return json.load(f)

    # Otherwise try to locate config.py in the same project root (or with default name)
    # Look for config.py in current working dir or in parent dirs
    candidate = None
    if path_or_none:
        candidate = Path(path_or_none).with_name('config.py')
        if not candidate.exists():
            # try project root (parent of path)
            candidate = Path(path_or_none).parent / 'config.py'
    if not candidate or not candidate.exists():
        # fallback: look in cwd
        candidate = Path('config.py')
    if not candidate.exists():
        raise FileNotFoundError("No config.json found and could not find config.py. "
                                "Provide a config.json or config.py in project root.")

    spec = importlib.util.spec_from_file_location('project_config', str(candidate))
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)

    # Build a minimal config dict the notebook expects
    # Convert Path objects in cfg.DIRS (if present) to strings
    directories = {}
    if hasattr(cfg, 'DIRS'):
        for k, v in getattr(cfg, 'DIRS').items():
            directories[k] = str(v)

    config_dict = {
        'random_seed': getattr(cfg, 'RANDOM_SEED', getattr(cfg, 'RANDOM_SEED', 42) if hasattr(cfg, 'RANDOM_SEED') else 42),
        'directories': directories
    }

    # Add more fields from config.py if you need them later (e.g., MODEL_CONFIG)
    if hasattr(cfg, 'MODEL_CONFIG'):
        config_dict['model_config'] = getattr(cfg, 'MODEL_CONFIG')

    return config_dict


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """Get available device (CPU for AMD GPU)"""
    # For AMD Radeon, use CPU unless ROCm is installed
    if torch.cuda.is_available():
        # This might work if ROCm is installed
        device = torch.device('cuda')
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(f"✓ Using CPU (PyTorch doesn't support AMD GPU natively)")
    
    return device

def print_section(title, char="="):
    """Print formatted section header"""
    width = 80
    print("\n" + char * width)
    print(title.center(width))
    print(char * width + "\n")

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Timer:
    """Simple timer for profiling"""
    def __init__(self):
        self.start_time = None
        self.elapsed = 0
    
    def start(self):
        self.start_time = time.time()
    
    def stop(self):
        if self.start_time:
            self.elapsed = time.time() - self.start_time
            return self.elapsed
        return 0
    
    def format_time(self, seconds):
        """Format seconds to readable string"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"✓ Checkpoint saved: {filepath}")

def load_checkpoint(model, optimizer, filepath, device):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_number(num):
    """Format large numbers with M/K suffixes"""
    if num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)

def print_model_summary(model):
    """Print model architecture summary"""
    total_params = count_parameters(model)
    print(f"\nModel Summary:")
    print(f"  Total parameters: {format_number(total_params)}")
    
    # Count parameters by module
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  {name}: {format_number(params)}")

def estimate_training_time(dataloader_size, seconds_per_batch, num_epochs):
    """Estimate total training time"""
    total_batches = dataloader_size * num_epochs
    total_seconds = total_batches * seconds_per_batch
    
    timer = Timer()
    time_str = timer.format_time(total_seconds)
    
    print(f"\nEstimated training time:")
    print(f"  Batches per epoch: {dataloader_size}")
    print(f"  Time per batch: {seconds_per_batch:.2f}s")
    print(f"  Total epochs: {num_epochs}")
    print(f"  Estimated total: {time_str}")
    
    return total_seconds

def check_disk_space(path, required_gb=5):
    """Check if sufficient disk space is available"""
    import shutil
    stat = shutil.disk_usage(path)
    free_gb = stat.free / (1024**3)
    
    print(f"\nDisk space check:")
    print(f"  Free space: {free_gb:.2f} GB")
    print(f"  Required: {required_gb} GB")
    
    if free_gb < required_gb:
        print(f"  ⚠ WARNING: Low disk space!")
        return False
    else:
        print(f"  ✓ Sufficient space")
        return True

def print_system_info():
    """Print system information"""
    import platform
    import psutil
    
    print("\nSystem Information:")
    print(f"  OS: {platform.system()} {platform.release()}")
    print(f"  CPU: {platform.processor()}")
    print(f"  CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    print(f"  RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"  Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    
    # GPU info (if available)
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    else:
        print(f"  GPU: Not available (using CPU)")

# CPU optimization tips
def optimize_cpu_performance():
    """Optimize PyTorch for CPU training"""
    import torch
    
    # Set number of threads
    num_threads = torch.get_num_threads()
    print(f"\nCPU Optimization:")
    print(f"  PyTorch threads: {num_threads}")
    
    # Enable TF32 (if available, though mainly for NVIDIA)
    try:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"  ✓ TF32 enabled")
    except:
        pass
    
    # Suggest MKL if available
    try:
        import mkl
        print(f"  ✓ Intel MKL available")
    except:
        print(f"  ℹ Install Intel MKL for faster CPU training:")
        print(f"    conda install mkl mkl-include")

if __name__ == "__main__":
    # Test utilities
    print_system_info()
    optimize_cpu_performance()
    device = get_device()
    print(f"\nDevice: {device}")