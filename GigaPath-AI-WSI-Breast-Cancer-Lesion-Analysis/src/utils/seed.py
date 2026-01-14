"""
Reproducibility utilities for setting random seeds across the pipeline.
"""
import random
import numpy as np
import torch
import os


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
        deterministic: If True, enables deterministic algorithms (slower but reproducible)
    
    Note:
        Deterministic mode may have ~5-10% performance overhead but ensures
        reproducibility for academic work and experiment validation.
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Environment variable (for some libraries)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if deterministic:
        # Enable deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # PyTorch 1.8+ deterministic algorithms
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            # Fallback for older PyTorch versions
            pass
    else:
        # Enable cuDNN autotuner for better performance (non-deterministic)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    print(f"[OK] Random seed set to {seed}")
    print(f"[OK] Deterministic mode: {'ENABLED' if deterministic else 'DISABLED'}")
    if deterministic:
        print("  (Note: ~5-10% slower but fully reproducible)")
