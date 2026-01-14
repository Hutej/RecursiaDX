"""
GPU monitoring and safety utilities for RTX 4070 8GB.
Prevents OOM errors by monitoring VRAM usage and suggesting safe batch sizes.
"""
import torch
from typing import Tuple, Optional
from .logger import get_logger

logger = get_logger(__name__)


class GPUMonitor:
    """Monitor GPU memory usage and provide safety checks."""
    
    def __init__(self, max_vram_gb: float = 7.5):
        """
        Initialize GPU monitor.
        
        Args:
            max_vram_gb: Maximum VRAM to use (GB). Default 7.5 for 8GB GPU with safety margin.
        """
        self.max_vram_gb = max_vram_gb
        self.max_vram_bytes = max_vram_gb * 1024**3
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.device.type == "cuda":
            self.gpu_name = torch.cuda.get_device_name(0)
            self.total_vram = torch.cuda.get_device_properties(0).total_memory
            logger.info(f"GPU detected: {self.gpu_name}")
            logger.info(f"Total VRAM: {self.total_vram / 1024**3:.2f} GB")
            logger.info(f"Max usage limit: {max_vram_gb:.2f} GB")
        else:
            logger.error("GPU REQUIRED — CPU EXECUTION DISALLOWED")
            raise RuntimeError("CUDA NOT AVAILABLE — GPU REQUIRED")
    
    def check_availability(self) -> bool:
        """Check if GPU is available."""
        return torch.cuda.is_available()
    
    def get_memory_info(self) -> Tuple[float, float, float]:
        """
        Get current GPU memory usage.
        
        Returns:
            Tuple of (allocated_gb, reserved_gb, free_gb)
        """
        if not self.check_availability():
            return 0.0, 0.0, 0.0
        
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        free = (self.total_vram - torch.cuda.memory_allocated(0)) / 1024**3
        
        return allocated, reserved, free
    
    def log_memory_status(self):
        """Log current GPU memory status."""
        if not self.check_availability():
            logger.info("GPU: Not available (using CPU)")
            return
        
        allocated, reserved, free = self.get_memory_info()
        logger.info(f"GPU Memory - Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB | Free: {free:.2f} GB")
    
    def check_memory_safe(self, threshold_pct: float = 0.9) -> bool:
        """
        Check if current memory usage is within safe limits.
        
        Args:
            threshold_pct: Warning threshold as percentage of max_vram_gb
        
        Returns:
            True if memory usage is safe, False otherwise
        """
        if not self.check_availability():
            return True  # CPU mode is always "safe"
        
        allocated, _, _ = self.get_memory_info()
        threshold = self.max_vram_gb * threshold_pct
        
        if allocated > threshold:
            logger.warning(f"⚠️ GPU memory usage ({allocated:.2f} GB) exceeds {threshold_pct*100}% threshold ({threshold:.2f} GB)")
            return False
        
        return True
    
    def clear_cache(self):
        """Clear GPU cache to free up memory."""
        if self.check_availability():
            torch.cuda.empty_cache()
            logger.info("[OK] GPU cache cleared")


def get_safe_batch_size(
    sample_size_mb: float,
    max_vram_gb: float = 7.5,
    safety_factor: float = 0.7
) -> int:
    """
    Calculate safe batch size based on sample size and available VRAM.
    
    Args:
        sample_size_mb: Size of one sample in MB
        max_vram_gb: Maximum VRAM to use in GB
        safety_factor: Safety margin (0.7 = use 70% of max VRAM)
    
    Returns:
        Recommended batch size
    
    Example:
        >>> # For 256x256 tiles (RGB) = ~0.19 MB/tile
        >>> batch_size = get_safe_batch_size(sample_size_mb=0.19, max_vram_gb=7.5)
        >>> print(batch_size)  # ~27 (with ResNet50 overhead)
    """
    available_vram_mb = max_vram_gb * 1024 * safety_factor
    batch_size = int(available_vram_mb / (sample_size_mb * 2))  # *2 for gradients/activations
    
    # Ensure minimum batch size of 1
    batch_size = max(1, batch_size)
    
    logger.info(f"Calculated safe batch size: {batch_size} (sample size: {sample_size_mb:.2f} MB)")
    
    return batch_size


def get_device(gpu_id: int = 0) -> torch.device:
    """
    Get torch device (GPU if available, otherwise CPU).
    
    Args:
        gpu_id: GPU device ID
    
    Returns:
        torch.device instance
    """
    if torch.cuda.is_available():
        if gpu_id >= 0 and gpu_id < torch.cuda.device_count():
            device = torch.device(f"cuda:{gpu_id}")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
        else:
            device = torch.device("cuda:0")
            logger.warning(f"Invalid GPU ID {gpu_id}, using cuda:0")
    else:
        logger.error("GPU REQUIRED — CPU EXECUTION DISALLOWED")
        raise RuntimeError("CUDA NOT AVAILABLE — GPU REQUIRED")
    
    return device
