import torch
import sys

print(f"Torch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA NOT AVAILABLE — GPU REQUIRED")

assert torch.cuda.is_available() == True, "CUDA is not available"

device = torch.device("cuda")
assert device.type == "cuda", "Device type is not cuda"

print(f"GPU Device: {torch.cuda.get_device_name(0)}")

print("GPU VERIFIED — RTX GPU ACTIVE")
print("CPU FALLBACK DISABLED")
