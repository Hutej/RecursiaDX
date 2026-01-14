import torch
import os

model_path = r"D:\programs vs\RecursiaDx\ml\models\__pycache__\best_resnet50_model.pth"

print(f"Loading model from: {model_path}")
print("=" * 70)

# Load with CPU mapping
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

print(f"\nCheckpoint type: {type(checkpoint)}")

if isinstance(checkpoint, dict):
    print(f"\nCheckpoint keys: {checkpoint.keys()}")
    
    # Check for state_dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("\nFound 'state_dict' key")
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("\nFound 'model_state_dict' key")
    else:
        state_dict = checkpoint
        print("\nUsing checkpoint as state_dict directly")
else:
    state_dict = checkpoint
    print("\nCheckpoint is the state_dict directly")

# Print first 20 keys to understand structure
print("\n" + "=" * 70)
print("Model layers (first 20):")
print("=" * 70)
for i, key in enumerate(list(state_dict.keys())[:20]):
    print(f"{i+1}. {key}: {state_dict[key].shape}")

# Print last 10 keys
print("\n" + "=" * 70)
print("Model layers (last 10):")
print("=" * 70)
for i, key in enumerate(list(state_dict.keys())[-10:]):
    print(f"{i+1}. {key}: {state_dict[key].shape}")

# Check if it's a ResNet50 with 'backbone.' prefix
has_backbone_prefix = any(k.startswith('backbone.') for k in state_dict.keys())
print(f"\n" + "=" * 70)
print(f"Has 'backbone.' prefix: {has_backbone_prefix}")

# Count total parameters
total_params = sum(p.numel() for p in state_dict.values())
print(f"Total parameters: {total_params:,}")
print("=" * 70)
