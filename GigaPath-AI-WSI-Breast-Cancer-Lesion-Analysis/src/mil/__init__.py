"""
Multiple Instance Learning (MIL) module.
"""
from .attention_mil import AttentionMIL, GatedAttention
from .mil_trainer import MILTrainer
from .mil_dataset import MILDataset

__all__ = [
    'AttentionMIL',
    'GatedAttention',
    'MILTrainer',
    'MILDataset',
]
