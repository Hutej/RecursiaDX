import sys
import pandas as pd
from pathlib import Path

# Add root to path
root_dir = Path(__file__).resolve().parent
sys.path.append(str(root_dir))

try:
    from src.utils import get_device
    print(f"[OK] Import successful. Device: {get_device()}")
except ImportError as e:
    print(f"[FAIL] Import failed: {e}")
    sys.exit(1)

# Create labels
output_path = root_dir / 'data' / 'labels.csv'
df = pd.DataFrame({
    'slide_name': ['normal/normal_001']*10 + ['tumor/tumor_001']*10,
    'label': [0]*10 + [1]*10
})
df.to_csv(output_path, index=False)
print(f"[OK] Created labels.csv at {output_path}")
print(df.head())
