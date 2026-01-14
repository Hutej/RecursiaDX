"""
Phase 0.1 Setup Verification Script
Validates installation and environment setup.
"""
import sys
from pathlib import Path

def print_section(title):
    """Print section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def check_python_version():
    """Check Python version."""
    print_section("Python Version Check")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version >= (3, 8):
        print("✓ Python version OK (3.8+)")
        return True
    else:
        print("✗ Python version too old. Need 3.8+")
        return False

def check_imports():
    """Check critical imports."""
    print_section("Dependency Check")
    
    checks = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'yaml': 'PyYAML',
        'PIL': 'Pillow',
        'cv2': 'OpenCV',
        'h5py': 'H5py',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'sklearn': 'scikit-learn',
        'tqdm': 'tqdm',
    }
    
    results = {}
    for module, name in checks.items():
        try:
            __import__(module)
            print(f"✓ {name} installed")
            results[name] = True
        except ImportError:
            print(f"✗ {name} NOT installed")
            results[name] = False
    
    return all(results.values())

def check_openslide():
    """Check OpenSlide installation (may fail on some systems)."""
    print_section("OpenSlide Check (Optional)")
    try:
        import openslide
        print(f"✓ OpenSlide installed (version: {openslide.__version__})")
        return True
    except ImportError:
        print("⚠ OpenSlide NOT installed (required for WSI processing)")
        print("  Install: pip install openslide-python")
        print("  Note: May require system-level OpenSlide library")
        return False

def check_gpu():
    """Check GPU availability."""
    print_section("GPU Check")
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            print(f"✓ CUDA available")
            print(f"  GPU 0: {gpu_name}")
            print(f"  Total GPUs: {gpu_count}")
            
            # Check VRAM
            if hasattr(torch.cuda, 'get_device_properties'):
                props = torch.cuda.get_device_properties(0)
                vram_gb = props.total_memory / (1024**3)
                print(f"  VRAM: {vram_gb:.2f} GB")
            
            return True
        else:
            print("⚠ CUDA NOT available (will use CPU)")
            print("  Note: Training will be significantly slower")
            return False
    except Exception as e:
        print(f"✗ Error checking GPU: {e}")
        return False

def check_project_structure():
    """Check project directory structure."""
    print_section("Project Structure Check")
    
    required_dirs = [
        'src/utils',
        'src/preprocessing',
        'src/feature_extraction',
        'src/sampling',
        'src/mil',
        'src/explainability',
        'configs',
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ NOT found")
            all_exist = False
    
    return all_exist

def check_config_file():
    """Check configuration file."""
    print_section("Configuration File Check")
    
    config_path = Path("configs/config.yaml")
    if config_path.exists():
        print(f"✓ config.yaml found")
        
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check required sections
            required = ['experiment', 'preprocessing', 'feature_extraction', 
                       'sampling', 'mil', 'hardware', 'paths']
            
            for section in required:
                if section in config:
                    print(f"  ✓ Section '{section}' present")
                else:
                    print(f"  ✗ Section '{section}' missing")
            
            # Check reproducibility config
            if 'experiment' in config:
                if 'seed' in config['experiment']:
                    print(f"  ✓ Seed configured: {config['experiment']['seed']}")
                if 'deterministic' in config['experiment']:
                    print(f"  ✓ Deterministic mode: {config['experiment']['deterministic']}")
            
            return True
        except Exception as e:
            print(f"✗ Error loading config: {e}")
            return False
    else:
        print("✗ config.yaml NOT found")
        return False

def check_utils_modules():
    """Check utility modules."""
    print_section("Utility Modules Check")
    
    try:
        sys.path.insert(0, str(Path.cwd()))
        
        from src.utils import setup_logger, set_seed, GPUMonitor, get_safe_batch_size
        from src.utils.config import load_config
        
        print("✓ logger module OK")
        print("✓ seed module OK")
        print("✓ gpu_monitor module OK")
        print("✓ config module OK")
        
        return True
    except ImportError as e:
        print(f"✗ Error importing utils: {e}")
        return False

def main():
    """Run all verification checks."""
    print("\n" + "="*60)
    print("  GigaPath WSI Pipeline - Phase 0.1 Verification")
    print("="*60)
    
    results = {}
    
    # Run all checks
    results['Python'] = check_python_version()
    results['Dependencies'] = check_imports()
    results['OpenSlide'] = check_openslide()
    results['GPU'] = check_gpu()
    results['Structure'] = check_project_structure()
    results['Config'] = check_config_file()
    results['Utils'] = check_utils_modules()
    
    # Summary
    print_section("Verification Summary")
    
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} - {check}")
    
    # Overall result
    critical = ['Python', 'Dependencies', 'Structure', 'Config', 'Utils']
    critical_passed = all(results[k] for k in critical if k in results)
    
    print("\n" + "="*60)
    if critical_passed:
        print("✓ VERIFICATION SUCCESSFUL - Ready for Phase 0.2")
    else:
        print("✗ VERIFICATION FAILED - Please fix errors above")
    print("="*60 + "\n")
    
    return 0 if critical_passed else 1

if __name__ == "__main__":
    sys.exit(main())
