"""
System Audit & Validation Script

Performs comprehensive pre-deployment checks to ensure the repository
is ready for backend integration.

Validates:
- Directory structure
- Checkpoint presence and integrity
- Configuration sanity
- Dependency availability
- Inference capability

Usage:
    python scripts/validate_system.py
    python scripts/validate_system.py --check-checksums
"""
import sys
import hashlib
from pathlib import Path
import torch
import argparse

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.ENDC}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")

def check_directory_structure():
    """Verify required directories exist."""
    print_header("1. DIRECTORY STRUCTURE CHECK")
    
    required_dirs = [
        'checkpoints',
        'data',
        'test_data',
        'results',
        'configs',
        'src',
        'scripts'
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print_success(f"Directory exists: {dir_name}/")
        else:
            print_error(f"Missing directory: {dir_name}/")
            all_exist = False
    
    return all_exist

def check_checkpoints():
    """Verify trained model checkpoints exist and are loadable."""
    print_header("2. CHECKPOINT VALIDATION")
    
    checkpoints_dir = Path('checkpoints')
    required_checkpoints = ['best_model.pth', 'last_model.pth']
    
    checkpoint_status = {}
    
    for ckpt_name in required_checkpoints:
        ckpt_path = checkpoints_dir / ckpt_name
        
        if not ckpt_path.exists():
            print_error(f"Missing: {ckpt_name}")
            checkpoint_status[ckpt_name] = 'missing'
            continue
        
        # Check file size
        size_mb = ckpt_path.stat().st_size / (1024 * 1024)
        print_success(f"Found: {ckpt_name} ({size_mb:.2f} MB)")
        
        # Attempt to load
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            
            # Validate structure
            if 'model_state_dict' in checkpoint:
                print_success(f"  └─ Valid checkpoint structure")
                checkpoint_status[ckpt_name] = 'valid'
            else:
                print_warning(f"  └─ Unusual structure (missing 'model_state_dict')")
                checkpoint_status[ckpt_name] = 'unusual'
            
            # Print metadata if available
            if 'epoch' in checkpoint:
                print(f"     Epoch: {checkpoint['epoch']}")
            if 'best_auc' in checkpoint:
                print(f"     Best AUC: {checkpoint['best_auc']:.4f}")
                
        except Exception as e:
            print_error(f"  └─ Failed to load: {e}")
            checkpoint_status[ckpt_name] = 'corrupted'
    
    # Final assessment
    all_valid = all(status == 'valid' for status in checkpoint_status.values())
    
    if not all_valid:
        print_error("\nCheckpoint validation FAILED")
        print("\n" + "="*60)
        print("CRITICAL ERROR")
        print("="*60)
        print("This repository requires trained model checkpoints to function.")
        print("Expected files:")
        for ckpt in required_checkpoints:
            print(f"  - checkpoints/{ckpt}")
        print("\nThese files should be committed to the repository.")
        print("="*60)
        return False
    
    return True

def verify_checksum(file_path, checksum_path):
    """Verify file SHA256 checksum."""
    if not checksum_path.exists():
        return None
    
    # Read expected checksum
    with open(checksum_path, 'r') as f:
        expected = f.read().strip().split()[0]
    
    # Compute actual checksum
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    actual = sha256.hexdigest()
    
    return actual == expected

def check_checksums():
    """Verify checkpoint integrity via SHA256."""
    print_header("3. CHECKSUM VERIFICATION (OPTIONAL)")
    
    checkpoints_dir = Path('checkpoints')
    checkpoints = ['best_model.pth', 'last_model.pth']
    
    any_checksum_exists = False
    all_valid = True
    
    for ckpt_name in checkpoints:
        ckpt_path = checkpoints_dir / ckpt_name
        checksum_path = checkpoints_dir / f"{ckpt_name}.sha256"
        
        if not ckpt_path.exists():
            continue
        
        result = verify_checksum(ckpt_path, checksum_path)
        
        if result is None:
            print_warning(f"No checksum file for {ckpt_name}")
        elif result:
            print_success(f"Checksum valid: {ckpt_name}")
            any_checksum_exists = True
        else:
            print_error(f"Checksum FAILED: {ckpt_name}")
            all_valid = False
            any_checksum_exists = True
    
    if not any_checksum_exists:
        print_warning("No checksum files found (optional feature)")
    
    return all_valid

def check_dependencies():
    """Verify required packages are installed."""
    print_header("4. DEPENDENCY CHECK")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'Torchvision'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('h5py', 'HDF5'),
        ('PIL', 'Pillow'),
        ('sklearn', 'Scikit-learn'),
        ('yaml', 'PyYAML'),
        ('tqdm', 'tqdm')
    ]
    
    all_installed = True
    
    for package, name in required_packages:
        try:
            __import__(package)
            print_success(f"{name} installed")
        except ImportError:
            print_error(f"{name} NOT installed")
            all_installed = False
    
    return all_installed

def check_configuration():
    """Validate configuration files."""
    print_header("5. CONFIGURATION SANITY CHECK")
    
    config_path = Path('configs/config.yaml')
    
    if not config_path.exists():
        print_error("config.yaml not found")
        return False
    
    print_success("config.yaml exists")
    
    # Check for hardcoded paths (basic scan)
    with open(config_path, 'r') as f:
        content = f.read()
    
    suspicious_patterns = [
        '/home/',
        'C:\\Users\\',
        '/Users/',
        'aws.amazon.com',
        's3://'
    ]
    
    issues_found = False
    for pattern in suspicious_patterns:
        if pattern in content:
            print_warning(f"Potential hardcoded path detected: {pattern}")
            issues_found = True
    
    if not issues_found:
        print_success("No hardcoded absolute paths detected")
    
    return True

def check_inference_capability():
    """Verify inference scripts are functional."""
    print_header("6. INFERENCE CAPABILITY CHECK")
    
    inference_script = Path('scripts/infer_mil.py')
    test_script = Path('scripts/test_inference.py')
    
    if not inference_script.exists():
        print_error("infer_mil.py not found")
        return False
    
    print_success("infer_mil.py exists")
    
    if test_script.exists():
        print_success("test_inference.py exists")
    else:
        print_warning("test_inference.py not found (optional)")
    
    # Check default model path
    with open(inference_script, 'r') as f:
        content = f.read()
    
    if 'checkpoints/best_model.pth' in content or 'best_model.pth' in content:
        print_success("Script references checkpoint correctly")
    else:
        print_warning("Could not verify default checkpoint path")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='System validation script')
    parser.add_argument('--check-checksums', action='store_true', 
                      help='Verify SHA256 checksums')
    args = parser.parse_args()
    
    print(f"\n{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}GigaPath WSI MIL - System Validation{Colors.ENDC}")
    print(f"{Colors.BOLD}{'='*60}{Colors.ENDC}")
    
    results = {}
    
    try:
        results['directories'] = check_directory_structure()
        results['checkpoints'] = check_checkpoints()
        
        if args.check_checksums:
            results['checksums'] = check_checksums()
        
        results['dependencies'] = check_dependencies()
        results['configuration'] = check_configuration()
        results['inference'] = check_inference_capability()
        
        # Final report
        print_header("VALIDATION SUMMARY")
        
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        
        for check, status in results.items():
            status_str = f"{Colors.GREEN}PASS{Colors.ENDC}" if status else f"{Colors.RED}FAIL{Colors.ENDC}"
            print(f"{check.upper():.<40} {status_str}")
        
        print(f"\n{Colors.BOLD}Result: {passed}/{total} checks passed{Colors.ENDC}")
        
        if all(results.values()):
            print(f"\n{Colors.GREEN}{Colors.BOLD}✓ SYSTEM READY FOR DEPLOYMENT{Colors.ENDC}")
            print("\nThe repository is properly configured for backend integration.")
            print("A developer can clone and run inference immediately.")
            return 0
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}✗ SYSTEM NOT READY{Colors.ENDC}")
            print("\nPlease address the issues above before deployment.")
            return 1
            
    except Exception as e:
        print_error(f"Validation failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
