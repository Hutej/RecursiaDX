"""
OpenSlide DLL and VIPS Setup for Windows

Automatically detects and configures OpenSlide/VIPS paths on Windows.
Provides helpful error messages if DLLs are not found.
"""
import os
import sys
import platform
from pathlib import Path


def setup_openslide_windows():
    """
    Set up OpenSlide DLLs for Windows.
    
    Searches common installation locations and adds to PATH if found.
    Provides installation instructions if not found.
    """
    if platform.system() != "Windows":
        return  # Only needed on Windows
    
    # Common OpenSlide installation directories
    search_paths = [
        Path(r"C:\openslide-win64\bin"),
        Path(r"C:\openslide\bin"),
        Path(r"C:\Program Files\OpenSlide\bin"),
        Path(r"C:\Program Files (x86)\OpenSlide\bin"),
        Path.home() / "openslide" / "bin",
        Path.home() / "Downloads" / "openslide-win64-20231011" / "bin",
        Path.home() / "Downloads" / "openslide-win64" / "bin",
    ]
    
    # Check if already in PATH
    current_path = os.environ.get("PATH", "")
    if "openslide" in current_path.lower():
        return  # Already configured
    
    # Search for OpenSlide installation
    openslide_found = False
    for search_path in search_paths:
        if search_path.exists():
            dll_file = search_path / "libopenslide-0.dll"
            if dll_file.exists():
                # Add to PATH
                os.environ["PATH"] = str(search_path) + os.pathsep + os.environ["PATH"]
                print(f"✓ OpenSlide DLLs found and added to PATH: {search_path}")
                openslide_found = True
                break
    
    if not openslide_found:
        print("\n" + "="*70)
        print("⚠️  OpenSlide DLL Not Found")
        print("="*70)
        print("\nOpenSlide is required for WSI processing but was not detected.")
        print("\n📥 Installation Instructions:")
        print("\n1. Download OpenSlide Windows binaries:")
        print("   https://openslide.org/download/")
        print("\n2. Extract to one of these locations:")
        for path in search_paths[:4]:
            print(f"   - {path.parent}")
        print("\n3. Restart your terminal/IDE")
        print("\nAlternative: Add OpenSlide bin directory to your System PATH")
        print("="*70)
        print("\n⚠️  Pipeline will fail if you proceed without OpenSlide installed.\n")


def setup_vips_windows():
    """
    Set up VIPS DLLs for Windows (optional, for heatmap generation).
    
    Only prints warning if not found, doesn't block execution.
    """
    if platform.system() != "Windows":
        return
    
    # Check if VIPS is in PATH
    current_path = os.environ.get("PATH", "")
    if "vips" in current_path.lower():
        return  # Already configured
    
    # Common VIPS installation directories
    vips_paths = [
        Path(r"C:\vips\bin"),
        Path(r"C:\Program Files\vips\bin"),
        Path.home() / "vips" / "bin",
    ]
    
    for vips_path in vips_paths:
        if vips_path.exists():
            os.environ["PATH"] = str(vips_path) + os.pathsep + os.environ["PATH"]
            print(f"✓ VIPS found and added to PATH: {vips_path}")
            return
    
    # VIPS is optional, just note if missing
    # (heatmap generation will fall back to alternative methods)


# Run setup when module is imported
if platform.system() == "Windows":
    setup_openslide_windows()
    setup_vips_windows()
