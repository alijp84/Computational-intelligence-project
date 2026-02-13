#!/usr/bin/env python3
"""Verify MetaMind dependencies are correctly installed."""

import sys
import importlib

REQUIRED_PACKAGES = [
    ("numpy", "1.21.0"),
    ("scipy", "1.7.0"),
    ("pandas", "1.3.0"),
    ("sklearn", "1.0.0"),
    ("openai", "1.0.0", True),  # Optional (mock mode works without)
    ("matplotlib", "3.5.0", True),  # Optional (visualization)
]

def check_package(name, min_version, optional=False):
    try:
        pkg = importlib.import_module(name)
        version = pkg.__version__
        
        # Special handling for sklearn (imported as sklearn but package is scikit-learn)
        if name == "sklearn":
            import sklearn
            version = sklearn.__version__
        
        # Version comparison (simple string comparison works for semantic versions)
        if version < min_version:
            print(f"❌ {name} {version} < required {min_version}")
            return False
        
        print(f"✅ {name} {version} >= {min_version}")
        return True
        
    except ImportError:
        if optional:
            print(f"⚠️  {name} not installed (optional for mock mode/visualization)")
            return True
        else:
            print(f"❌ {name} not installed (REQUIRED)")
            return False

def main():
    print("MetaMind Dependency Verification")
    print("=" * 50)
    
    all_ok = True
    for pkg_spec in REQUIRED_PACKAGES:
        if len(pkg_spec) == 2:
            name, min_ver = pkg_spec
            optional = False
        else:
            name, min_ver, optional = pkg_spec
        
        if not check_package(name, min_ver, optional):
            all_ok = False
    
    print("=" * 50)
    if all_ok:
        print("✅ All required dependencies satisfied")
        return 0
    else:
        print("❌ Missing required dependencies")
        print("\nInstall with: pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())