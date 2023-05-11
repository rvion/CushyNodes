
import sys
import subprocess
import importlib

def install():
    required_packages = {
        "requests": "requests",
        "segment_anything": "git+https://github.com/facebookresearch/segment-anything.git",
        "numpy": "numpy",
        "clip": "git+https://github.com/openai/CLIP.git",
    }

    for package_key, package_name in required_packages.items():
        try:
            importlib.import_module(package_key)
        except ImportError:
            print(f"Installing {package_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
