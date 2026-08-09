from __future__ import print_function
import importlib
import os
import sys

project_root = sys.argv[1]
if project_root not in sys.path:
    sys.path.insert(0, project_root)

modules = [
    "numpy",
    "PIL",
    "torch",
    "torchvision",
    "pygame",
    "cv2",
]

for name in modules:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        print("[OK]   {} | version={}".format(name, version))
    except Exception as exc:
        print("[FAIL] {} | {}".format(name, repr(exc)))

project_modules = [
    "simulation.connection",
    "simulation.sensors",
    "autoencoder_rgb.encoder_rgb",
    "autoencoder_rgb.decoder_rgb",
]

for name in project_modules:
    try:
        importlib.import_module(name)
        print("[OK]   project module {}".format(name))
    except Exception as exc:
        print("[FAIL] project module {} | {}".format(name, repr(exc)))
