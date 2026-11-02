"""Report the active FECD/BoxPromptIML runtime without installing packages."""

from __future__ import annotations

import importlib
import json
import sys
from importlib.metadata import PackageNotFoundError, version


PACKAGES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy",
    "opencv-python": "cv2",
    "timm": "timm",
    "albumentations": "albumentations",
    "tensorboardX": "tensorboardX",
    "scikit-learn": "sklearn",
    "thop": "thop",
    "segment-anything": "segment_anything",
    "scipy": "scipy",
    "Pillow": "PIL",
    "tqdm": "tqdm",
    "PyYAML": "yaml",
}


def main() -> None:
    report = {"python": sys.version, "executable": sys.executable, "packages": {}}
    for dist, module_name in PACKAGES.items():
        try:
            dist_version = version(dist)
        except PackageNotFoundError:
            dist_version = None
        try:
            module = importlib.import_module(module_name)
            import_error = None
            module_version = getattr(module, "__version__", None)
            module_path = getattr(module, "__file__", None)
        except Exception as exc:  # report exact environment failures
            import_error = f"{type(exc).__name__}: {exc}"
            module_version = None
            module_path = None
        report["packages"][dist] = {
            "distribution_version": dist_version,
            "module_version": module_version,
            "module_path": module_path,
            "import_error": import_error,
        }

    try:
        from boxpromptiml import BoxPromptIML
        report["boxpromptiml_import"] = {"ok": True, "class": str(BoxPromptIML)}
    except Exception as exc:
        report["boxpromptiml_import"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    import torch
    report["cuda"] = {
        "available": torch.cuda.is_available(),
        "version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["boxpromptiml_import"]["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
