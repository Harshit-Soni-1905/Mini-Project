# Environment Reference

Tested configuration for the development machine.  These are the versions
known to work together — they are documented here rather than pinned in
`pyproject.toml` because PyTorch installation depends on CUDA version and
platform.

## Tested Versions (2026-10-09)

| Component      | Version           | Notes                          |
| -------------- | ----------------- | ------------------------------ |
| Python         | 3.14.7            | CPython, MSC v.1944, 64-bit    |
| PyTorch        | 2.14.1+cu132      | CUDA 13.2 build                |
| torchvision    | 0.29.1+cu132      | Must match PyTorch CUDA build  |
| CUDA toolkit   | 13.2              | Bundled with PyTorch wheel     |
| cuDNN          | 9.24.0 (92400)    | Bundled with PyTorch wheel     |
| Ultralytics    | 8.4.174           | YOLOv11 support                |
| OpenCV         | 5.0.0             | opencv-python                  |
| NumPy          | 2.5.2             |                                |
| Pillow         | 12.3.0            |                                |
| pyttsx3        | 2.99              | Windows SAPI5 backend          |

## GPU

| Property | Value                          |
| -------- | ------------------------------ |
| Model    | NVIDIA GeForce RTX 3050 Laptop |
| VRAM     | 4096 MB                        |

## PyTorch Installation

PyTorch wheels with CUDA support are installed from the PyTorch index:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

For CPU-only machines, replace `cu132` with `cpu`.

## Reproducibility Notes

- The virtual environment lives at `.venv/` and is git-ignored.
- Model weights (`*.pt`, `*.pth`, `*.onnx`) are git-ignored; they are
  downloaded on first run by Ultralytics.
- A full `pip freeze` snapshot can be generated with:
  ```bash
  pip freeze > requirements-lock.txt
  ```
  This file is **not** committed because it contains platform-specific
  wheels.  Use `pyproject.toml` for portable dependency specification.
