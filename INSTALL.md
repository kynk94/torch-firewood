# Installation

## Requirements

### Common requirements
- Linux or macOS or Windows
- Python >= 3.8
- PyTorch >= 1.7.0
- torchvision that matches the PyTorch installation. You can install them together as explained at pytorch.org to make sure of this.

### CUDA requirements
- gcc & g++ ≥ 9 for CUDA extensions

### Tests/Linting

For developing on top of `torch-firewood` or contributing, need to run the linter and tests.
```bash
# Tests/Linting
pip install -r requirements-dev.txt
git submodule -q update --init --recursive
pre-commit install
```

## Install
After installing the above dependencies, run one of the following commands:

### 1. Install from PyPI.
```bash
pip install torch-firewood
```

### 2. Install from a local clone
```bash
git clone https://github.com/kynk94/torch-firewood.git
cd torch-firewood && python setup.py install
```

### 3. Install after build from a local clone 
```bash
git clone https://github.com/kynk94/torch-firewood.git
cd torch-firewood
chmod +x ./install.sh && sh ./install.sh
```
