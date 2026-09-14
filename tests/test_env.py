import sys

def verify():
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version: {sys.version}")
    
    import numpy as np
    print(f"NumPy: {np.__version__}")
    
    import torch
    print(f"PyTorch: {torch.__version__} (CUDA Available: {torch.cuda.is_available()})")
    
    import fastapi
    print(f"FastAPI: {fastapi.__version__}")
    
    import sklearn
    print(f"Scikit-learn: {sklearn.__version__}")
    
    import scipy
    print(f"SciPy: {scipy.__version__}")
    
    import PIL
    print(f"Pillow: {PIL.__version__}")
    
    print("Environment Verification: ALL CHECKS PASSED!")

if __name__ == "__main__":
    verify()
