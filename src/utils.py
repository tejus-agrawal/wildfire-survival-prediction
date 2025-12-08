import torch
import platform

def get_device():
    """
    Returns the appropriate PyTorch device based on available hardware.
    Prioritizes MPS (Apple Silicon), then CUDA (NVIDIA), then CPU.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✅ Using Apple MPS (Neural Engine)")
        print(f"   PyTorch Version: {torch.__version__}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Using NVIDIA CUDA")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU (Slower)")
    
    return device

def get_device_str():
    """Returns the device string ('mps', 'cuda', 'cpu')"""
    device = get_device()
    return str(device).split(':')[0] if ':' in str(device) else str(device)

