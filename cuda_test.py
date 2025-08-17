import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
print(f"GPU count: {torch.cuda.device_count()}")

# 簡単なGPUテスト
if torch.cuda.is_available():
    x = torch.rand(5, 3).cuda()
    y = torch.rand(5, 3).cuda()
    z = x + y
    print(f"GPU computation successful: {z.shape}")
    print("✅ PyTorch with CUDA is working correctly!")
else:
    print("❌ CUDA not available")