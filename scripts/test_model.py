from pathlib import Path

import torch

from models.hybrid_model import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "hybrid_resnet_densenet_checkpoint.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, class_names = load_checkpoint(str(MODEL), DEVICE)

print("Model loaded successfully")
print(f"Device: {DEVICE}")
print(f"Classes: {len(class_names)}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

with torch.inference_mode():
    output = model(torch.randn(1, 3, 224, 224, device=DEVICE))

print(f"Output shape: {tuple(output.shape)}")
print("Inference smoke test passed")
