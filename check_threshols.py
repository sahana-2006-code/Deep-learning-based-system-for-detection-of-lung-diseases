import torch
ckpt = torch.load("best_model_full.pth", map_location="cpu")
classes    = ckpt["classes"]
thresholds = ckpt["thresholds"]

print("\nPer-Class Thresholds from your trained model:")
print("-" * 40)
for cls, t in zip(classes, thresholds):
    print(f"  {cls:20s}: {t:.2f}")
