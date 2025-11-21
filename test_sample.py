import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os

# -----------------------------
# Inputs / configuration
# -----------------------------

# Pull base directory from dev.env
base_dir = '/service'

# Configuration
model_path = f'{base_dir}/model/model.pth'
image_path = f'{base_dir}/screenshot.png'

target_markers = ['CD45RA', 'CD8', 'CD38', 'CCR7', 'CD20']

pdf_path = 'outputs/marker_pairplots_by_subject_with_labels.pdf'

# Check if files exist
if not os.path.exists(model_path):
    print(f"ERROR: Model not found at {model_path}")
    exit(1)

if not os.path.exists(image_path):
    print(f"ERROR: Image not found at {image_path}")
    print(f"\nPlease set image_path to an existing image file.")
    exit(1)

print(f"Image: {image_path}")
print(f"Model: {model_path}")

# Load model
print("\nLoading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()
print("✓ Model loaded")

# Load and preprocess image
print("\nLoading image...")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img = Image.open(image_path).convert('RGB')
print(f"✓ Image loaded: {img.size}")

img_tensor = transform(img).unsqueeze(0).to(device)

# Make prediction
print("\nRunning prediction...")
with torch.no_grad():
    output = model(img_tensor)
    prob = torch.sigmoid(output).item()
    pred = 1 if prob > 0.5 else 0

# Display results
print("\n" + "="*80)
print("RESULTS")
print("="*80)

pred_label = "SPILLOVER" if pred == 1 else "CLEAN"
confidence = abs(prob - 0.5) / 0.5 * 100

print(f"\nPrediction: {pred_label}")
print(f"Probability: {prob:.4f}")
print(f"Confidence: {confidence:.1f}%")

print("\n" + "="*80)
