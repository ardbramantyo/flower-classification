import torch
from torchvision import models, transforms
import argparse
from PIL import Image

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('image_path', help='path to the image')
parser.add_argument('checkpoint', help='path to the checkpoint file')
parser.add_argument('--top_k', type=int, default=5, help='return top K most likely classes')
parser.add_argument('--category_names', help='path to category names mapping file')
parser.add_argument('--gpu', action='store_true', help='use GPU for inference')

# Parse command-line arguments
args = parser.parse_args()

# Load the checkpoint
checkpoint = torch.load(args.checkpoint)

# Load the model architecture
model = models.__dict__[checkpoint['arch']](pretrained=True)
model.fc = torch.nn.Sequential(torch.nn.Linear(model.fc.in_features, 512),
                               torch.nn.ReLU(),
                               torch.nn.Dropout(0.2),
                               torch.nn.Linear(512, len(checkpoint['class_to_idx'])),
                               torch.nn.LogSoftmax(dim=1))
model.load_state_dict(checkpoint['state_dict'])
model.class_to_idx = checkpoint['class_to_idx']

# Process the image
image = Image.open(args.image_path)
image_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image = image_transforms(image)
image = image.unsqueeze(0)

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
model.to(device)
image = image.to(device)

# Perform the prediction
model.eval()
with torch.no_grad():
    output = model(image)

# Get the top K predictions
probs, indices = torch.topk(torch.exp(output), args.top_k)
probs = probs.squeeze().tolist()
indices = indices.squeeze().tolist()

# Map indices to class labels
idx_to_class = {v: k for k, v in model.class_to_idx.items()}
labels = [idx_to_class[idx] for idx in indices]

# Load category names mapping if provided
if args.category_names:
    import json
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    labels = [cat_to_name[label] for label in labels]

# Print the top K predictions
for prob, label in zip(probs, labels):
    print(f'Class: {label}, Probability: {prob:.3f}')