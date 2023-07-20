import argparse
import json
import torch
import torch.nn.functional as F
from torchvision import models
from PIL import Image
import numpy as np

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, len(checkpoint['class_to_idx']))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def process_image(image_path):
    image = Image.open(image_path)
    image = image.resize((256, 256))
    left = (256 - 224) / 2
    top = (256 - 224) / 2
    right = (256 + 224) / 2
    bottom = (256 + 224) / 2
    image = image.crop((left, top, right, bottom))
    np_image = (np.array(image.convert('RGB')) / 255.0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))
    tensor_image = torch.tensor(np_image).float()  # Convert to float tensor explicitly
    return tensor_image

def predict(image_path, model, topk=5, device='cpu'):
    model.eval()
    image = process_image(image_path)
    image = image.unsqueeze(0)
    image = image.float()  # Convert to float tensor explicitly
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        top_probs, top_indices = probabilities.topk(topk, dim=1)

    top_probs = top_probs.squeeze().tolist()
    top_indices = top_indices.squeeze().tolist()

    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_indices]

    # Convert probabilities to percentages with two decimal points
    top_probs_percentage = [round(prob * 100, 2) for prob in top_probs]

    return top_probs_percentage, top_classes

def main():
    parser = argparse.ArgumentParser(description='Predict flower name from an image')
    parser.add_argument('input', type=str, help='Path to the input image')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint file')
    parser.add_argument('--top_k', type=int, default=1, help='Return top KK most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to category names mapping file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    args = parser.parse_args()

    # Load the checkpoint
    model = load_checkpoint(args.checkpoint)

    # Check if GPU is available
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

    # Move the model to the appropriate device
    model = model.to(device)

    # Predict the flower name and class probability
    top_probs, top_classes = predict(args.input, model, args.top_k, device)

    # Load category names mapping
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    # Map indices to actual flower names
    flower_names = [cat_to_name[class_idx] for class_idx in top_classes]

    # Print the results
    for i in range(args.top_k):
        print(f"Prediction {i + 1}: {flower_names[i]}, Probability: {top_probs[i]:.3f}%")

if __name__ == "__main__":
    main()