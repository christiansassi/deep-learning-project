import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image

import json
from os.path import basename
from pathlib import Path
import requests

def load_labels() -> list[str]:

    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    path = Path(basename(url))

    # Check if labels file already exists
    if not path.exists():
        response = requests.get(url)
        path.write_text(response.text)

    # Load labels
    with open(path, "r") as f:
        labels = json.load(f)

    return labels

def load_model(model_name: str = "google/vit-base-patch16-384") -> tuple[ViTForImageClassification, ViTImageProcessor]:

    # Load the pre-trained model
    model = ViTForImageClassification.from_pretrained(model_name)

    # Load the image processor
    processor = ViTImageProcessor.from_pretrained(model_name)

    return model, processor

def classify_image(model: ViTForImageClassification, processor: ViTImageProcessor, img: str) -> tuple[dict, dict]:

    # Load the image
    image = Image.open(img).convert("RGB")

    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract probabilities from model's output logits
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()

    labels = load_labels()
    results = {}

    for index, probability in enumerate(probabilities):
        results[index] = {
            "index": index,
            "label": labels[index],
            "probability": probability.item()
        }

    predicted = probabilities.argmax(-1).item()
    predicted = {
        "index": predicted,
        "label": labels[predicted],
        "probability": probabilities[predicted].item()
    }

    return predicted, results

if __name__ == "__main__":

    # Load model (only once)
    model, processor = load_model()

    # Predict image
    img = "img.png"
    predicted, results = classify_image(model, processor, img)