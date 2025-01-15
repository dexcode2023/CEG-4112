from flask import Flask, request, jsonify
from torchvision import models, transforms
from PIL import Image
import torch

app = Flask(__name__)


model = models.resnet50(pretrained=True)
model.eval()


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    image = Image.open(file.stream)
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)

    return jsonify({"prediction": predicted.item()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
