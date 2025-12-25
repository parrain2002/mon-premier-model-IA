# predict.py
import torch
from PIL import Image
from torchvision import transforms
from model import get_gesture_model

# 1. Configurer les transformations (doivent être IDENTIQUES au test)
predict_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2. Charger le modèle et ses poids entraînés
classes = ['fist', 'palm', 'thumb']
model = get_gesture_model(num_classes=3)
model.load_state_dict(torch.load("gesture_model.pth")) # On charge ton travail !
model.eval()

def predict(image_path):
    # Charger l'image
    img = Image.open(image_path).convert('RGB')
    
    # Appliquer les transformations
    img_tensor = predict_transforms(img).unsqueeze(0) # Ajoute la dimension "batch" (1, 3, 224, 224)
    
    # Prédire sans calculer de gradients
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, pred = torch.max(probabilities, 0)
    
    print(f"Résultat : {classes[pred.item()]} ({confidence.item()*100:.2f}%)")

if __name__ == "__main__":
    # Teste avec une image de ton choix
    predict("test_hand.jpg")