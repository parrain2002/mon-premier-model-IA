# model.py
import torch.nn as nn
from torchvision import models

def get_gesture_model(num_classes=3):
    # 1. Charger ResNet18 pré-entraîné
    model = models.resnet18(pretrained=True)
    # Tu télécharges un modèle ResNet-18 qui a déjà appris à reconnaître 1 000 objets différents (animaux, voitures, etc.) sur la base de données ImageNet. Le modèle possède déjà une "compréhension" visuelle du monde
    
    # 2. Geler tous les paramètres (poids)
    for param in model.parameters():
        param.requires_grad = False
        # Cette boucle parcourt tous les neurones du modèle et les "verrouille".
        # ici On ne veut pas modifier les couches qui savent déjà reconnaître des formes (lignes, textures). Cela rend l'entraînement beaucoup plus rapide et évite de "casser" ce que le modèle sait déjà faire.
    
    # 3. Remplacer la dernière couche pour nos 3 gestes
    num_ftrs = model.fc.in_features
    # C'est la dernière couche (Fully Connected). À l'origine, elle avait 1 000 sorties.
    # in_features : On récupère le nombre de connexions entrantes (512 pour ResNet-18)
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model