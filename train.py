import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Importation de ton modèle depuis ton fichier model.py
from model import get_gesture_model 

def get_loaders(data_dir, batch_size=32):
    # Transformations pour l'entraînement
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Transformations pour le test (pas d'augmentation de données)
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Chargement des datasets
    train_set = datasets.ImageFolder(root=f"{data_dir}/train", transform=train_transforms)
    test_set = datasets.ImageFolder(root=f"{data_dir}/test", transform=test_transforms)

    # Création des loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

if __name__ == "__main__":
    # 1. Préparer les données
    train_loader, test_loader = get_loaders("./data")
    print(f"Dataset prêt : {len(train_loader.dataset)} images d'entraînement.")

    # 2. Charger le modèle (3 classes : fist, palm, thumb)
    model = get_gesture_model(num_classes=3)

    # 3. Configurer l'entraînement
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # 4. Boucle d'entraînement
    print("Début de l'entraînement...")
    for epoch in range(5):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/5 - Perte: {running_loss/len(train_loader):.4f}")

    # 5. Sauvegarder le modèle
    torch.save(model.state_dict(), "gesture_model.pth")
    print("Modèle sauvegardé sous 'gesture_model.pth'")

    # 6. Évaluation finale
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Précision finale sur le test : {100 * correct / total:.2f}%")