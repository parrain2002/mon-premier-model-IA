import streamlit as st
import torch
from PIL import Image
from model import get_gesture_model
from torchvision import transforms

# Configuration de la page
st.set_page_config(page_title="Détecteur de Gestes", page_icon="✋")

st.title("🤖 Mon Classificateur de Gestes")
st.write("Télécharge une photo de ta main (Fist, Palm ou Thumb) pour voir si mon modèle la reconnaît !")

# 1. Chargement du modèle (identique à ton predict.py)
@st.cache_resource # Pour ne pas recharger le modèle à chaque clic
def load_my_model():
    model = get_gesture_model(num_classes=3)
    model.load_state_dict(torch.load("gesture_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_my_model()
classes = ['fist', 'palm', 'thumb']

# 2. Préparation des transformations
predict_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Interface de téléchargement
uploaded_file = st.file_uploader("Choisis une image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Affichage de l'image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Image envoyée', use_column_width=True)
    
    # Prédiction
    st.write("Analyse en cours...")
    img_tensor = predict_transforms(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, pred = torch.max(probabilities, 0)
        
    # Affichage du résultat final
    res = classes[pred.item()]
    st.success(f"Résultat : **{res}** avec une confiance de **{confidence.item()*100:.2f}%**")