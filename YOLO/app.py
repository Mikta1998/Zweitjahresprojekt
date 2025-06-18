import streamlit as st
import os
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import numpy as np

# Function for prediction using YOLO model
def prediction_yolo(yolo_path: str, image: Image.Image):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_yolo = YOLO(yolo_path)

    transform_test = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    image_tensor = transform_test(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model_yolo(image_tensor)
        # Use top1 for prediction
        predicted_class = outputs[0].probs.top1

    return predicted_class

# Main function of the Streamlit app
def main():
    st.set_page_config(page_title="Hautläsion Klassifikation", page_icon="🩺", layout="wide")
    
    st.markdown("<h1 style='text-align: center;'>🔍 Hautläsion Klassifikation</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center;'>
        Willkommen zu unserer <strong>Hautläsion Klassifikations-App</strong>! Nutzen Sie diese App, um zu überprüfen, ob eine Hautläsion bösartig oder gutartig ist.
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.header("Über diese App")
    st.sidebar.write("""
    Diese App verwendet ein vortrainiertes YOLO-Modell zur Klassifikation von Hautläsionen.
    Wählen Sie einfach ein Bild aus den vorhandenen Bildern aus oder laden Sie Ihr eigenes Bild hoch, um eine Vorhersage zu erhalten.
    """)
    
    # Directory with existing images
    image_dir = 'YOLO/images/test'
    categories = ['boesartig', 'gutartig']

    image_files = []
    for category in categories:
        category_path = os.path.join(image_dir, category)
        if os.path.exists(category_path):
            files = [os.path.join(category, f) for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
            image_files.extend(files)

    # Two column layout
    col1, col2 = st.columns(2)

    with col1:
        selected_image = st.selectbox("Wählen Sie ein Bild aus den vorhandenen Bildern aus:", image_files)
        uploaded_image = st.file_uploader("Oder laden Sie ein Bild hoch:", type=["jpg", "jpeg", "png"])

        if uploaded_image:
            image = Image.open(uploaded_image).convert('RGB')
            st.image(image, caption='Hochgeladenes Bild', use_column_width=True)
        elif selected_image:
            image_path = os.path.join(image_dir, selected_image)
            image = Image.open(image_path)
            st.image(image, caption='Ausgewähltes Bild', use_column_width=True)

    with col2:
        if uploaded_image or selected_image:
            st.markdown("<h3 style='text-align: center;'>Vorhersage</h3>", unsafe_allow_html=True)
            if st.button("Vorhersagen"):
                with st.spinner("Vorhersage wird durchgeführt..."):
                    try:
                        YOLO_PATH = 'runs/classify/train3/weights/best_for_2_classes.pt'  
                        prediction = prediction_yolo(YOLO_PATH, image)
                        labels = ["Bösartig", "Gutartig"]  

                        
                        st.write(f"Predicted class index: {prediction}")
                        st.write(f"Predicted label: {labels[prediction]}")

                        label = labels[prediction]
                        if label == "Bösartig":
                            st.error(f"**Vorhersage:** {label}")
                        else:
                            st.success(f"**Vorhersage:** {label}")
                    except Exception as e:
                        st.error(f"Fehler bei der Vorhersage: {e}")

if __name__ == '__main__':
    main()