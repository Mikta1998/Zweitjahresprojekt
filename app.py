import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import EfficientNet_B4_Weights
from PIL import Image

MODEL_PATH = "EfficientNetB4/train1/best_model_loss.pt"
class_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
malignant_classes = {'akiec', 'bcc', 'mel'}

@st.cache_resource
def load_model():
    weights = EfficientNet_B4_Weights.DEFAULT
    model = models.efficientnet_b4(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

# Match input size to EfficientNetB4 (380x380)
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.CenterCrop(380),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

st.markdown("""
    <style>
        .main { background-color: #f6f6f6; }
        .title { text-align: center; font-size: 36px; font-weight: bold; margin-bottom: 0.5em; }
        .subtitle { text-align: center; font-size: 16px; color: gray; margin-bottom: 2em; }
        .prediction-card {
            padding: 20px; border-radius: 10px; background-color: #ffffff;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.05); margin-top: 20px; text-align: center;
        }
        .malignant { color: #b00020; font-weight: bold; }
        .benign { color: #0f9d58; font-weight: bold; }
        .footer { font-size: 13px; color: gray; margin-top: 2em; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>Skin Lesion Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload a dermoscopic image and get a model prediction.</div>", unsafe_allow_html=True)

st.info("**Disclaimer:** This app is for entertainment and educational use only. Not for medical use.")

uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    input_tensor = transform(image).unsqueeze(0)
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_class].item()

    predicted_label = class_labels[pred_class]
    is_malignant = predicted_label in malignant_classes
    risk_class = "malignant" if is_malignant else "benign"
    risk_text = "Malignant" if is_malignant else "Benign"

    st.markdown(f"""
        <div class="prediction-card">
            <h3>Prediction: <span class="{risk_class}">{predicted_label.upper()} ({risk_text})</span></h3>
            <p><strong>Confidence:</strong> {confidence:.2%}</p>
        </div>
    """, unsafe_allow_html=True)