import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image

# --- 1. Page Setup ---
st.set_page_config(page_title="Brain MRI Classifier", page_icon="🧠")

# --- 2. Load Model ---
# We rebuild the "empty" architecture first, then load your saved weights
@st.cache_resource 
def load_model():
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 4) 

    # Load weights from the local directory (Make sure the .pth file is in the exact same folder)
    # Using map_location='cpu' ensures it boots up smoothly on any laptop/desktop
    model.load_state_dict(torch.load('best_brain_mri_model.pth', map_location=torch.device('cpu')))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    return model, device

model, device = load_model()
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# --- 3. Image Preprocessing ---
# This must match the 'Testing' transforms from your original training script exactly
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 4. The User Interface ---
st.title("🧠 Neuro-Imaging Classification Assistant")
st.write("Upload a Brain MRI scan to detect potential abnormalities.")

uploaded_file = st.file_uploader("Choose an MRI image (JPG/PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded MRI Scan', use_container_width=True)
    
    st.write("Analyzing scan...")
    
    # Preprocess the image and push it to the correct hardware device
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make the Prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        _, predicted_class = torch.max(output, 1)
        
    # Extract results
    result = class_names[predicted_class.item()]
    confidence = probabilities[predicted_class.item()].item() * 100
    
    # Display results
    st.success(f"**Diagnosis:** {result}")
    st.info(f"**Confidence:** {confidence:.2f}%")