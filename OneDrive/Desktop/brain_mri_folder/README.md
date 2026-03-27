# 🧠 Brain Tumor MRI Classification Dashboard

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B)

## 📌 Overview
This project is an AI-powered diagnostic support system designed to classify brain MRI scans. Built with **PyTorch** and deployed via a custom **Streamlit** dashboard, the model utilizes a fine-tuned **ResNet50** architecture to identify potential abnormalities in neuro-imaging with high accuracy.

The user interface features a modern, glassmorphism-inspired dark mode dashboard for seamless clinical use.

## 🔬 Classification Categories
The neural network has been trained to detect and classify four specific conditions:
1. **Glioma Tumor**
2. **Meningioma Tumor**
3. **Pituitary Tumor**
4. **No Tumor** (Healthy)

## ⚙️ Features
* **Real-time Inference:** Upload an MRI scan and receive instant predictions.
* **Confidence Scoring:** The system outputs a percentage-based confidence metric for its primary diagnosis.
* **Probability Map:** View the exact probability breakdown across all four categories.
* **Custom UI:** A highly customized Streamlit interface bypassing default themes for a professional "Medical Dashboard" aesthetic.

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/ML-MODELS-OF-2026.git
cd ML-MODELS-OF-2026
```

### 2. Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

*(Or manually: `pip install torch torchvision streamlit pillow`)*

### 3. Run the Dashboard
```bash
streamlit run app.py
```
*Note: If the background image does not load immediately, click the three dots (⋮) in the top right of the browser app, go to Settings, and change the Theme to "Light".*

## 📁 Repository Structure
* `app.py`: The main Streamlit web application and UI code.
* `best_brain_mri_model.pth`: The trained ResNet50 weights (Managed via Git LFS).
* `image_f4f5a3.jpg`: The background UI asset.
* `README.md`: Project documentation.

## ⚠️ Disclaimer
*This tool is for educational and research purposes only. It is not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified health provider with any questions you may have regarding a medical condition.*
