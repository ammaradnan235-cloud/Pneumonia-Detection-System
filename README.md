🫁 Pneumonia Detection System (AI Powered)

An AI-based web application that detects Pneumonia from Chest X-ray images using Deep Learning (CNN) and a Streamlit interface.

📌 Project Overview

This project uses a trained TensorFlow/Keras model to classify chest X-ray images into:

NORMAL

PNEUMONIA

The application is built with Streamlit, making it easy to upload images and get predictions with confidence scores.

🚀 Features

Upload chest X-ray images (JPG / PNG / JPEG)

AI-based Pneumonia detection

Displays prediction result and confidence

Simple and user-friendly Streamlit UI

Pre-trained deep learning model (.keras)

🧠 Tech Stack

Python

TensorFlow / Keras

NumPy

Pillow (PIL)

Streamlit

MobileNetV3 (Preprocessing)

📁 Project Structure
├── app.py                     # Streamlit application
├── pneumonia_model.keras      # Trained deep learning model
├── requirements.txt           # Required Python libraries
├── README.md                  # Project documentation

⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/your-username/pneumonia-detection-app.git
cd pneumonia-detection-app

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the Streamlit app
streamlit run app.py

🖼️ How to Use

Open the app in your browser

Upload a chest X-ray image

Click Detect Pneumonia

View prediction result and confidence score

📊 Model Details

Image size: 128 × 128

Input: RGB Chest X-ray image

Output classes:

NORMAL

PNEUMONIA

Framework: TensorFlow Keras
