# DL_project

# Detecting Autism Spectrum Disorder from facial images

This repository contains the code and resources for a deep learning project focused on predicting Autism Spectrum Disorder (ASD) from facial images using a Convolutional Neural Network (CNN). The project aims to provide a non-invasive method for early autism detection, leveraging advanced image analysis techniques.

# Project Overview
The goal of this project is to develop a deep learning model that predicts whether a child is autistic based on their facial images. By analyzing facial features with a CNN, the model seeks to facilitate early diagnosis and intervention, potentially improving outcomes for children with autism.

# Table of Contents

- Background
- Dataset
- Data Preprocessing
- Model Development
- Prediction and Web App
- Results and Insights
- Conclusion
- Camera Access Issue


# Background
Autism Spectrum Disorder (ASD) is a developmental disorder that affects communication, behavior, and social interactions. Early detection is critical for providing effective support and interventions. This project addresses the need for innovative diagnostic tools by applying deep learning to facial image analysis.

# Dataset
The dataset for this project consists of facial images categorized by autism diagnosis status. The images are used to train the CNN model to recognize patterns and features indicative of autism.

# Data Preprocessing
The preprocessing steps for this project include:

- Image Resizing: Adjusting image dimensions to fit the model's input requirements.
- Grayscale Conversion: Converting images to grayscale to standardize input.
- Normalization: Scaling pixel values to prepare the data for model training.

# Model Development
A Convolutional Neural Network (CNN) was chosen for this deep learning project due to its effectiveness in image classification tasks. Key steps in model development include:

- Designing a CNN architecture suitable for binary classification.
- Training the model on preprocessed facial images.
- Evaluating model performance through accuracy metrics and validation.

# Prediction and Web App
A Streamlit web app has been created to interact with the deep learning model. Users can choose between two options:

- Upload Image: Upload a facial image for prediction.
- Capture Image: Use the webcam to capture an image and obtain predictions directly.

# Results and Insights
The deep learning model provides predictions about whether a child is likely to be autistic based on facial images. The web app makes it easy for users to test the model and obtain results in real-time. This tool aims to support early autism detection and provide valuable insights for further research.

# Conclusion
This deep learning project showcases the application of CNNs in the field of autism detection from facial images. By combining sophisticated image analysis with a user-friendly web application, the project contributes to early detection efforts and enhances the support available for children with ASD.

# Camera Access Issue
If you encounter issues accessing the webcam in the deployed app, it may be due to browser or deployment environment restrictions. Here are some tips to address this:

- Browser Permissions: Ensure your browser has permission to access the webcam. Check your browser settings and grant access if prompted.
- Deployment Platform: Some platforms may restrict webcam access due to security reasons.
  Consider running the app locally if you face issues with webcam functionality in the deployed version.
