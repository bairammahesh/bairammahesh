# Respiratory Disease Detection using Convolutional Neural Networks (CNN)

This project is an automated diagnostic system capable of identifying various lung infections—including Asthma, COPD, Pneumonia, and Bronchiectasis—by analyzing respiratory sound signals (cough and lung sounds).

## Key Features

- **Automated Diagnosis**: Uses a deep learning model to classify respiratory diseases from audio recordings.
- **User-Friendly Interface**: A desktop application built with Tkinter allows for easy interaction with the model.
- **Data Visualization**: Displays accuracy and loss graphs to visualize model performance.

## Technical Overview

- **Digital Signal Processing (DSP) Pipeline**: Engineered a robust feature extraction pipeline using **Librosa** to transform raw 1D respiratory audio signals into 2D spectral representations. Implemented **Mel-Frequency Cepstral Coefficients (MFCC)** extraction, utilizing a fixed-window approach to generate feature vectors of 6,348 dimensions, subsequently reshaped into (46, 46, 3) tensors for spatial feature learning.
- **Deep Learning Architecture**: Designed and implemented a multi-layer **Convolutional Neural Network (CNN)** using **TensorFlow/Keras**. The architecture consists of stacked `Conv2D` layers for spatial feature extraction, `MaxPooling2D` for dimensionality reduction, and a 256-node fully connected dense layer for high-level feature abstraction, culminating in an 8-way **Softmax** classifier.
- **Optimization & Training**: Employed the **Adam optimizer** with **Categorical Crossentropy** loss to minimize classification error across 8 disease classes. Implemented data normalization (min-max scaling to [0,1]) and stochastic shuffling to enhance model generalization and convergence stability over 50 training epochs.
- **Data Engineering & Integration**: Integrated heterogeneous data sources by merging raw audio datasets with relational patient metadata (CSV) using **Pandas**. Managed binary data persistence using **NumPy (.npy)** and model serialization via **JSON/H5** formats for efficient loading and inference.
- **Inference System Development**: Built an end-to-end inference engine with a **Tkinter-based GUI**, featuring real-time MFCC computation on user-uploaded `.wav` files and dynamic visualization of training metrics (accuracy/loss curves) using **Matplotlib**.

## Technical Stack

- **Frameworks**: TensorFlow 2.x, Keras, Librosa, Scikit-learn
- **Data Science**: NumPy, Pandas, Matplotlib
- **System**: Python 3.x, Tkinter (GUI), Pickle (Serialization)
- **Architecture**: CNN (Convolutional Neural Networks), MFCC (Spectral Analysis)

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Nikhiljai03/Respiratory-Sound-Diagnostic-Engine.git
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application:**
   ```bash
   python Main.py
   ```
