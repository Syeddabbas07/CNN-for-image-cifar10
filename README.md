
# CIFAR-10 Image Classification Project

This project focuses on building a custom Convolutional Neural Network (CNN) from scratch using TensorFlow/Keras to classify images from the CIFAR-10 dataset. It includes a well-documented pipeline for exploratory data analysis (EDA), data preprocessing, model development, training, evaluation, and visualization of misclassifications.

📦 Requirements

- Python 3+
- TensorFlow
- NumPy
- Matplotlib
- seaborn
- scikit-learn

Install dependencies via:
- The following code was tested and created on Google Colab.
- No specific version used.

```bash
pip install -r requirements.txt
```

---

🚀 How It Works

1. 🧠 Exploratory Data Analysis (EDA)

- Bar plot for class distribution
- Visualization of sample images with labels

2. 🛠️ Preprocessing

- One-hot encoding for labels
- Augmentation using `ImageDataGenerator`
- 20% validation split for rigorous testing

3. 🧱 Model Architecture

- Custom functional API-based CNN model
- Dropout for regularization
- BatchNormalization for faster convergence
- Adam optimizer with cosine learning rate decay

4. 🎯 Model Training

- 50 epochs with early stopping callback
- Validation set for generalization monitoring
- Training loss and accuracy curves plotted

5. 📊 Evaluation

- Classification report with precision, recall, F1-score
- Confusion matrix with true vs predicted class names
- Baseline model (simpler Sequential CNN) for comparison

6. 🖼️ Visualization

- Misclassified images shown with true and predicted labels
- Loss curves smoothed using moving average for clarity

📄 License

This project is for academic use. You are welcome to build on it, but please reference original ideas if reused.

---
