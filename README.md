
# CIFAR-10 Image Classification Project

This project focuses on building a custom Convolutional Neural Network (CNN) from scratch using TensorFlow/Keras to classify images from the CIFAR-10 dataset. It includes a well-documented pipeline for exploratory data analysis (EDA), data preprocessing, model development, training, evaluation, and visualization of misclassifications.

---

## 📁 Project Structure

```
├── model.py                  # Custom CNN architecture
├── train.py                  # Model training pipeline
├── evaluate.py               # Evaluation scripts and confusion matrix
├── visualization.py          # Plot learning curves and misclassifications
├── README.md                 # Project documentation (this file)
```

---

## 📦 Requirements

- Python 3.8+
- TensorFlow >= 2.9
- NumPy
- Matplotlib
- seaborn
- scikit-learn

Install dependencies via:

```bash
pip install -r requirements.txt
```

---

## 🚀 How It Works

### 1. 🧠 Exploratory Data Analysis (EDA)

- Bar plot for class distribution
- Visualization of sample images with labels

### 2. 🛠️ Preprocessing

- One-hot encoding for labels
- Augmentation using `ImageDataGenerator`
- 20% validation split for rigorous testing

### 3. 🧱 Model Architecture

- Custom functional API-based CNN model
- SpatialDropout2D for regularization
- BatchNormalization for faster convergence
- Adam optimizer with exponential learning rate decay

### 4. 🎯 Model Training

- 50 epochs with early stopping callback
- Validation set for generalization monitoring
- Training loss and accuracy curves plotted

### 5. 📊 Evaluation

- Classification report with precision, recall, F1-score
- Confusion matrix with true vs predicted class names
- Baseline model (simpler Sequential CNN) for comparison

### 6. 🖼️ Visualization

- Misclassified images shown with true and predicted labels
- Loss curves smoothed using moving average for clarity

---

## 🧑‍🏫 For the Module Leader (Meeting Prep)

During the 15-minute meeting, you should be prepared to explain:

- Why you chose to avoid the Sequential model
- How SpatialDropout differs from standard Dropout
- Justifications for hyperparameters (e.g., learning rate decay, dropout, batch size)
- How data augmentation benefits generalization
- Why you added a baseline model for comparison
- What insights you gained from misclassification analysis

Consider walking through one Jupyter Notebook section at a time with visual support.

---

## 🧩 Next Steps

- Try transfer learning with a pretrained model (e.g., ResNet50)
- Apply this pipeline to a custom dataset (non-CIFAR)
- Expand the evaluation with more metrics (ROC-AUC, Cohen's Kappa)

---

## 📄 License

This project is for academic use. You are welcome to build on it, but please reference original ideas if reused.

---

## 📬 Contact

For questions, contact your module leader or the project contributor.
