# Handwritten Digit Recognizer 🔢

A deep learning project that recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset, achieving **98.93% test accuracy**. Features a live web app built with Gradio.

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Hugging%20Face-orange)](https://huggingface.co/spaces/anshunigam/mnist-digit-recognizer)
[![GitHub](https://img.shields.io/badge/GitHub-anshunigam--trigger-black?logo=github)](https://github.com/anshunigam-trigger/mnist-digit-recognizer)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://tensorflow.org)

---

## 🚀 Live Demo

👉 **[Try it here](https://huggingface.co/spaces/anshunigam/mnist-digit-recognizer)** — Draw any digit and get instant predictions!

---

## 📊 Model Performance

| Model | Test Accuracy | Test Loss |
|-------|--------------|-----------|
| ANN (Baseline) | 97.66% | 0.0989 |
| **CNN (Final)** | **98.93%** | **0.0389** |

### Classification Report (CNN)

| Digit | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| 0 | 1.00 | 0.99 | 0.99 |
| 1 | 1.00 | 0.99 | 0.99 |
| 2 | 0.99 | 0.99 | 0.99 |
| 3 | 0.99 | 0.99 | 0.99 |
| 4 | 0.99 | 0.99 | 0.99 |
| 5 | 0.97 | 0.99 | 0.98 |
| 6 | 0.98 | 0.99 | 0.99 |
| 7 | 0.99 | 0.99 | 0.99 |
| 8 | 0.99 | 0.99 | 0.99 |
| 9 | 0.99 | 0.97 | 0.98 |
| **Overall** | **0.99** | **0.99** | **0.99** |

---

## 🏗️ Model Architecture

```
Input (28×28×1)
      ↓
Conv2D (32 filters, 3×3, ReLU)
      ↓
MaxPooling2D (2×2)
      ↓
Conv2D (64 filters, 3×3, ReLU)
      ↓
MaxPooling2D (2×2)
      ↓
Flatten → Dense (128, ReLU)
      ↓
Dense (10, Softmax) → Predicted Digit
```

**Total Parameters:** 225,034

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Deep Learning:** TensorFlow / Keras
- **Data Processing:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Web App:** Gradio
- **Deployment:** Hugging Face Spaces

---

## 📁 Project Structure

```
mnist-digit-recognizer/
├── mnist-CNN.ipynb          # Main notebook (EDA + ANN + CNN)
├── app.py                   # Gradio web app
├── mnist_cnn_model.keras    # Saved CNN model
├── requirements.txt         # Dependencies
└── README.md
```

---

## ⚙️ Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/anshunigam-trigger/mnist-digit-recognizer.git
cd mnist-digit-recognizer
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download dataset**

Download from [Kaggle MNIST CSV](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) and place `mnist_train.csv` and `mnist_test.csv` in the project folder.

**4. Run the notebook**
```bash
jupyter notebook mnist-CNN.ipynb
```

**5. Launch the web app**
```bash
python app.py
```

---

## 🔍 Key Findings

- CNN outperforms ANN by **1.27%** — reducing errors from 234 to 107 on 10,000 test images
- Digits **5** and **9** are the hardest to classify due to visual similarity with 6 and 4
- Experimented with **Batch Normalization** — did not improve performance on MNIST due to dataset simplicity, but shows real gains on deeper networks
- Mild **overfitting** observed in ANN (train: 99.81% vs val: 97.39%) — addressed by switching to CNN

---

## 👨‍💻 Author

**Anshu Nigam**
- B.Tech CSE (IoT) — IEM Kolkata, Batch 2024–2028
- GitHub: [@anshunigam-trigger](https://github.com/anshunigam-trigger)
- Hugging Face: [@anshunigam](https://huggingface.co/spaces/anshunigam/mnist-digit-recognizer)

---

## 📄 License

This project is licensed under the MIT License.
