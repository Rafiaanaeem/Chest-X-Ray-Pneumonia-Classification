```markdown
## 🫁 Chest X-Ray Pneumonia Classification using Deep Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rafiaanaeem/Chest-X-Ray-Pneumonia-Classification/blob/main/main.ipynb)

## 📌 Overview
This project is a deep learning-based system for automated detection of pneumonia from chest X-ray images. It uses transfer learning with two pretrained Convolutional Neural Network (CNN) architectures and compares their diagnostic performance:
- **DenseNet-121**
- **EfficientNet-B0**

The system performs binary classification:
- `NORMAL`
- `PNEUMONIA`

---

##  Problem Statement
Chest X-rays are fundamental for pneumonia diagnosis, but manual interpretation can be time-consuming, subjective, and prone to human error. This project implements a transfer-learning pipeline to automatically classify X-ray scans, offering a fast and reliable decision-support tool.

---

##  Models Used

### DenseNet-121
- Pretrained on ImageNet
- Convolutional feature extractor layers frozen
- Custom final linear classifier trained for binary output

### EfficientNet-B0
- Lightweight architecture utilizing compound scaling
- Pretrained on ImageNet with frozen backbone
- Modified classification head trained for binary output

---

##  Dataset Structure
This project utilizes the [Chest X-Ray Pneumonia Balanced Dataset](https://www.kaggle.com/datasets/yusufmurtaza01/chest-xray-pneumonia-balanced-dataset) from Kaggle.

```text
dataset/
├── train/
│   ├── NORMAL
│   └── PNEUMONIA
├── val/
│   ├── NORMAL
│   └── PNEUMONIA
└── test/
    ├── NORMAL
    └── PNEUMONIA

```

---

##  Workflow

### 1. Data Preprocessing & Augmentation

* Resizing images to 224×224 pixels
* Normalization using ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`
* Training augmentations: Random crops (224x224), 10° random rotations, and horizontal flips

### 2. Model Training & Validation

* Loss Function: `CrossEntropyLoss`
* Optimizer: `Adam` (learning rate = 0.001)
* Performance evaluated per epoch using accuracy and loss tracking across training and validation sets

### 3. Evaluation Metrics

* Test Accuracy
* Weighted Precision, Recall, and F1-Score
* Classification Report
* Confusion Matrices for both Train and Test sets

---

##  How to Run in Google Colab

1. Click the **Open in Colab** badge at the top of this README.
2. In **Cell 2**, enter your own Kaggle credentials:
```python
os.environ['KAGGLE_USERNAME'] = "your_actual_username"
os.environ['KAGGLE_KEY'] = "your_actual_api_key"

```


3. Run all cells in sequential order (`Runtime > Run all`).
4. View trained model metrics, confusion matrices, or run the final cell to test predictions on custom single images.

---

##  Tools & Libraries

* **Language:** Python
* **Deep Learning Framework:** PyTorch, Torchvision
* **Data Handling & Metrics:** Scikit-learn, PIL
* **Visualization:** Matplotlib, Seaborn
* **Environment:** Google Colab (GPU accelerated)

---

##  Conclusion

This comparative study demonstrates that transfer learning with modern CNN architectures achieves high classification accuracy for pneumonia detection, offering scalable automated assistance for medical image analysis.

```

```