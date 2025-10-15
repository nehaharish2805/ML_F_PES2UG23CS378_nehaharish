# Fake News Detection Using Images with Deep Learning 🧠📰

## 📘 Project Overview
This project focuses on detecting **fake news images** using deep learning. With the increasing use of manipulated or misleading images in social media and online news, traditional text-based detection is insufficient.  
Our goal is to classify whether a news image is **real or fake** using **Convolutional Neural Networks (CNNs)** such as **ResNet50**, **ResNet18**, and **VGG16**.

## 🚀 Features
- Image-only fake news detection.
- Transfer learning with pre-trained CNN models.
- Supports evaluation metrics: accuracy, precision, recall, F1-score.
- Visual explanation with Grad-CAM (optional).
- Web-based interactive detector (can be deployed with Gradio/Streamlit).

## 🧩 Technologies Used
- **Python 3.10+**
- **TensorFlow / Keras**
- **NumPy, Matplotlib, scikit-learn**
- **Kaggle Dataset Integration**
- **Google Colab or Kaggle Notebook** environment

## 📂 Dataset
Dataset used: [Multilingual Fake News Images (BN + EN)](https://www.kaggle.com/datasets/evilspirit05/multilingual-fake-news-images-bn-en)

- Contains real and fake news images in multiple languages.
- Directory format:  
  ```
  train/
      fake/
      real/
  test/
      fake/
      real/
  ```

## ⚙️ How to Run

### Option 1 — Run on Kaggle Notebook
1. Go to [Kaggle Notebooks](https://www.kaggle.com/code).
2. Create a new notebook.
3. Click **Add Dataset** → search for “multilingual-fake-news-images-bn-en” and attach it.
4. Enable GPU from Notebook Settings.
5. Upload this notebook: **MINI_PROJECT_TEAM25_(2).ipynb**.
6. Run all cells sequentially.

### Option 2 — Run Locally
1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/fake-news-image-detector.git
   cd fake-news-image-detector
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset from Kaggle and place it inside `data/` folder.
4. Run the notebook in Jupyter or VSCode.

## 📊 Model Performance
Example output (ResNet18 model):
```
Found 722 images belonging to 2 classes.
Test Accuracy: 0.701 | Test Loss: 0.591

Confusion Matrix:
[[195 142]
 [ 74 311]]

Classification Report:
              precision    recall  f1-score   support
        Fake       0.72      0.58      0.64       337
        Real       0.69      0.81      0.74       385
```

## 🌐 Web App Demo (Optional)
An optional Gradio-based interface can be deployed:
```python
import gradio as gr
gr.Interface(fn=predict_image, inputs="image", outputs="label").launch()
```
This allows users to upload an image and instantly check whether it’s real or fake.

## 🔮 Future Improvements
- Combine image and text (multimodal fake news detection).
- Use advanced architectures (EfficientNet, Vision Transformers).
- Larger multilingual datasets.
- Explainability integration using Grad-CAM or LIME.

## 👩‍💻 Authors
**Team 25 — Fake News Detection Using Images**  
Department of Computer Science, [Your Institution Name]

---
📅 *Project Type:* Mini Project  
📚 *Domain:* Machine Learning / Deep Learning  
🧠 *Objective:* Identify fake news from image data to combat misinformation.
