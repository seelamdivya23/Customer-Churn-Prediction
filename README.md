# 🧠 Customer Churn Prediction using Deep Learning (ANN)

This project builds a deep learning model using an Artificial Neural Network (ANN) to predict customer churn for a bank. The model is trained on real-world data to identify customers likely to leave, helping businesses take proactive actions to retain them.

---

## 📂 Project Structure

````
customer-churn-prediction/
│
├── ccenv/                             # Virtual environment (ignored in Git)
├── logs/                              # Logs (for training/testing)
│
├── ann.ipynb                          # Model training notebook (ANN)
├── app.py                             # Deployment script (e.g., Flask or Streamlit)
├── prediction.ipynb                   # Prediction workflow or UI
│
├── Churn_Modelling.csv                # Dataset
├── model.h5                           # Saved ANN model
├── test_model.h5                      # Optional/test model
├── scaler.pkl                         # Saved scaler object
├── label_encoder_gender.pkl           # Encoder for gender
├── onehot_encoder_geo.pkl             # Encoder for geography
│
├── requirements.txt                   # Required Python packages
├── .gitignore                         # Ignored files/folders
└── README.md                          # Project documentation
````

---

## 📌 Problem Statement

**Customer churn** refers to when customers stop doing business with a company. Using historical customer data, we aim to build a model that predicts whether a customer will churn or not.

---

## 🛠️ Technologies Used

- **Python 3.9**
- **TensorFlow / Keras**
- **NumPy, Pandas, Scikit-learn**
- **Matplotlib / Seaborn**
- **Jupyter Notebook**
- **ANN (Artificial Neural Network)**

---

## 📊 Dataset

- Source: [Kaggle - Churn Modeling Dataset](https://www.kaggle.com/datasets)
- Contains customer information such as:
  - Credit Score
  - Geography
  - Gender
  - Age
  - Tenure
  - Balance
  - Number of Products
  - Has Credit Card
  - Is Active Member
  - Estimated Salary
  - Exited (Target variable)

---

## 🧠 Model Architecture

- Input Layer: 11 features
- Hidden Layers:
  - Dense(6), Activation = ReLU
  - Dense(6), Activation = ReLU
- Output Layer:
  - Dense(1), Activation = Sigmoid

---

## 📈 Evaluation Metrics

- Accuracy
- Confusion Matrix

---

## 🚀 How to Run

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/seelamdivya23/customer-churn-prediction.git
cd customer-churn-prediction
```
### 2️⃣ Create a Virtual Environment
```
python -m venv ccenv
ccenv\Scripts\activate  # On Windows
```
### 3️⃣ Install Dependencies
```
pip install -r requirements.txt
```
### 4️⃣ Train the Model
```
python churn_train.py
```
### 5️⃣ Predict New Data
```
python churn_predict.py
```
