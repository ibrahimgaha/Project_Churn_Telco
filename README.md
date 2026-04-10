# 📘 MLA Project – Churn Prediction Platform (MLOps v2.0)

This project has been transformed into a professional **ML Experimentation Platform**. It allows machine learning engineers and researchers to train, tune, compare, and visualize multiple classification models for Telco Churn prediction, with full experiment tracking via **MLflow**.

---

## 🚀 Key Features Added
- **5 Classification Algorithms**: Support for Logistic Regression, KNN, SVM, Decision Tree, and Random Forest.
- **Automated Hyperparameter Tuning**: Integrated `GridSearchCV` to find optimal model parameters automatically.
- **MLflow Tracking**: Every training run logs parameters, metrics (Accuracy, F1, Recall, etc.), and the model artifact.
- **Best Model Auto-Identification**: The system identifies the best-performing model based on F1-score.
- **Exploratory Visualization**: PCA and t-SNE 2D projections to see how well the data segments into churn vs. non-churn.
- **Feature Importance**: Visual charts showing which customer attributes (tenure, charges, etc.) affect the predictions most.
- **Premium UI**: A modern, modular React dashboard with real-time feedback and glassmorphism design.

---

## ▶️ How to Run the Project

### 1. Install Dependencies
Open your terminal and run:

```bash
# Install backend dependencies
cd backend
pip install -r requirements.txt

# Install frontend dependencies
cd ../frontend
npm install
```

### 2. Run the Backend (FastAPI)
From the `backend/` directory:
```bash
uvicorn main:app --reload
```
The API will be available at `http://localhost:8000`.

### 3. Run the Frontend (React/Vite)
From the `frontend/` directory:
```bash
npm run dev
```
The UI will be available at `http://localhost:5173`.

### 4. Run MLflow UI
To see all your logs and compare runs in detail:
From the `backend/` directory:
```bash
mlflow ui
```
Open `http://localhost:5000` in your browser.

---

## 📊 How to Use It

1. **Training Tab**: Select the models you want to compare. You can manually adjust hyperparameters or use the **"Auto Tune"** button to find the best configuration automatically.
2. **Performance Tab**: Compare models side-by-side using Accuracy, Precision, Recall, F1-score, and ROC-AUC.
3. **Exploration Tab**: Run PCA or t-SNE to see a 2D map of your customer segments.
4. **Inference Tab**: Upload a CSV file and a **Run ID** from the history to generate churn predictions for new customers.
5. **Run History**: View all past experiments and copy Run IDs for prediction.

---

## 🧠 Simple Explanation for Students

### 🏆 Which Model is Best?
In our testing, **Random Forest** often performs best because it combines multiple decision trees to reduce error and handle non-linear relationships. 

For churn prediction, we prioritize the **F1-Score** or **Recall** over simple Accuracy. 
*Why?* Because a customer "No Churn" prediction for someone who actually churns (False Negative) is very expensive for a company.

### ⚙️ What are Hyperparameters?
Think of them as the "settings" of the algorithm.
- **KNN (k)**: A small `k` makes the model sensitive to noise; a large `k` might make it too "generalized".
- **Random Forest (n_estimators)**: More trees usually improve performance but make the model slower to train.
- **Logistic Regression (C)**: Controls regularization. A smaller `C` prevents the model from overreacting to specific data points (overfitting).

### 📈 What is MLflow doing?
MLflow acts like a **scientist's notebook**. Every time you click "Train", it records exactly which settings you used and what the result was. This ensures your experiments are repeatable and organized.
