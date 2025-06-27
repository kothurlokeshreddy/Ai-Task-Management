# 🧠 AI-Powered Task Management System

This project is an end-to-end AI-powered application that classifies tasks based on their descriptions, predicts their priority levels, and provides interactive dashboards for analysis and insights.

The goal is to help organizations or individuals **automatically understand task types, prioritize them**, and **balance workload across team members**, improving productivity and task clarity.

---

### 🔹 Step 1: Data Collection & Preprocessing
- Collected synthetic and real-world task data.
- Applied Exploratory Data Analysis (EDA).
- NLP preprocessing: tokenization, stopword removal, Lemmatization, etc.

### 🔹 Step 2: Feature Engineering & Classification
- Feature extraction with TF-IDF & Word Embeddings (Word2Vec, BERT).
- Built classification models using Naive Bayes & SVM.
- Evaluated using precision, recall, and accuracy.

### 🔹 Step 3: Advanced Modeling
- Implemented priority prediction using Random Forest and XGBoost.
- Applied GridSearchCV for hyperparameter tuning.
- Designed heuristic-based workload balancing among users.

### 🔹 Step 4: Finalization & Dashboard
- Finalized models and saved them.
- Developed a sleek and interactive dashboard using Streamlit.
- Added predictions, visualizations, and batch upload functionality.

---

## ⚙️ Tools & Libraries Used

- **Python**
- **Pandas, NumPy, Matplotlib, Seaborn**
- **Scikit-learn**
- **XGBoost, RandomForest Classifier**
- **NLTK, WordCloud**
- **Streamlit** for the interactive web dashboard
- **Joblib** for model serialization
- **GitHub** for version control

---

## 🚀 Project Structure

Here's the folder structure converted into Markdown format:
```
Ai-Task-Management/
│
│──📁 Python Notebooks/
│ └── app.py
│
├── 📁 app/
│ └── app.py # Main Streamlit dashboard
│
├── 📁 data/
│ ├── refined_dataset.csv # Final dataset
│ └── predicted_tasks.csv # Output from batch prediction
│
├── 📁 models/
│ ├── final_priority_xgboost_model.pkl
│ ├── final_task_type_classifier_svm.pkl
│ ├── task_type_tfidf_vectorizer.pkl
│ ├── status_label_encoder.pkl
│ ├── priority_label_encoder.pkl
│ └── task_type_label_encoder.pkl
│
├── 📁 reports/
│ ├── screenshots/
│ │ ├── dashboard_overview.png
│ │ ├── prediction_visuals.png
│ │ └── eda_insights.png
│ └── summary_report.md
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🖥️ How to Run the App

### 📦 1. Clone the Repository
```bash
git clone https://github.com/kothurlokeshreddy/Ai-Task-Management.git
cd task-prediction-dashboard
```

### 🧪 2. Install Required Libraries
```bash
pip install -r requirements.txt
```

### 🚀 3. Run the Streamlit App
```bash
streamlit run app/app.py
```
#### or
```bash
cd app
python -m streamlit run app.py
```

### Note: All trained models and encoders are already saved under /models. Make sure refined_dataset.csv exists in the data/ folder.

# 📊 Results Summary

---

## ✅ Task Type Classification

* **Model:** SVM (TF-IDF + Meta Features)
* **Accuracy:** ~100%
* **Reason:** Dataset was carefully structured with a strong correlation between descriptions and task types.

---

## 📌 Priority Prediction

* **Model:** XGBoost (with tuned hyperparameters)
* **Accuracy:** ~96.3%
* **Key Features:** Status, Estimated Hours, Actual Hours, Predicted Task Type

---

## ⚖️ Workload Balancing

* Implemented a heuristic algorithm to redistribute tasks for uniform load among team members.

---

## 📸 Screenshots

* Predictor View
* Visualizations
* EDA Page

---

## 📤 Future Improvements

* Add support for **BERT-based embeddings** for deeper semantic understanding.
* **Auto-update model training** on new data.
* Enable **integration with project management tools** like Trello/Jira.
