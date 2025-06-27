# streamlit_app.py
import matplotlib as mpl
mpl.rcParams.update({'text.color': 'white', 'axes.labelcolor': 'white', 'xtick.color': 'white', 'ytick.color': 'white'})
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import time
from sklearn.utils.extmath import softmax

# === Load Models & Preprocessors ===
task_model = joblib.load("./models/final_task_type_classifier_svm.pkl")
priority_model = joblib.load("./models/final_priority_xgboost_model.pkl")

status_encoder = joblib.load("./models/status_label_encoder.pkl")
priority_encoder = joblib.load("./models/priority_label_encoder.pkl")
task_type_encoder = joblib.load("./models/task_type_label_encoder.pkl")
tfidf_vectorizer = joblib.load("./models/task_type_tfidf_vectorizer.pkl")

# === Load Dataset for EDA Page ===
@st.cache_data
def load_data():
    return pd.read_csv("./data/refined_dataset.csv")
data = load_data()

# === Custom Styling ===
st.set_page_config(page_title="AI Task Dashboard", layout="wide")
st.markdown("""
    <style>
    .stApp {
        background-color: #1E1E2F;
        color: white;
    }
    .result-box {
        background-color: #2E2E3E;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        margin-top: 20px;
        margin-bottom: 20px;
    }
    h1, h2, h3 {
        color: #FFA726;
    }
    .css-18e3th9 {
        background-color: #2B2B3D !important;
    }
    .st-c6 {
        background-color: #2B2B3D !important;
    }
    </style>
""", unsafe_allow_html=True)

dark_palette = "rocket" # seaborn dark style

# === Navigation Menu ===
with st.sidebar:
    selected = option_menu("AI Task Dashboard", ["Predictor", "EDA & Insights", "Batch Prediction"],
                        icons=['activity', 'bar-chart', 'upload'],
                        menu_icon="layers", default_index=0)

# === Page 1: Task & Priority Predictor ===
if selected == "Predictor":
    st.title("🔮 Task Type & Priority Prediction")
    with st.form("prediction_form"):
        description = st.text_area("Task Description")
        # Ensure the status options match your encoder's training data if 'Done' and 'Blocked' are new
        status = st.selectbox("Status", ["To Do", "In Progress", "Done", "Blocked"]) # Added 'Done', 'Blocked'
        est_hours = st.slider("Estimated Hours", 1, 20, 5)
        act_hours = st.slider("Actual Hours", 1, 20, 4)
        submitted = st.form_submit_button("Predict")

    if submitted:
        if description.strip() == "":
            st.warning("Please enter a task description to proceed.")
        else:
            with st.spinner("Making predictions... please wait"):
                time.sleep(1)  # Optional: simulate model latency for UX
            # --- Preprocess Inputs for Task Type Prediction ---
            desc_vector = tfidf_vectorizer.transform([description])
            status_encoded = status_encoder.transform([status])[0]
            numeric_features_task_type = np.array([[status_encoded, est_hours, act_hours]])
            combined_for_task = hstack([desc_vector, numeric_features_task_type])

            # === Perform Task Type Prediction ===
            task_pred_encoded = task_model.predict(combined_for_task)[0]
            task_pred_label = task_type_encoder.inverse_transform([task_pred_encoded])[0]

            # --- Preprocess Inputs for Priority Prediction (including predicted task type) ---
            # Your key change: include the encoded predicted task type
            numeric_features_priority = np.array([[status_encoded, est_hours, act_hours, task_pred_encoded]])

            # === Perform Priority Prediction ===
            priority_pred_encoded = priority_model.predict(numeric_features_priority)[0]
            priority_pred_label = priority_encoder.inverse_transform([priority_pred_encoded])[0]

            st.markdown(f"""
            <div class='result-box'>
                <h3>🎯 Prediction Results</h3>
                <p><strong>Predicted Task Type:</strong> {task_pred_label}</p>
                <p><strong>Predicted Priority:</strong> {priority_pred_label}</p>
            </div>
            """, unsafe_allow_html=True)

            # === Visualizations After Prediction ===
            st.subheader("📈 Visual Insights")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Estimated vs Actual Hours**")
                fig4, ax4 = plt.subplots(facecolor='#1E1E2F')
                ax4.set_facecolor('#1E1E2F')
                sns.barplot(x=["Estimated", "Actual"], y=[est_hours, act_hours], palette=dark_palette, ax=ax4)
                st.pyplot(fig4)

            with col2:
                st.markdown("**WordCloud from Input Description**")
                wc_input = WordCloud(width=600, height=300, background_color='black', colormap='plasma').generate(description)
                fig5, ax5 = plt.subplots(facecolor='#1E1E2F')
                ax5.set_facecolor('#1E1E2F')
                ax5.imshow(wc_input, interpolation='bilinear')
                ax5.axis('off')
                st.pyplot(fig5)

            st.markdown("**📊 Similar Task Type Distribution**")
            similar_df = data[data['task_type'] == task_pred_label] # Use the decoded label for filtering
            fig6, ax6 = plt.subplots(facecolor='#1E1E2F')
            ax6.set_facecolor('#1E1E2F')
            sns.histplot(similar_df['estimated_hours'], bins=20, kde=True, ax=ax6, palette=dark_palette)
            ax6.set_title(f"Estimated Hours Distribution for '{task_pred_label}' Tasks")
            st.pyplot(fig6)

# === Page 2: EDA ===
elif selected == "EDA & Insights":
    st.title("📊 EDA & Visual Insights")
    st.subheader("📌 Distribution of Task Type")
    fig1, ax1 = plt.subplots(facecolor='#1E1E2F')
    ax1.set_facecolor('#1E1E2F')
    sns.countplot(data=data, x='task_type', order=data['task_type'].value_counts().index, palette=dark_palette, ax=ax1)
    st.pyplot(fig1)

    st.subheader("⏱️ Priority by Estimated Hours")
    fig2, ax2 = plt.subplots(facecolor='#1E1E2F')
    ax2.set_facecolor('#1E1E2F')
    sns.boxplot(data=data, x='priority', y='estimated_hours', palette=dark_palette, ax=ax2)
    st.pyplot(fig2)

    st.subheader("☁️ WordCloud for Task Descriptions")
    text = ' '.join(data['task_description'].dropna().tolist())
    wc = WordCloud(width=600, height=300, background_color='black', colormap="plasma").generate(text)
    fig3, ax3 = plt.subplots(facecolor='#1E1E2F')
    ax2.set_facecolor('#1E1E2F')
    ax3.imshow(wc, interpolation='bilinear')
    ax3.axis('off')
    st.pyplot(fig3)

# === Page 3: Batch Prediction ===
elif selected == "Batch Prediction":
    st.title("📁 Batch Prediction on Uploaded File")
    file = st.file_uploader("Upload CSV File", type=["csv"])

    if file:
        df = pd.read_csv(file)
        required_cols = ['task_description', 'status', 'estimated_hours', 'actual_hours']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Error: The uploaded CSV must contain the columns: {', '.join(required_cols)}")
        else:
            try:
                # Preprocess for Task Type Prediction
                desc_vec = tfidf_vectorizer.transform(df['task_description'])
                status_enc = status_encoder.transform(df['status'])
                meta_for_task = np.column_stack((status_enc, df['estimated_hours'], df['actual_hours']))
                combined_for_task = hstack([desc_vec, meta_for_task])

                # Predict Task Type
                task_preds_encoded = task_model.predict(combined_for_task)
                df['predicted_task_type_encoded'] = task_preds_encoded # Store encoded for priority prediction
                df['predicted_task_type'] = task_type_encoder.inverse_transform(task_preds_encoded)

                # Preprocess for Priority Prediction (including the newly predicted task type)
                meta_for_priority = np.column_stack((status_enc, df['estimated_hours'], df['actual_hours'], df['predicted_task_type_encoded']))

                # Predict Priority
                priority_preds_encoded = priority_model.predict(meta_for_priority)
                df['predicted_priority'] = priority_encoder.inverse_transform(priority_preds_encoded)

                # Drop the temporary encoded column if not needed in final output
                df = df.drop(columns=['predicted_task_type_encoded'])

                st.dataframe(df.head())
                st.download_button("Download Predictions", df.to_csv(index=False), file_name="predicted_tasks.csv")

            except Exception as e:
                st.error(f"Prediction failed: {e}. Please ensure data types and values are correct. You might need to add 'Done' and 'Blocked' to your status encoder if they are new categories.")
