import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🍄 Mushroom Classification App")

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("mushrooms.csv")
    label_encoders = {}
    for column in data.columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    return data

data = load_data()
X = data.drop("class", axis=1)
y = data["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

st.sidebar.title("Model Selection")
classifier = st.sidebar.selectbox("Choose Classifier", ("SVM", "Logistic Regression", "Random Forest"))

def plot_metrics(y_test, y_pred):
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    st.pyplot(plt)

    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Precision:", precision_score(y_test, y_pred))
    st.write("Recall:", recall_score(y_test, y_pred))

if classifier == "SVM":
    C = st.sidebar.slider("C (Regularization)", 0.01, 10.0)
    kernel = st.sidebar.selectbox("Kernel", ("linear", "rbf", "poly", "sigmoid"))
    model = SVC(C=C, kernel=kernel)
elif classifier == "Logistic Regression":
    C = st.sidebar.slider("C (Regularization)", 0.01, 10.0)
    model = LogisticRegression(C=C, max_iter=1000)
else:
    n_estimators = st.sidebar.slider("Number of trees", 10, 200)
    model = RandomForestClassifier(n_estimators=n_estimators)

if st.sidebar.button("Train Model"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    plot_metrics(y_test, y_pred)
