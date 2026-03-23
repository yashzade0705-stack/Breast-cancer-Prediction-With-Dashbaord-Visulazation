import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

st.set_page_config(page_title="Breast Cancer Prediction Dashboard", layout="centered")

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target
feature_names = data.feature_names

# ---------------------------------------------------
# Train Model
# ---------------------------------------------------
X = data.data
y = data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# ---------------------------------------------------
# Streamlit UI
# ---------------------------------------------------
st.markdown("# 🎗️ Breast Cancer Prediction")
st.caption("Interactive ML app with analytics & prediction")

# Patient Inputs
st.markdown("## 🔍 Patient Details")
st.caption("Adjust feature values for prediction")

col1, col2, col3 = st.columns(3)

user_input = []
for i, f in enumerate(feature_names):
    col = [col1, col2, col3][i % 3]
    with col:
        val = st.slider(
            f, float(df[f].min()), float(df[f].max()), float(df[f].mean()), key=f
        )
    user_input.append(val)

# Prediction
if st.button("🔮 Predict", type="primary"):
    inp = np.array(user_input).reshape(1, -1)
    inp_scaled = scaler.transform(inp)

    pred = model.predict(inp_scaled)[0]
    prob = model.predict_proba(inp_scaled)[0][1]

    result = "Benign (SAFE)" if pred == 1 else "Malignant (CANCER)"

    st.markdown("### 🩺 Result")
    if pred == 1:
        st.success(f"**{result}** - Probability: {prob:.2%}")
    else:
        st.error(f"**{result}** - Probability: {(1-prob):.2%}")

# Tabs Layout
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "📈 Viz",
    "🤖 Model",
    "📝 Data"
])

# ------------------- Dataset Overview -------------------
with tab1:
    st.markdown("### Dataset Summary")
    st.dataframe(df.describe(), use_container_width=True)

    st.markdown("### Class Distribution")
    st.bar_chart(df["target"].value_counts())

# ------------------- Visualizations -------------------
with tab2:
    st.markdown("### Analytics")

    col1, col2 = st.columns(2)

    # Histogram
    with col1:
        st.markdown("#### Histogram")
        feature = st.selectbox("Feature", feature_names, key="hist")
        fig, ax = plt.subplots(figsize=(4,3))
        sns.histplot(df[feature], kde=True, ax=ax)
        st.pyplot(fig)

    # Scatter Plot
    with col2:
        st.markdown("#### Scatter")
        f1 = st.selectbox("X", feature_names, key="x")
        f2 = st.selectbox("Y", feature_names, key="y")
        fig2, ax2 = plt.subplots(figsize=(4,3))
        sns.scatterplot(x=df[f1], y=df[f2], hue=df["target"], palette="Set1", ax=ax2)
        st.pyplot(fig2)

    # Heatmap
    st.markdown("#### Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(8,5))
    sns.heatmap(df.corr(), cmap="coolwarm", ax=ax3, cbar=False)
    st.pyplot(fig3)

# ------------------- Model Performance -------------------
with tab3:
    st.markdown("### Evaluation")

    # Accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.metric("Accuracy", f"{accuracy:.2%}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm, cbar=False)
        st.pyplot(fig_cm)

    with col2:
        st.markdown("#### ROC Curve")
        y_prob = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        fig_roc, ax_roc = plt.subplots(figsize=(4,3))
        ax_roc.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        ax_roc.plot([0,1], [0,1], linestyle="--", color="gray")
        ax_roc.set_xlabel("FPR")
        ax_roc.set_ylabel("TPR")
        ax_roc.legend()
        st.pyplot(fig_roc)

# ------------------- RAW DATA -------------------
with tab4:
    st.markdown("### Dataset")
    st.dataframe(df.head(100), use_container_width=True)