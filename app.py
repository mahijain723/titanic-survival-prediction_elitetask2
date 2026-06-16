import streamlit as st
import pandas as pd
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.big-font {
    font-size: 48px !important;
    font-weight: 700;
    text-align: center;
}

.subtitle {
    text-align: center;
    color: gray;
    font-size: 20px;
}

.footer {
    text-align: center;
    color: gray;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    with open("titanic_model.sav", "rb") as f:
        return pickle.load(f)

model = load_model()

# ---------------- HEADER ----------------
st.markdown(
    "<div class='big-font'>🚢 Titanic Survival Prediction</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Predict passenger survival using Machine Learning</div>",
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------- MODEL METRICS ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="🤖 Model",
        value="Logistic Regression"
    )

with col2:
    st.metric(
        label="🎯 Accuracy",
        value="80%"
    )

with col3:
    st.metric(
        label="📊 Features",
        value="7"
    )

st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.header("📝 Passenger Information")

pclass = st.sidebar.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

sex = st.sidebar.radio(
    "Gender",
    ["Male", "Female"]
)

age = st.sidebar.slider(
    "Age",
    0,
    100,
    25
)

sibsp = st.sidebar.number_input(
    "Siblings / Spouses",
    min_value=0,
    max_value=10,
    value=0
)

parch = st.sidebar.number_input(
    "Parents / Children",
    min_value=0,
    max_value=10,
    value=0
)

fare = st.sidebar.number_input(
    "Fare",
    min_value=0.0,
    value=50.0
)

embarked = st.sidebar.selectbox(
    "Embarked Port",
    ["C", "Q", "S"]
)

# ---------------- ENCODING ----------------
sex_encoded = 1 if sex == "Male" else 0

embarked_map = {
    "C": 0,
    "Q": 1,
    "S": 2
}

embarked_encoded = embarked_map[embarked]

# ---------------- DASHBOARD ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Passenger Class", pclass)

with col2:
    st.metric("Age", age)

with col3:
    st.metric("Fare", f"${fare:.2f}")

st.markdown("---")

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Survival", use_container_width=True):

    input_data = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex_encoded],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked_encoded]
    })

    try:
        prediction = model.predict(input_data)[0]

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_data)[0][1] * 100
        else:
            probability = None

        st.markdown("---")

        if prediction == 1:
            st.success("🎉 Passenger is likely to SURVIVE!")
            st.info("The model predicts a higher chance of survival.")
        else:
            st.error("❌ Passenger is NOT likely to survive.")
            st.warning("The model predicts a lower chance of survival.")

        if probability is not None:
            st.metric(
                "🎯 Survival Probability",
                f"{probability:.2f}%"
            )

            st.progress(int(probability))

        with st.expander("📋 Passenger Data"):
            st.dataframe(input_data)

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# ---------------- TITANIC FACTS ----------------
st.markdown("---")

with st.expander("ℹ️ Titanic Facts"):
    st.write("""
    🚢 RMS Titanic sank on 15 April 1912.

    👥 More than 2200 passengers were onboard.

    ⚓ Over 1500 people lost their lives.

    👩 Women and children had higher survival rates.

    🎫 Passenger class significantly affected survival.
    """)

# ---------------- FOOTER ----------------
st.markdown("---")

st.markdown(
    """
    <div class='footer'>
        🚢 <b>Titanic Survival Prediction</b><br>
        Developed by <b>Mahi Jain</b><br>
        Machine Learning Internship Project
    </div>
    """,
    unsafe_allow_html=True
)
