import streamlit as st
import pandas as pd
import joblib

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.big-title{
    font-size:45px;
    font-weight:700;
    color:#FF4B4B;
    text-align:center;
}

.subtitle{
    text-align:center;
    color:#CFCFCF;
    font-size:18px;
    margin-bottom:30px;
}

div.stButton > button{
    width:100%;
    background: linear-gradient(90deg,#ff416c,#ff4b2b);
    color:white;
    border:none;
    border-radius:12px;
    height:55px;
    font-size:20px;
    font-weight:bold;
}

div.stButton > button:hover{
    transform:scale(1.03);
    transition:0.3s;
}

.card{
    padding:20px;
    border-radius:15px;
    background:#161B22;
    box-shadow:0px 0px 15px rgba(255,75,75,0.15);
}

</style>
""", unsafe_allow_html=True)

# ---------- LOAD FILES ----------
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# ---------- HEADER ----------
st.markdown('<p class="big-title">❤️ Heart Disease Prediction</p>',
unsafe_allow_html=True)

st.markdown(
'<p class="subtitle">AI Powered Heart Risk Detection System | Developed by Saif Khan</p>',
unsafe_allow_html=True)

# ---------- LAYOUT ----------
col1,col2 = st.columns(2)

with col1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("👤 Personal Details")

    age = st.slider("Age",18,100,40)

    sex = st.selectbox("Sex",["M","F"])

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA","NAP","TA","ASY"]
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure",
        80,200,120
    )

    cholesterol = st.number_input(
        "Cholesterol",
        100,600,200
    )

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar >120",
        [0,1]
    )

    st.markdown('</div>', unsafe_allow_html=True)

with col2:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("🩺 Clinical Details")

    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal","ST","LVH"]
    )

    max_hr = st.slider(
        "Maximum Heart Rate",
        60,220,150
    )

    exercise_angina = st.selectbox(
        "Exercise Angina",
        ["Y","N"]
    )

    oldpeak = st.slider(
        "Oldpeak",
        0.0,6.0,1.0
    )

    st_slope = st.selectbox(
        "ST Slope",
        ["Up","Flat","Down"]
    )

    st.markdown('</div>', unsafe_allow_html=True)

st.write("")

# ---------- PREDICTION ----------
if st.button("🔍 Predict Heart Risk"):

    raw_input = {col:0 for col in expected_columns}

    raw_input['Age'] = age
    raw_input['RestingBP'] = resting_bp
    raw_input['Cholesterol'] = cholesterol
    raw_input['FastingBS'] = fasting_bs
    raw_input['MaxHR'] = max_hr
    raw_input['Oldpeak'] = oldpeak

    raw_input['Sex_'+sex] = 1
    raw_input['ChestPainType_'+chest_pain] = 1
    raw_input['RestingECG_'+resting_ecg] = 1
    raw_input['ExerciseAngina_'+exercise_angina] = 1
    raw_input['ST_Slope_'+st_slope] = 1

    input_df = pd.DataFrame([raw_input])

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]

    probability = model.predict_proba(scaled_input)[0][1]

    st.divider()

    st.subheader("📊 Prediction Result")

    st.progress(float(probability))

    st.metric(
        label="Prediction Confidence",
        value=f"{probability:.2%}"
    )

    if prediction == 1:

        st.error("⚠️ HIGH RISK OF HEART DISEASE")

        st.markdown("""
        ### Recommendations
        ✔ Consult a cardiologist  
        ✔ Monitor blood pressure  
        ✔ Reduce cholesterol intake  
        ✔ Exercise regularly
        """)

    else:

        st.success("✅ LOW RISK OF HEART DISEASE")

        st.markdown("""
        ### Healthy Status
        ✔ Maintain healthy diet  
        ✔ Continue regular exercise  
        ✔ Annual medical checkups
        """)

st.write("")
st.caption("Built with Streamlit • Machine Learning • KNN Model")
