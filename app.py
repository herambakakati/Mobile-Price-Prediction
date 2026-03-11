import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------------------------------------------------

# Page configuration

# ---------------------------------------------------

st.set_page_config(
    page_title="Mobile Price Predictor",
    page_icon="📱",
    layout="centered"
)

# ---------------------------------------------------

# Custom UI Styling

# ---------------------------------------------------

st.markdown("""

<style>

[data-testid="stAppViewContainer"]{
background-image: url("https://images.unsplash.com/photo-1511707171634-5f897ff02aa9");
background-size: cover;
background-position: center;
}

[data-testid="stHeader"]{
background: rgba(0,0,0,0);
}

.block-container{
background-color: rgba(0,0,0,0.65);
padding: 2rem;
border-radius: 15px;
color: white;
}

/* Text color */
h1,h2,h3,label{
color:white !important;
}

/* Input fields */
div[data-baseweb="input"] > div{
background-color:#ffffff;
border-radius:8px;
}

div[data-baseweb="input"] input{
color:#000000;
font-weight:500;
}

/* Button styling */
.stButton>button{
background: linear-gradient(90deg,#00C9A7,#007CF0);
color:white;
font-size:18px;
border-radius:10px;
height:3em;
width:100%;
border:none;
}

.stButton>button:hover{
background: linear-gradient(90deg,#007CF0,#00C9A7);
}

</style>

""", unsafe_allow_html=True)

# ---------------------------------------------------

# Load trained model pipeline

# ---------------------------------------------------

model = pickle.load(open("mobile_price_pipeline.pkl","rb"))

# ---------------------------------------------------

# Title

# ---------------------------------------------------

st.title("📱 Mobile Price Prediction")
st.write("Enter smartphone specifications to estimate the expected price")

# ---------------------------------------------------

# Input layout

# ---------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    Sale = st.number_input("Sale Count", min_value=0.0, value=100.0)
    weight = st.number_input("Weight (grams)", min_value=50.0, value=150.0)
    resolution = st.number_input("Screen Resolution", min_value=1.0, value=5.0)
    ppi = st.number_input("PPI (Pixel Density)", min_value=100.0, value=300.0)
    cpu_core = st.number_input("CPU Cores", min_value=1, max_value=12, value=4)
    cpu_freq = st.number_input("CPU Frequency (GHz)", min_value=0.5, value=1.5)

with col2:
    internal_mem = st.number_input("Internal Memory (GB)", min_value=1.0, value=32.0)
    ram = st.number_input("RAM (GB)", min_value=1.0, value=4.0)
    RearCam = st.number_input("Rear Camera (MP)", min_value=1.0, value=12.0)
    Front_Cam = st.number_input("Front Camera (MP)", min_value=0.0, value=5.0)
    battery = st.number_input("Battery Capacity (mAh)", min_value=500.0, value=3000.0)
    thickness = st.number_input("Thickness (mm)", min_value=5.0, value=8.0)

# ---------------------------------------------------

# Prediction

# ---------------------------------------------------

if st.button("Predict Mobile Price"):
    input_df = pd.DataFrame({
        "Sale":[np.log1p(Sale)],
        "weight":[np.log1p(weight)],
        "resoloution":[resolution],   # spelling matches training data
        "ppi":[ppi],
        "cpu core":[cpu_core],
        "cpu freq":[cpu_freq],
        "internal mem":[np.log1p(internal_mem)],
        "ram":[ram],
        "RearCam":[RearCam],
        "Front_Cam":[Front_Cam],
        "battery":[np.log1p(battery)],
        "thickness":[np.log1p(thickness)]
    })
    
    prediction = model.predict(input_df)[0]

    st.success(f"💰 Estimated Mobile Price: ₹ {round(prediction,2)}")

    st.balloons()

# ---------------------------------------------------

# Developer Credit

# ---------------------------------------------------
st.markdown(
""" <div style='text-align:center; margin-top:30px;'> <p style='color:#d3d3d3; font-size:15px; opacity:0.8;'>
© 2026 Machine Learning Project | Developed by Heramba Kakati </p> </div>
""",
unsafe_allow_html=True
)
