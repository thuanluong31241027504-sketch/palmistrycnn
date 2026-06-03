import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import time
import os

# Cấu hình trang
st.set_page_config(
    page_title="Palmistry AI",
    page_icon="🔮",
    layout="centered"
)

# Custom CSS (giữ nguyên của bạn)
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stApp {
        background-color: #0a0a0a;
    }
    
    * {
        font-family: 'Courier New', monospace;
    }
    
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0; }
    }
    
    .blinking-underscore {
        animation: blink 1s step-end infinite;
        display: inline-block;
        width: 8px;
        margin-left: 2px;
    }
    
    .static-underscore {
        display: inline-block;
        width: 8px;
        margin-left: 2px;
        opacity: 1;
    }
    
    .page-title {
        font-family: 'Courier New', monospace;
        font-size: 1.2rem;
        text-align: center;
        color: #c084fc;
        margin-top: 40px;
        margin-bottom: 40px;
    }
    
    .instruction {
        font-family: 'Courier New', monospace;
        font-size: 0.7rem;
        color: #8b5cf6;
        margin-top: 2rem;
        line-height: 1.8;
        display: inline-block;
        text-align: left;
    }
    
    .instruction-container {
        display: flex;
        justify-content: center;
        margin-top: 2rem;
    }
    
    .stButton > button {
        background: transparent;
        color: #c084fc !important;
        border: 1px solid #c084fc !important;
        border-radius: 0px !important;
        font-family: 'Courier New', monospace !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        background: #c084fc20 !important;
    }
    
    .result-box {
        border: 1px solid #c084fc;
        padding: 20px;
        margin-top: 20px;
        background-color: #0a0a0a;
    }
</style>
""", unsafe_allow_html=True)

# Load model TFLite
@st.cache_resource
def load_model():
    try:
        interpreter = tf.lite.Interpreter(model_path="palmistry.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except:
        return None

interpreter = load_model()

if interpreter is None:
    st.error("⚠️ model not found. Please upload palmistry.tflite")
    st.stop()

# Lấy thông tin model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
target_size = (input_shape[1], input_shape[2])
num_classes = output_details[0]['shape'][1]

# Class names (cập nhật theo model của bạn)
CLASS_NAMES = [f"Class_{i}" for i in range(num_classes)]

# ========== GIAO DIỆN CHÍNH (GIỮ NGUYÊN CỦA BẠN) ==========
st.markdown("""
<div class="page-title">
    > palmistry analysis
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    camera_image = st.camera_input("", label_visibility="collapsed")

if camera_image:
    image = Image.open(camera_image)
    st.image(image, caption="", width=250)
    
    if st.button("> analyze"):
        with st.spinner("processing..."):
            # Xử lý ảnh
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            img = image.resize(target_size)
            img_array = np.array(img).astype(np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Dự đoán
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])[0]
            
            idx = np.argmax(predictions)
            confidence = float(predictions[idx])
            
            st.markdown(f"""
            <div class="result-box">
                <h3 style="color:#c084fc; text-align:center;">> {CLASS_NAMES[idx]}</h3>
                <p style="color:#ffffff; text-align:center;">confidence: {confidence:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("<p style='color:#c084fc;'>top predictions:</p>", unsafe_allow_html=True)
            top3_idx = np.argsort(predictions)[-3:][::-1]
            for i, idx in enumerate(top3_idx, 1):
                prob = float(predictions[idx])
                st.progress(prob, text=f"{i}. {CLASS_NAMES[idx]} - {prob:.2%}")

# ========== HƯỚNG DẪN (GIỮ NGUYÊN CỦA BẠN) ==========
st.markdown("""
<div class="instruction-container">
    <div class="instruction">
        > chup anh long ban tay du anh sang, de trong khung hinh<span class="static-underscore">_</span><br>
        > nhan phan tich - app tu dong nhan dien<span class="static-underscore">_</span><br>
        > doc ket qua giai ma<span class="blinking-underscore">_</span>
    </div>
</div>
""", unsafe_allow_html=True)
