import streamlit as st
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort
import os
import re
from collections import Counter

st.set_page_config(
    page_title="Palmistry AI - Phân Tích Chỉ Tay",
    page_icon="🔮",
    layout="wide"
)

# Custom CSS
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
        border-radius: 0px;
    }
    
    .result-title {
        color: #c084fc;
        font-size: 1rem;
        margin-bottom: 10px;
        font-weight: bold;
    }
    
    .result-content {
        color: #ffffff;
        font-size: 0.8rem;
        line-height: 1.5;
    }
    
    .probability-bar {
        background-color: #1a1a1a;
        border-radius: 0px;
        margin: 5px 0;
    }
    
    .probability-fill {
        background-color: #c084fc;
        padding: 4px;
        text-align: right;
        color: #0a0a0a;
        font-size: 0.7rem;
    }
    
    hr {
        border-color: #c084fc30;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== HÀM TRÍCH XUẤT ĐẶC TRƯNG ====================
def extract_image_features(img_array):
    """Trích xuất các đặc trưng từ ảnh để phân tích"""
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    brightness = np.mean(gray) / 255.0
    contrast = np.std(gray) / 255.0
    edges = cv2.Canny(gray, 30, 100)
    edge_density = np.mean(edges) / 255.0
    
    hue_mean = np.mean(hsv[:,:,0]) / 180.0
    sat_mean = np.mean(hsv[:,:,1]) / 255.0
    
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    texture = np.std(gray.astype(int) - blur.astype(int)) / 128.0
    texture = float(np.clip(texture, 0, 1))
    
    h, w = gray.shape
    center = gray[h//4:3*h//4, w//4:3*w//4]
    center_edge = np.mean(cv2.Canny(center, 30, 100)) / 255.0
    center_var = float(np.var(center) / (255.0**2))
    
    return {
        'brightness': brightness,
        'contrast': contrast,
        'edge_density': edge_density,
        'hue': hue_mean,
        'saturation': sat_mean,
        'texture': texture,
        'center_edge': center_edge,
        'center_var': center_var,
    }

# ==================== HÀM PHÂN LỚP ====================
def to_class(value, low, high):
    if value < low:
        return 0
    elif value < high:
        return 1
    return 2

def analyze_palm(features):
    """Phân tích 9 đặc tính từ đặc trưng ảnh"""
    # Ngưỡng mặc định (có thể điều chỉnh sau khi train model)
    thr = {
        'edge_density': {'low': 0.15, 'high': 0.35},
        'center_edge': {'low': 0.12, 'high': 0.30},
        'contrast': {'low': 0.20, 'high': 0.45},
        'texture': {'low': 0.10, 'high': 0.25},
        'brightness': {'low': 0.40, 'high': 0.70},
        'saturation': {'low': 0.30, 'high': 0.60},
        'center_var': {'low': 0.05, 'high': 0.15},
        'hue': {'low': 0.25, 'high': 0.50},
    }
    
    # Tính các chỉ số
    life_line = to_class(
        features['edge_density'] * 0.6 + features['center_edge'] * 0.4,
        thr['edge_density']['low'] * 0.6 + thr['center_edge']['low'] * 0.4,
        thr['edge_density']['high'] * 0.6 + thr['center_edge']['high'] * 0.4
    )
    
    heart_line = to_class(
        features['center_edge'] * 0.7 + features['contrast'] * 0.3,
        thr['center_edge']['low'] * 0.7 + thr['contrast']['low'] * 0.3,
        thr['center_edge']['high'] * 0.7 + thr['contrast']['high'] * 0.3
    )
    
    head_line = to_class(
        features['texture'] * 0.6 + features['edge_density'] * 0.4,
        thr['texture']['low'] * 0.6 + thr['edge_density']['low'] * 0.4,
        thr['texture']['high'] * 0.6 + thr['edge_density']['high'] * 0.4
    )
    
    huong_ngoai = to_class(
        features['brightness'] * 0.5 + features['saturation'] * 0.5,
        thr['brightness']['low'] * 0.5 + thr['saturation']['low'] * 0.5,
        thr['brightness']['high'] * 0.5 + thr['saturation']['high'] * 0.5
    )
    
    lanh_dao = to_class(
        features['contrast'] * 0.6 + features['center_var'] * 0.4,
        thr['contrast']['low'] * 0.6 + thr['center_var']['low'] * 0.4,
        thr['contrast']['high'] * 0.6 + thr['center_var']['high'] * 0.4
    )
    
    noi_tam = to_class(
        features['texture'] * 0.5 + features['hue'] * 0.5,
        thr['texture']['low'] * 0.5 + thr['hue']['low'] * 0.5,
        thr['texture']['high'] * 0.5 + thr['hue']['high'] * 0.5
    )
    
    sang_tao = to_class(
        features['center_edge'] * 0.4 + features['hue'] * 0.6,
        thr['center_edge']['low'] * 0.4 + thr['hue']['low'] * 0.6,
        thr['center_edge']['high'] * 0.4 + thr['hue']['high'] * 0.6
    )
    
    diem_tinh = to_class(
        1.0 - features['contrast'] * 0.5 - features['texture'] * 0.5,
        1.0 - thr['contrast']['high'] * 0.5 - thr['texture']['high'] * 0.5,
        1.0 - thr['contrast']['low'] * 0.5 - thr['texture']['low'] * 0.5
    )
    
    fortune_val = features['brightness'] * 0.5 + features['center_edge'] * 0.5
    fortune = to_class(
        fortune_val,
        thr['brightness']['low'] * 0.5 + thr['center_edge']['low'] * 0.5,
        thr['brightness']['high'] * 0.5 + thr['center_edge']['high'] * 0.5
    )
    
    return {
        'heart_line': heart_line,
        'head_line': head_line,
        'life_line': life_line,
        'huong_ngoai': huong_ngoai,
        'lanh_dao': lanh_dao,
        'noi_tam': noi_tam,
        'sang_tao': sang_tao,
        'diem_tinh': diem_tinh,
        'fortune_class': fortune
    }

# ==================== TEXT HIỂN THỊ ====================
PREDICTIONS_TEXT = {
    'heart_line': {
        0: ["▶ Đường Tâm Đạo của bạn khá mờ và ngắn", "Bạn có xu hướng giấu kín cảm xúc, ít khi thể hiện tình cảm ra bên ngoài. Cần nhiều thời gian để thực sự tin tưởng và mở lòng."],
        1: ["▶ Đường Tâm Đạo ở mức trung bình, rõ nét vừa phải", "Bạn biết cân bằng giữa lý trí và tình cảm. Hòa đồng, thân thiện nhưng vẫn giữ được giới hạn cần thiết."],
        2: ["▶ Đường Tâm Đạo của bạn rất sâu và rõ nét", "Bạn là người cực kỳ giàu tình cảm, nồng nhiệt và chân thành. Luôn hết mình vì người mình yêu thương."]
    },
    'head_line': {
        0: ["▶ Đường Trí Đạo của bạn khá ngắn hoặc mờ", "Bạn làm việc thiên về trực giác và bản năng hơn logic. Thích những gì đơn giản, đi thẳng vào vấn đề."],
        1: ["▶ Đường Trí Đạo ở mức trung bình và khá liền mạch", "Bạn có lối tư duy thực tế, suy nghĩ thấu đáo trước khi làm. Khả năng tiếp thu và giải quyết vấn đề tốt."],
        2: ["▶ Đường Trí Đạo của bạn rất sâu, rõ và dài", "Bạn sở hữu trí tuệ sắc sảo, tư duy phân tích logic tuyệt vời. Có góc nhìn cực kỳ độc đáo."]
    },
    'life_line': {
        0: ["▶ Đường Sinh Đạo của bạn có phần mờ và ngắn", "Thể trạng nhạy cảm với sự thay đổi. Cần chú ý nghỉ ngơi và tránh làm việc quá sức."],
        1: ["▶ Đường Sinh Đạo hiện lên rõ ràng ở mức trung bình", "Sức khỏe và sinh lực khá tốt và ổn định. Có khả năng phục hồi nhanh sau mệt mỏi."],
        2: ["▶ Đường Sinh Đạo của bạn rất sâu, dài và liền mạch", "Bạn sở hữu nguồn sinh lực dồi dào và sức đề kháng tuyệt vời. Luôn tràn đầy năng lượng."]
    },
    'huong_ngoai': {
        0: ["▶ Hướng Ngoại: Mức độ thấp", "Bạn mang đậm nét hướng nội, tìm thấy năng lượng khi ở một mình. Tỏa sáng trong các công việc độc lập, cần sự tập trung."],
        1: ["▶ Hướng Ngoại: Mức độ trung bình", "Bạn là Ambivert chính hiệu. Linh hoạt trong giao tiếp, thích nghi tốt với nhiều môi trường khác nhau."],
        2: ["▶ Hướng Ngoại: Mức độ cao", "Bạn là tâm điểm của sự chú ý, tràn đầy năng lượng khi giao tiếp. Khả năng ăn nói lưu loát, tự tin."]
    },
    'lanh_dao': {
        0: ["▶ Lãnh Đạo: Mức độ thấp", "Bạn thích làm người hỗ trợ hơn là người đứng đầu. Cảm thấy thoải mái khi được giao nhiệm vụ cụ thể."],
        1: ["▶ Lãnh Đạo: Mức độ trung bình", "Bạn có tiềm năng lãnh đạo khá. Trong tình huống cần thiết, có thể đứng lên dẫn dắt đội nhóm."],
        2: ["▶ Lãnh Đạo: Mức độ cao", "Tố chất Lãnh đạo cực kỳ mạnh mẽ! Bạn có tầm nhìn xa, sự quyết đoán và khí chất thu hút người khác."]
    },
    'noi_tam': {
        0: ["▶ Nội Tâm: Mức độ thấp", "Bạn sống hướng ra bên ngoài, có gì nói đó. Ít khi tự dằn vặt hay suy nghĩ quá nhiều về quá khứ."],
        1: ["▶ Nội Tâm: Mức độ trung bình", "Đời sống nội tâm phong phú và cân bằng. Biết cách tự phản tỉnh nhưng không bị mắc kẹt trong suy nghĩ tiêu cực."],
        2: ["▶ Nội Tâm: Mức độ cao", "Đời sống nội tâm cực kỳ sâu sắc! Bạn nhạy cảm, hay suy nghĩ và có những chiêm nghiệm triết lý về cuộc đời."]
    },
    'sang_tao': {
        0: ["▶ Sáng Tạo: Mức độ cơ bản", "Bạn tuân thủ nguyên tắc, thích làm việc theo quy trình. Giỏi duy trì và tối ưu hệ thống."],
        1: ["▶ Sáng Tạo: Mức độ tốt", "Bạn có thể đưa ra giải pháp cải tiến hiệu quả. Biết áp dụng sáng tạo vào đúng thời điểm."],
        2: ["▶ Sáng Tạo: Mức độ cao", "Tư duy Sáng tạo bùng nổ! Bạn luôn nhìn thế giới qua lăng kính khác biệt, ý tưởng độc đáo và mới lạ."]
    },
    'diem_tinh': {
        0: ["▶ Điềm Tĩnh: Mức độ thấp", "Bạn dễ xúc động và phản ứng mạnh trước tình huống bất ngờ. Cần học cách kiểm soát cảm xúc."],
        1: ["▶ Điềm Tĩnh: Mức độ khá", "Trong phần lớn tình huống, bạn giữ được thái độ hòa nhã, bình tĩnh. Biết cách kiểm soát cảm xúc tốt."],
        2: ["▶ Điềm Tĩnh: Mức độ cao", "Bạn sở hữu sự Điềm tĩnh đến kinh ngạc! Khả năng kiểm soát cảm xúc và giữ bình tĩnh dưới áp lực là vũ khí lợi hại."]
    },
    'fortune_class': {
        0: ["▶ Vận May: Mức độ Bình thường", "Hãy kiên nhẫn và tiếp tục nỗ lực. Cơ hội tốt sẽ đến khi bạn sẵn sàng."],
        1: ["▶ Vận May: Mức độ Tốt", "Đây là thời điểm thuận lợi để thực hiện các kế hoạch. Hãy nắm bắt cơ hội để tiến lên."],
        2: ["▶ Vận May: Mức độ Rất Tốt", "Chúc mừng! Mọi việc đều hanh thông và có nhiều cơ hội lớn đang chờ đón bạn."]
    }
}

# Tên hiển thị
DISPLAY_NAMES = {
    'heart_line': '💖 Đường Tâm Đạo',
    'head_line': '🧠 Đường Trí Đạo',
    'life_line': '🌿 Đường Sinh Đạo',
    'huong_ngoai': '🌐 Hướng Ngoại',
    'lanh_dao': '⚡ Lãnh Đạo',
    'noi_tam': '🎭 Nội Tâm',
    'sang_tao': '✨ Sáng Tạo',
    'diem_tinh': '☯️ Điềm Tĩnh',
    'fortune_class': '🍀 Vận May'
}

# ==================== GIAO DIỆN CHÍNH ====================
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
    st.image(image, caption="", width=280)
    
    if st.button("> analyze"):
        with st.spinner("Đang phân tích..."):
            # Xử lý ảnh
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            img_array = np.array(image)
            
            # Trích xuất đặc trưng
            features = extract_image_features(img_array)
            
            # Phân tích
            results = analyze_palm(features)
            
            # Hiển thị kết quả
            st.markdown("---")
            st.markdown('<p style="color:#c084fc; text-align:center; font-size:1rem;">📜 KẾT QUẢ PHÂN TÍCH</p>', unsafe_allow_html=True)
            
            # Hiển thị 3 đường chỉ tay chính
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                val = results['heart_line']
                st.markdown(f"""
                <div class="result-box">
                    <div class="result-title">{DISPLAY_NAMES['heart_line']}</div>
                    <div class="result-content">{PREDICTIONS_TEXT['heart_line'][val][0]}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                val = results['head_line']
                st.markdown(f"""
                <div class="result-box">
                    <div class="result-title">{DISPLAY_NAMES['head_line']}</div>
                    <div class="result-content">{PREDICTIONS_TEXT['head_line'][val][0]}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_c:
                val = results['life_line']
                st.markdown(f"""
                <div class="result-box">
                    <div class="result-title">{DISPLAY_NAMES['life_line']}</div>
                    <div class="result-content">{PREDICTIONS_TEXT['life_line'][val][0]}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Hiển thị 5 tính cách
            st.markdown("---")
            st.markdown('<p style="color:#c084fc; text-align:center; font-size:1rem;">🎭 PHÂN TÍCH TÍNH CÁCH</p>', unsafe_allow_html=True)
            
            personality_keys = ['huong_ngoai', 'lanh_dao', 'noi_tam', 'sang_tao', 'diem_tinh']
            
            for key in personality_keys:
                val = results[key]
                st.markdown(f"""
                <div class="result-box">
                    <div class="result-title">{DISPLAY_NAMES[key]}</div>
                    <div class="result-content">{PREDICTIONS_TEXT[key][val][0]}</div>
                    <div class="result-content" style="margin-top:8px; color:#c084fc80;">{PREDICTIONS_TEXT[key][val][1]}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Vận may
            st.markdown("---")
            val = results['fortune_class']
            st.markdown(f"""
            <div class="result-box">
                <div class="result-title">{DISPLAY_NAMES['fortune_class']}</div>
                <div class="result-content">{PREDICTIONS_TEXT['fortune_class'][val][0]}</div>
                <div class="result-content" style="margin-top:8px; color:#c084fc80;">{PREDICTIONS_TEXT['fortune_class'][val][1]}</div>
            </div>
            """, unsafe_allow_html=True)

# Hướng dẫn
st.markdown("""
<div class="instruction-container">
    <div class="instruction">
        > chup anh long ban tay du anh sang, de trong khung hinh<span class="static-underscore">_</span><br>
        > nhan phan tich - app tu dong phan tich 9 chi so<span class="static-underscore">_</span><br>
        > doc ket qua giai ma ve tinh cach va van menh<span class="blinking-underscore">_</span>
    </div>
</div>
""", unsafe_allow_html=True)
