import streamlit as st
import numpy as np
from PIL import Image, ImageFilter
import time

st.set_page_config(
    page_title="Palmistry Analysis",
    page_icon="",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stApp {
        background-color: #1a0b2e;
    }
    
    * {
        font-family: 'Courier New', 'SF Mono', monospace;
    }
    
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }
    
    @keyframes neon {
        0% { text-shadow: 0 0 5px #ff00ff, 0 0 10px #ff00ff, 0 0 20px #ff00ff; color: #ff00ff; }
        25% { text-shadow: 0 0 5px #00ffff, 0 0 10px #00ffff, 0 0 20px #00ffff; color: #00ffff; }
        50% { text-shadow: 0 0 5px #ff6600, 0 0 10px #ff6600, 0 0 20px #ff6600; color: #ff6600; }
        75% { text-shadow: 0 0 5px #ffff00, 0 0 10px #ffff00, 0 0 20px #ffff00; color: #ffff00; }
        100% { text-shadow: 0 0 5px #ff00ff, 0 0 10px #ff00ff, 0 0 20px #ff00ff; color: #ff00ff; }
    }
    
    @keyframes neonText {
        0% { text-shadow: 0 0 3px #ff00ff, 0 0 6px #ff00ff; color: #ff00ff; }
        33% { text-shadow: 0 0 3px #00ffff, 0 0 6px #00ffff; color: #00ffff; }
        66% { text-shadow: 0 0 3px #ff6600, 0 0 6px #ff6600; color: #ff6600; }
        100% { text-shadow: 0 0 3px #ff00ff, 0 0 6px #ff00ff; color: #ff00ff; }
    }
    
    @keyframes rainbow-line {
        0% { background-color: #ff00ff; box-shadow: 0 0 5px #ff00ff, 0 0 10px #ff00ff; }
        20% { background-color: #00ffff; box-shadow: 0 0 5px #00ffff, 0 0 10px #00ffff; }
        40% { background-color: #ff6600; box-shadow: 0 0 5px #ff6600, 0 0 10px #ff6600; }
        60% { background-color: #ffff00; box-shadow: 0 0 5px #ffff00, 0 0 10px #ffff00; }
        80% { background-color: #00ff00; box-shadow: 0 0 5px #00ff00, 0 0 10px #00ff00; }
        100% { background-color: #ff00ff; box-shadow: 0 0 5px #ff00ff, 0 0 10px #ff00ff; }
    }
    
    .blinking-cursor {
        animation: blink 1s step-end infinite;
        display: inline-block;
        width: 10px;
    }
    
    .main-title {
        font-family: 'Courier New', monospace;
        font-size: 2.5rem;
        margin-bottom: 1rem;
        font-weight: normal;
        animation: neon 3s infinite;
    }
    
    .rainbow-hr {
        height: 2px;
        width: 100%;
        margin: 20px 0;
        animation: rainbow-line 1s infinite;
        border: none;
    }
    
    .instruction {
        font-family: 'Courier New', monospace;
        font-size: 0.8rem;
        line-height: 2;
        text-align: left;
    }
    
    .instruction-line {
        animation: neonText 3s infinite;
        margin: 5px 0;
    }
    
    .stButton > button {
        background-color: #2d1b4e !important;
        color: #00ffff !important;
        border: 1px solid #00ffff !important;
        border-radius: 0px !important;
        font-family: 'Courier New', monospace !important;
        font-weight: bold !important;
        padding: 0.5rem 1rem !important;
        width: 100% !important;
        box-shadow: 0 0 5px #00ffff;
    }
    
    .stButton > button:hover {
        background-color: #3d2b5e !important;
        color: #ff00ff !important;
        border: 1px solid #ff00ff !important;
        box-shadow: 0 0 10px #ff00ff;
    }
    
    .result-box {
        border: 1px solid #ff00ff;
        padding: 15px;
        margin-top: 15px;
        background-color: #1a0b2e;
        box-shadow: 0 0 5px #ff00ff;
    }
    
    .result-title {
        color: #00ffff;
        font-size: 0.85rem;
        margin-bottom: 8px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 0 0 3px #00ffff;
    }
    
    .result-content {
        color: #ffffff;
        font-size: 0.75rem;
        line-height: 1.5;
    }
    
    .result-sub {
        color: #ff6600;
        font-size: 0.7rem;
        margin-top: 8px;
        text-shadow: 0 0 2px #ff6600;
    }
    
    .section-title {
        font-size: 1rem;
        text-align: center;
        color: #00ffff;
        margin: 20px 0;
        font-weight: bold;
        letter-spacing: 3px;
        text-shadow: 0 0 5px #00ffff;
        animation: neonText 3s infinite;
    }
</style>
""", unsafe_allow_html=True)

# ==================== HÀM TRÍCH XUẤT ĐẶC TRƯNG ====================
def extract_image_features_pil(img):
    gray = img.convert('L')
    gray_array = np.array(gray)
    
    hsv = img.convert('HSV')
    hsv_array = np.array(hsv)
    
    brightness = np.mean(gray_array) / 255.0
    contrast = np.std(gray_array) / 255.0
    
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges_array = np.array(edges)
    edge_density = np.mean(edges_array) / 255.0
    
    if len(hsv_array.shape) == 3:
        hue_mean = np.mean(hsv_array[:,:,0]) / 255.0
        sat_mean = np.mean(hsv_array[:,:,1]) / 255.0
    else:
        hue_mean = 0.5
        sat_mean = 0.5
    
    blur = gray.filter(ImageFilter.BLUR)
    blur_array = np.array(blur)
    texture = np.std(gray_array.astype(int) - blur_array.astype(int)) / 128.0
    texture = float(np.clip(texture, 0, 1))
    
    h, w = gray_array.shape
    center = gray_array[h//4:3*h//4, w//4:3*w//4]
    
    center_img = Image.fromarray(center)
    center_edges = center_img.filter(ImageFilter.FIND_EDGES)
    center_edges_array = np.array(center_edges)
    center_edge = np.mean(center_edges_array) / 255.0
    
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
        0: ["duong tam dao mo va ngan", "ban co xu huong giau kin cam xuc, it khi the hien tinh cam ra ben ngoai."],
        1: ["duong tam dao trung binh", "ban biet can bang giua ly tri va tinh cam. hoa dong, than thien."],
        2: ["duong tam dao sau va ro net", "ban la nguoi cuc ky giau tinh cam, nong nhiet va chan thanh."]
    },
    'head_line': {
        0: ["duong tri dao ngan hoac mo", "ban lam viec thien ve truc giac va ban nang hon logic."],
        1: ["duong tri dao trung binh", "ban co loi tu duy thuc te, suy nghi thau dao truoc khi lam."],
        2: ["duong tri dao sau ro va dai", "ban so huu tri tue sac sao, tu duy phan tich logic tuyet voi."]
    },
    'life_line': {
        0: ["duong sinh dao mo va ngan", "can chu y nghi ngoi va tranh lam viec qua suc."],
        1: ["duong sinh dao trung binh", "suc khoe va sinh luc kha tot va on dinh."],
        2: ["duong sinh dao sau dai va lien mach", "ban so huu nguon sinh luc doi dao va suc de khang tuyet voi."]
    },
    'huong_ngoai': {
        0: ["huong ngoai: thap", "ban mang dam net huong noi, toa sang trong cong viec doc lap."],
        1: ["huong ngoai: trung binh", "ban la ambivert, linh hoat trong giao tiep, thich nghi tot."],
        2: ["huong ngoai: cao", "ban la tam diem, tran day nang luong khi giao tiep, tu tin."]
    },
    'lanh_dao': {
        0: ["lanh dao: thap", "ban thich lam nguoi ho tro hon la nguoi dung dau."],
        1: ["lanh dao: trung binh", "ban co tiem nang lanh dao, co the dan dat doi nhom khi can."],
        2: ["lanh dao: cao", "to chat lanh dao manh me ban co tam nhin xa va su quyet doan."]
    },
    'noi_tam': {
        0: ["noi tam: thap", "ban song huong ngoai, it khi suy nghi nhieu ve qua khu."],
        1: ["noi tam: trung binh", "doi song noi tam phong phu, biet tu phan tinh ban than."],
        2: ["noi tam: cao", "doi song noi tam cuc ky sau sac, nhay cam va hay suy nghi."]
    },
    'sang_tao': {
        0: ["sang tao: co ban", "ban tuan thu nguyen tac, thich lam viec theo quy trinh."],
        1: ["sang tao: tot", "ban co the dua ra giai phap cai tien hieu qua."],
        2: ["sang tao: cao", "tu duy sang tao bung no y tuong doc dao va moi la."]
    },
    'diem_tinh': {
        0: ["diem tinh: thap", "ban de xuc dong, can hoc cach kiem soat cam xuc."],
        1: ["diem tinh: kha", "ban giu duoc thai do binh tinh trong hầu hết tinh huong."],
        2: ["diem tinh: cao", "kha nang kiem soat cam xuc va giu binh tinh la vu khi loi hai."]
    },
    'fortune_class': {
        0: ["van may: binh thuong", "hay kien nhan, co hoi tot se den khi ban san sang."],
        1: ["van may: tot", "thoi diem thuan loi de thuc hien cac ke hoach cua ban."],
        2: ["van may: rat tot", "moi viec deu hanh thong, nhieu co hoi lon dang cho don."]
    }
}

DISPLAY_NAMES = {
    'heart_line': 'tam dao',
    'head_line': 'tri dao',
    'life_line': 'sinh dao',
    'huong_ngoai': 'huong ngoai',
    'lanh_dao': 'lanh dao',
    'noi_tam': 'noi tam',
    'sang_tao': 'sang tao',
    'diem_tinh': 'diem tinh',
    'fortune_class': 'van may'
}

# ==================== GIAO DIỆN CHÍNH ====================
st.markdown('<div class="main-title">> palmistry analysis<span class="blinking-cursor">_</span></div>', unsafe_allow_html=True)

# Rainbow line - 1 line thoi
st.markdown('<div class="rainbow-hr"></div>', unsafe_allow_html=True)

# Layout 2 cột
col_left, col_right = st.columns([0.5, 0.5])

with col_left:
    camera_image = st.camera_input("", label_visibility="collapsed")

with col_right:
    st.markdown("""
    <div class="instruction">
        <div class="instruction-line">> chup anh long ban tay du anh sang, de trong khung hinh</div>
        <div class="instruction-line">> nhan phan tich - app tu dong phan tich cac chi so</div>
        <div class="instruction-line">> doc ket qua giai ma ve tinh cach va van menh</div>
    </div>
    """, unsafe_allow_html=True)

if camera_image:
    image = Image.open(camera_image)
    
    with col_left:
        st.image(image, caption="", width=250)
        
        if st.button("> phan tich"):
            with st.spinner("dang xu ly..."):
                if image.mode == 'RGBA':
                    image = image.convert('RGB')
                
                features = extract_image_features_pil(image)
                results = analyze_palm(features)
                
                # Hien thi ket qua
                st.markdown('<div class="section-title">/ ket qua phan tich</div>', unsafe_allow_html=True)
                
                # 3 duong chi tay chinh
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    val = results['heart_line']
                    st.markdown(f"""
                    <div class="result-box">
                        <div class="result-title">/ {DISPLAY_NAMES['heart_line']}</div>
                        <div class="result-content">{PREDICTIONS_TEXT['heart_line'][val][0]}</div>
                        <div class="result-sub">{PREDICTIONS_TEXT['heart_line'][val][1]}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_b:
                    val = results['head_line']
                    st.markdown(f"""
                    <div class="result-box">
                        <div class="result-title">/ {DISPLAY_NAMES['head_line']}</div>
                        <div class="result-content">{PREDICTIONS_TEXT['head_line'][val][0]}</div>
                        <div class="result-sub">{PREDICTIONS_TEXT['head_line'][val][1]}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_c:
                    val = results['life_line']
                    st.markdown(f"""
                    <div class="result-box">
                        <div class="result-title">/ {DISPLAY_NAMES['life_line']}</div>
                        <div class="result-content">{PREDICTIONS_TEXT['life_line'][val][0]}</div>
                        <div class="result-sub">{PREDICTIONS_TEXT['life_line'][val][1]}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 5 tinh cach
                st.markdown('<div class="section-title">/ phan tich tinh cach</div>', unsafe_allow_html=True)
                
                personality_keys = ['huong_ngoai', 'lanh_dao', 'noi_tam', 'sang_tao', 'diem_tinh']
                for key in personality_keys:
                    val = results[key]
                    st.markdown(f"""
                    <div class="result-box">
                        <div class="result-title">/ {DISPLAY_NAMES[key]}</div>
                        <div class="result-content">{PREDICTIONS_TEXT[key][val][0]}</div>
                        <div class="result-sub">{PREDICTIONS_TEXT[key][val][1]}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Van may
                st.markdown('<div class="section-title">/ van menh</div>', unsafe_allow_html=True)
                val = results['fortune_class']
                st.markdown(f"""
                <div class="result-box">
                    <div class="result-title">/ {DISPLAY_NAMES['fortune_class']}</div>
                    <div class="result-content">{PREDICTIONS_TEXT['fortune_class'][val][0]}</div>
                    <div class="result-sub">{PREDICTIONS_TEXT['fortune_class'][val][1]}</div>
                </div>
                """, unsafe_allow_html=True)
