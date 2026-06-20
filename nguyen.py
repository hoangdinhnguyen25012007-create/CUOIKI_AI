# ============================================================
# Palmistry — Public palm reading app (Streamlit) — Dashboard UI
# ============================================================
import json
from io import BytesIO
from pathlib import Path
from datetime import datetime

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ============================================================
# INTERNAL CONFIG
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
ASPECTS = ["love", "health", "career", "finance"]
IMG_SIZE = 224
DROPOUT = 0.3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ASPECT_META = {
    "love": {"category": "Tình duyên", "palm_line": "Đường Tâm Đạo", "palm_line_en": "Heart Line",
              "label": "Tình cảm", "icon": "❤️", "accent": "#C44D6E", "accent_bg": "#FDF0F3"},
    "health": {"category": "Sức khỏe", "palm_line": "Đường Sinh Đạo", "palm_line_en": "Life Line",
               "label": "Sức khỏe", "icon": "🌱", "accent": "#7C5CFC", "accent_bg": "#F1EEFE"},
    "career": {"category": "Sự nghiệp", "palm_line": "Đường Trí Đạo", "palm_line_en": "Head Line",
               "label": "Sự nghiệp", "icon": "🧠", "accent": "#3D7BF7", "accent_bg": "#EAF1FE"},
    "finance": {"category": "Tài chính", "palm_line": "Đường Tài Vận", "palm_line_en": "Fate / Money Line",
                "label": "Tài chính", "icon": "💰", "accent": "#2E8B6E", "accent_bg": "#EDF7F2"},
}

CLASS_LABEL_VI = {
    "developing": "Đang phát triển", "favorable": "Thuận lợi", "need_nurturing": "Cần vun đắp",
    "stable": "Ổn định", "advancing": "Đang thăng tiến", "building": "Đang xây dựng",
    "abundant": "Dư dả", "balanced": "Cân bằng", "need_saving": "Cần tiết kiệm",
    "attention": "Cần chú ý", "fairly_good": "Khá tốt", "good": "Rất tốt",
}

READINGS = {
    "love": {
        "developing": {"verdict": "Tình duyên đang dần nở rộ — hãy kiên nhẫn vun đắp cảm xúc.",
            "body": "Đường **Tâm Đạo** (Heart Line) cho thấy bạn không vội vàng trong chuyện tình cảm, nhưng khi đã tin tưởng thì rất sâu sắc.",
            "highlights": ["Bạn mang năng lượng **chân thành** và đáng tin cậy.", "Tình cảm cần **thời gian** — đừng vội kết luận.", "Người kiên nhẫn thường được đền đáp xứng đáng."],
            "tips": ["Dành thời gian lắng nghe cảm xúc của bản thân.", "Chia sẻ suy nghĩ với người bạn tin tưởng.", "Một cuộc trò chuyện chân thành có thể thay đổi nhiều điều."],
            "tone": "info", "score": 65},
        "favorable": {"verdict": "Tình duyên thuận lợi — bạn biết cho và nhận tình cảm cân bằng.",
            "body": "Đường **Tâm Đạo** rõ và hài hòa, gợi ý sự ấm áp tự nhiên trong cách bạn kết nối với người khác.",
            "highlights": ["Năng lượng tình cảm của bạn **tích cực và ổn định**.", "Bạn thu hút sự tin tưởng từ người xung quanh.", "Đây là giai đoạn thuận lợi để **mở rộng** các mối quan hệ."],
            "tips": ["Giữ vững sự chân thành — đó là thế mạnh của bạn.", "Dành thời gian chất lượng cho người thân yêu.", "Mở lòng đón nhận những điều tích cực bất ngờ."],
            "tone": "success", "score": 82},
        "need_nurturing": {"verdict": "Tình duyên cần được chăm sóc thêm — hãy dành tâm huyết cho các mối quan hệ.",
            "body": "Đường **Tâm Đạo** nhắc bạn rằng cảm xúc cũng cần được nuôi dưỡng.",
            "highlights": ["Bạn có tiềm năng tình cảm lớn nhưng cần **chủ động hơn**.", "Đừng để khoảng cách im lặng kéo dài.", "Sự quan tâm nhỏ tạo nên **gắn kết lớn**."],
            "tips": ["Bày tỏ cảm xúc thay vì giữ trong lòng.", "Lên kế hoạch dành thời gian cho người quan trọng.", "Lắng nghe và tha thứ là chìa khóa gắn kết."],
            "tone": "warning", "score": 55},
    },
    "health": {
        "good": {"verdict": "Sức khỏe rất tốt — năng lượng dồi dào, phục hồi nhanh.",
            "body": "Đường **Sinh Đạo** (Life Line) dài và rõ nét, phản ánh thể trạng tốt và sức bền đáng ngưỡng mộ.",
            "highlights": ["Nguồn **năng lượng sống** của bạn đang rất mạnh mẽ.", "Khả năng phục hồi thể chất ở mức tốt.", "Đây là thời điểm lý tưởng để **duy trì** thói quen lành mạnh."],
            "tips": ["Duy trì 7–8 giờ ngủ mỗi đêm.", "Uống đủ nước và ăn nhiều rau xanh.", "Dành 20 phút mỗi ngày cho hoạt động thể chất nhẹ."],
            "tone": "success", "score": 88},
        "fairly_good": {"verdict": "Sức khỏe khá tốt — cần cân bằng thêm giữa làm việc và nghỉ ngơi.",
            "body": "Đường **Sinh Đạo** cho thấy thể trạng ổn định, nhưng cơ thể thỉnh thoảng cần được lắng nghe nhiều hơn.",
            "highlights": ["Sức khỏe **ổn định** nhưng không nên chủ quan.", "Cân bằng công việc – nghỉ ngơi là chìa khóa.", "Cơ thể đang nhắc bạn **chậm lại** đôi chút."],
            "tips": ["Đặt giới hạn thời gian làm việc và nghỉ đúng lúc.", "Thử thiền hoặc hít thở sâu 5 phút mỗi sáng.", "Giảm caffeine sau 2 giờ chiều."],
            "tone": "info", "score": 70},
        "attention": {"verdict": "Sức khỏe cần chú ý — đừng bỏ qua tín hiệu mệt mỏi.",
            "body": "Đường **Sinh Đạo** gợi ý năng lượng đang cần được phục hồi.",
            "highlights": ["Cơ thể đang gửi **tín hiệu cần nghỉ ngơi**.", "Đừng đẩy giới hạn quá mức trong thời gian dài.", "Chăm sóc bản thân là **ưu tiên hàng đầu**."],
            "tips": ["Nghỉ ngơi hoàn toàn ít nhất một ngày mỗi tuần.", "Giảm thời gian màn hình trước khi ngủ.", "Cân nhắc khám sức khỏe định kỳ nếu mệt kéo dài."],
            "tone": "warning", "score": 48},
    },
    "career": {
        "stable": {"verdict": "Sự nghiệp ổn định — tư duy logic và kế hoạch là thế mạnh của bạn.",
            "body": "Đường **Trí Đạo** (Head Line) thẳng và vững cho thấy khả năng phân tích, lập kế hoạch và kiên trì theo đuổi mục tiêu nghề nghiệp.",
            "highlights": ["Bạn có **tư duy hệ thống** và làm việc có kỷ luật.", "Sự nghiệp đang đi đúng hướng — hãy kiên trì.", "Ổn định hôm nay là **bệ phóng** cho ngày mai."],
            "tips": ["Tiếp tục củng cố kỹ năng chuyên môn cốt lõi.", "Ghi chép mục tiêu dài hạn và rà soát mỗi quý.", "Tận dụng thế mạnh lập kế hoạch trong công việc."],
            "tone": "success", "score": 78},
        "advancing": {"verdict": "Sự nghiệp đang thăng tiến — đây là lúc nắm bắt cơ hội mới!",
            "body": "Đường **Trí Đạo** gợi ý khả năng học hỏi và thích nghi nhanh trong công việc.",
            "highlights": ["Bạn đang ở **đà tăng trưởng** mạnh mẽ.", "Khả năng học hỏi là **vũ khí bí mật** của bạn.", "Cơ hội mới có thể xuất hiện **bất ngờ** — hãy sẵn sàng."],
            "tips": ["Tham gia một dự án hoặc khóa học mới trong 30 ngày tới.", "Kết nối với người có kinh nghiệm trong ngành.", "Tin vào khả năng của bản thân."],
            "tone": "success", "score": 85},
        "building": {"verdict": "Sự nghiệp đang xây nền móng — kiên trì hôm nay, thành quả ngày mai.",
            "body": "Đường **Trí Đạo** cho thấy bạn đang tích lũy kiến thức và kinh nghiệm nghề nghiệp.",
            "highlights": ["Giai đoạn **xây nền** — đừng nản lòng.", "Kiến thức tích lũy sẽ mở ra nhiều cánh cửa.", "Sự kiên trì là **đức tính quý giá** nhất lúc này."],
            "tips": ["Chia mục tiêu lớn thành các bước nhỏ, dễ thực hiện.", "Ghi lại tiến trình để thấy rõ sự trưởng thành.", "Học hỏi từ sai lầm — đó là bài học quý giá."],
            "tone": "info", "score": 60},
    },
    "finance": {
        "abundant": {"verdict": "Tài chính dư dả — khả năng thu hút và giữ tiền khá tốt.",
            "body": "Đường **Tài Vận** (Fate / Money Line) gợi ý bạn có trực giác tốt trong quản lý tài chính.",
            "highlights": ["Năng lượng **tài lộc** đang thuận lợi.", "Bạn biết cách **tạo và giữ** giá trị.", "Hành động có kế hoạch sẽ mang lại kết quả tốt."],
            "tips": ["Xem xét đa dạng hóa nguồn thu nhập.", "Đặt mục tiêu tài chính cụ thể cho 6 tháng tới.", "Theo dõi chi tiêu để duy trì đà tích cực."],
            "tone": "success", "score": 80},
        "balanced": {"verdict": "Tài chính cân bằng — chi tiêu hợp lý, tích lũy đều đặn.",
            "body": "Đường **Tài Vận** cho thấy bạn biết cân bằng giữa hưởng thụ và tiết kiệm.",
            "highlights": ["Bạn đang trên con đường **tài chính bền vững**.", "Chi tiêu và tích lũy ở mức **hài hòa**.", "Duy trì thói quen hiện tại là điều khôn ngoan."],
            "tips": ["Tiếp tục theo dõi chi tiêu hàng tháng.", "Duy trì quỹ dự phòng ít nhất 3 tháng chi phí.", "Thưởng cho bản thân đôi khi — bạn xứng đáng!"],
            "tone": "info", "score": 75},
        "need_saving": {"verdict": "Tài chính cần tiết kiệm hơn — lập kế hoạch sẽ giúp bạn an tâm.",
            "body": "Đường **Tài Vận** nhắc bạn cân nhắc kỹ trước các khoản chi lớn.",
            "highlights": ["Đây là lúc cần **chủ động quản lý** chi tiêu.", "Tiết kiệm nhỏ mỗi ngày tạo **an tâm lớn**.", "Tránh quyết định tài chính bốc đồng."],
            "tips": ["Lập ngân sách chi tiêu hàng tuần.", "Trì hoãn quyết định mua sắm lớn ít nhất 48 giờ.", "Bắt đầu tiết kiệm tự động mỗi tháng."],
            "tone": "warning", "score": 50},
    },
}


# ============================================================
# SESSION STATE
# ============================================================
def init_session_state():
    defaults = {
        "hand_image_bytes": None,
        "image_signature": None,
        "reading_results": None,
        "analyzed": False,
        "active_page": "Dashboard",
        "active_detail_tab": "Tổng quan",
        "history": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def save_hand_image(img: Image.Image, source: str = "Tải lên") -> None:
    buf = BytesIO()
    img.save(buf, format="PNG")
    raw = buf.getvalue()
    if raw == st.session_state.image_signature:
        return
    st.session_state.hand_image_bytes = raw
    st.session_state.image_signature = raw
    st.session_state.analyzed = False
    st.session_state.reading_results = None
    st.session_state["_last_source"] = source


def get_hand_image() -> Image.Image | None:
    if not st.session_state.hand_image_bytes:
        return None
    return Image.open(BytesIO(st.session_state.hand_image_bytes)).convert("RGB")


def clear_session():
    st.session_state.hand_image_bytes = None
    st.session_state.image_signature = None
    st.session_state.reading_results = None
    st.session_state.analyzed = False


# ============================================================
# BACKEND (model loading / inference) — unchanged logic
# ============================================================
def _norm_key(raw: str) -> str:
    return raw.strip().lower().replace(" ", "_")


def _find_class_names_file(aspect_dir: Path) -> Path:
    for name in (f"class_names_{aspect_dir.name}.json", f"class_names {aspect_dir.name}.json", "class_names.json"):
        p = aspect_dir / name
        if p.exists():
            return p
    files = sorted(aspect_dir.glob("class_names*.json"))
    if files:
        return files[0]
    raise FileNotFoundError(aspect_dir)


def _find_model_file(aspect_dir: Path) -> Path:
    for name in (f"best_model_{aspect_dir.name}.pth", f"best_model {aspect_dir.name}.pth", "best_model.pth", "model.pth"):
        p = aspect_dir / name
        if p.exists():
            return p
    files = sorted(aspect_dir.glob("*.pth"))
    if files:
        return files[0]
    raise FileNotFoundError(aspect_dir)


def _label_vi(raw_label: str) -> str:
    return CLASS_LABEL_VI.get(_norm_key(raw_label), raw_label.strip())


def _reading(aspect: str, raw_label: str) -> dict:
    key = _norm_key(raw_label)
    meta = ASPECT_META[aspect]
    data = READINGS.get(aspect, {}).get(key)
    if data:
        return {**data, "label_vi": _label_vi(raw_label), "aspect": aspect}
    return {
        "aspect": aspect, "label_vi": _label_vi(raw_label),
        "verdict": f"{meta['category']}: {_label_vi(raw_label)}",
        "body": f"{meta['palm_line']} ({meta['palm_line_en']}) đang ở trạng thái đặc biệt. Hãy lắng nghe trực giác và quan sát thêm qua thời gian.",
        "highlights": ["Giữ bình an và tin vào hành trình của bạn."],
        "tips": ["Dành thời gian suy ngẫm về bản thân mỗi ngày."],
        "tone": "info", "score": 60,
    }


@st.cache_resource(show_spinner=False)
def load_all_models():
    loaded = {}
    for aspect in ASPECTS:
        aspect_dir = MODELS_DIR / aspect
        if not aspect_dir.is_dir():
            continue
        try:
            classes_path = _find_class_names_file(aspect_dir)
            model_path = _find_model_file(aspect_dir)
        except FileNotFoundError:
            continue
        try:
            with open(classes_path, encoding="utf-8") as f:
                class_names = [n.strip() for n in json.load(f)]
            model = models.mobilenet_v3_small(weights=None)
            in_features = model.classifier[0].in_features
            model.classifier = nn.Sequential(
                nn.Linear(in_features, 256), nn.Hardswish(), nn.Dropout(p=DROPOUT),
                nn.Linear(256, len(class_names)),
            )
            try:
                state = torch.load(model_path, map_location=DEVICE, weights_only=True)
            except TypeError:
                state = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(state)
            model.to(DEVICE).eval()
            loaded[aspect] = {"model": model, "classes": class_names}
        except Exception:
            continue
    return loaded


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict_all(img: Image.Image, models_dict: dict) -> dict:
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    results = {}
    with torch.no_grad():
        for aspect, data in models_dict.items():
            outputs = data["model"](tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            top_idx = probs.argmax().item()
            raw = data["classes"][top_idx]
            results[aspect] = _reading(aspect, raw)
    return results


def run_analysis_callback(models_dict: dict, source: str = "Tải lên") -> None:
    img = get_hand_image()
    if img is None:
        return
    results = predict_all(img, models_dict)
    st.session_state.reading_results = results
    st.session_state.analyzed = True
    st.session_state.history.insert(0, {
        "time": datetime.now().strftime("%d/%m/%Y - %H:%M"),
        "source": source,
        "love": results.get("love", {}).get("score", 0),
        "career": results.get("career", {}).get("score", 0),
        "thumb": st.session_state.hand_image_bytes,
    })


# ============================================================
# STYLES
# ============================================================
def inject_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
        .stApp { background: #F6F7FB; }

        .block-container { max-width: 1280px; padding-top: 1.4rem; padding-bottom: 2rem; }

        #MainMenu, footer { visibility: hidden; }
        header[data-testid="stHeader"] { background: transparent; }

        /* ---- Sidebar ---- */
        section[data-testid="stSidebar"] {
            background: #FFFFFF;
            border-right: 1px solid #ECEDF2;
        }
        section[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }

        .sb-brand { display:flex; align-items:center; gap:.6rem; margin-bottom:1.6rem; padding: 0 .2rem;}
        .sb-brand-icon { width:38px; height:38px; border-radius:10px; background:#6D5BF0;
            display:flex; align-items:center; justify-content:center; font-size:1.1rem; color:#fff; flex-shrink:0;}
        .sb-brand-title { font-weight:700; font-size:1rem; color:#1A1A2E; margin:0; line-height:1.2;}
        .sb-brand-sub { font-size:.7rem; color:#9A9CB0; margin:0; }

        div[data-testid="stSidebar"] button {
            background: transparent !important;
            color: #5B5D72 !important;
            border: none !important;
            text-align: left !important;
            justify-content: flex-start !important;
            font-weight: 500 !important;
            border-radius: 10px !important;
            padding: .55rem .8rem !important;
        }
        div[data-testid="stSidebar"] button:hover { background: #F1EFFE !important; color:#6D5BF0 !important; }
        div[data-testid="stSidebar"] button p { text-align:left !important; }

        .sb-active button {
            background: #6D5BF0 !important;
            color: #FFFFFF !important;
        }

        .premium-card {
            background: linear-gradient(160deg,#EFEBFF,#F8F6FF);
            border: 1px solid #E4DEFB;
            border-radius: 16px;
            padding: 1.1rem 1rem;
            text-align: center;
            margin: 1.4rem 0 1rem 0;
        }
        .premium-card .crown { font-size:1.6rem; }
        .premium-card h4 { margin:.4rem 0 .2rem 0; color:#4B3FB0; font-size:.95rem; }
        .premium-card p { color:#8C8DA3; font-size:.78rem; margin-bottom:.6rem; }
        .premium-list { text-align:left; font-size:.78rem; color:#5B5D72; margin: .4rem 0 .8rem .2rem; line-height:1.7;}

        .sb-profile { display:flex; align-items:center; gap:.6rem; padding:.6rem .2rem; border-top:1px solid #ECEDF2; margin-top:.6rem;}
        .sb-profile-name { font-weight:600; font-size:.85rem; color:#1A1A2E; margin:0;}
        .sb-profile-email { font-size:.72rem; color:#9A9CB0; margin:0;}
        .sb-avatar { width:34px; height:34px; border-radius:50%; background:#D8D4FB; flex-shrink:0;
            display:flex; align-items:center; justify-content:center; font-weight:600; color:#6D5BF0; }

        /* ---- Top bar ---- */
        .topbar { display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:1.4rem; flex-wrap:wrap; gap:.6rem;}
        .topbar h1 { font-size:1.5rem; font-weight:700; color:#1A1A2E; margin:0; }
        .topbar .sub { color:#8C8DA3; font-size:.88rem; margin-top:.2rem; }
        .topbar-right { display:flex; align-items:center; gap:.7rem; }
        .pill { background:#FFFFFF; border:1px solid #ECEDF2; border-radius:999px; padding:.45rem .9rem;
            font-size:.82rem; color:#5B5D72; font-weight:500; white-space:nowrap; }
        .icon-btn { width:38px; height:38px; border-radius:50%; background:#FFFFFF; border:1px solid #ECEDF2;
            display:flex; align-items:center; justify-content:center; font-size:1rem; }

        /* ---- Cards ---- */
        .card {
            background:#FFFFFF; border:1px solid #ECEDF2; border-radius:16px;
            padding:1.1rem 1.25rem; box-shadow:0 2px 10px rgba(20,20,43,0.03); margin-bottom:1rem;
        }
        .upload-zone { border:2px dashed #DBDCE8; border-radius:14px; padding:1.4rem; text-align:center; }
        .upload-zone .ic { font-size:1.6rem; }
        .upload-zone h4 { margin:.4rem 0 .1rem 0; font-size:.95rem; color:#1A1A2E; }
        .upload-zone p { font-size:.78rem; color:#9A9CB0; margin:0; }

        .stat-card { display:flex; align-items:flex-start; gap:.7rem; }
        .stat-ic { width:42px; height:42px; border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:1.15rem; flex-shrink:0;}
        .stat-label { color:#8C8DA3; font-size:.82rem; margin:0 0 .15rem 0; }
        .stat-value { font-size:1.4rem; font-weight:700; color:#1A1A2E; margin:0; }
        .stat-tag { font-size:.72rem; color:#9A9CB0; }
        .stat-bar-bg { background:#EFEFF5; border-radius:6px; height:6px; margin-top:.55rem; overflow:hidden;}
        .stat-bar-fill { height:6px; border-radius:6px; }

        .traits-row { display:flex; flex-wrap:wrap; gap:.5rem; margin: .4rem 0 .9rem 0; }
        .trait-pill { padding:.35rem .8rem; border-radius:999px; font-size:.8rem; font-weight:500; }

        .insight-box { background:#F7F6FE; border-radius:12px; padding:.9rem 1rem; display:flex; gap:.7rem; align-items:flex-start;}
        .insight-box .star { color:#6D5BF0; font-size:1.1rem; }
        .insight-box p { margin:0; font-size:.85rem; color:#4B4C63; line-height:1.6; }

        .line-badge { display:inline-block; font-size:.74rem; font-weight:600; padding:.22rem .6rem; border-radius:999px; margin-bottom:.5rem;}

        div[data-testid="stTable"] table, .hist-table { width:100%; border-collapse:collapse; font-size:.83rem; }
        .hist-table th { text-align:left; color:#9A9CB0; font-weight:500; padding:.4rem .5rem; border-bottom:1px solid #ECEDF2; }
        .hist-table td { padding:.5rem; border-bottom:1px solid #F2F2F7; color:#3D3E52; vertical-align:middle;}

        div[data-testid="stButton"] > button[kind="primary"] {
            background:#6D5BF0 !important; color:#fff !important; border:none !important;
            border-radius:10px !important; font-weight:600 !important; box-shadow:0 4px 12px rgba(109,91,240,.28);
        }
        div[data-testid="stButton"] > button[kind="primary"]:hover { background:#5B49DE !important; }

        .stTabs [data-baseweb="tab-list"] { gap:.2rem; background:#F1F1F6; border-radius:10px; padding:4px; flex-wrap:wrap;}
        .stTabs [data-baseweb="tab"] { border-radius:8px; color:#5B5D72; font-size:.82rem; font-weight:500; background:transparent; border:none;}
        .stTabs [aria-selected="true"] { background:#FFFFFF !important; color:#1A1A2E !important; box-shadow:0 1px 4px rgba(0,0,0,.08); font-weight:600 !important;}

        div[data-testid="stImage"] img { border-radius:12px; border:1px solid #ECEDF2; }
        div[data-testid="stCamera"] { max-height:320px; overflow:hidden; border-radius:12px;}

        @media (max-width: 768px) {
            .topbar { flex-direction:column; align-items:flex-start; }
            .block-container { padding-left:.7rem; padding-right:.7rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR
# ============================================================
NAV_ITEMS = [
    ("Dashboard", "🏠"), ("Phân tích mới", "📷"), ("Lịch sử phân tích", "🕘"),
    ("Kết quả đã lưu", "🔖"), ("Báo cáo", "📊"), ("Cài đặt", "⚙️"), ("Giới thiệu", "ℹ️"),
]


def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div class="sb-brand">
                <div class="sb-brand-icon">🖐</div>
                <div>
                    <p class="sb-brand-title">PalmInsight AI</p>
                    <p class="sb-brand-sub">Khám phá vận mệnh qua lòng bàn tay</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for name, icon in NAV_ITEMS:
            active = st.session_state.active_page == name
            wrap_class = "sb-active" if active else ""
            st.markdown(f'<div class="{wrap_class}">', unsafe_allow_html=True)
            if st.button(f"{icon}  {name}", key=f"nav_{name}", use_container_width=True):
                st.session_state.active_page = name
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="premium-card">
                <div class="crown">👑</div>
                <h4>Nâng cấp Premium</h4>
                <p>Mở khóa toàn bộ tính năng nâng cao và trải nghiệm phân tích chuyên sâu.</p>
                <div class="premium-list">
                    ✅ Phân tích chuyên sâu<br>
                    ✅ Không giới hạn lượt<br>
                    ✅ Lưu trữ kết quả<br>
                    ✅ Báo cáo chi tiết
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button("Nâng cấp ngay →", type="primary", use_container_width=True, key="btn_upgrade")

        st.markdown("<div style='margin-top:.8rem;'></div>", unsafe_allow_html=True)
        dark_mode = st.toggle("🌙 Chế độ tối", key="dark_mode_toggle")

        st.markdown(
            """
            <div class="sb-profile">
                <div class="sb-avatar">H</div>
                <div>
                    <p class="sb-profile-name">Hoàng Đình Nguyên</p>
                    <p class="sb-profile-email">hoangdinhnguyen25012007@gmail.com</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# TOP BAR
# ============================================================
def render_topbar():
    remaining = max(0, 3 - len(st.session_state.history))
    st.markdown(
        f"""
        <div class="topbar">
            <div>
                <h1>Xin chào, Hoàng Đình Nguyên! 👋</h1>
                <p class="sub">Hãy bắt đầu phân tích chỉ tay của bạn để khám phá những điều thú vị.</p>
            </div>
            <div class="topbar-right">
                <span class="pill">💎 {remaining} lượt phân tích còn lại</span>
                <div class="icon-btn">🔔</div>
                <div class="icon-btn">🙂</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# UPLOAD / CAMERA
# ============================================================
def handle_image_input():
    col1, col2 = st.columns([1.4, 1], gap="large")
    with col1:
        st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
        st.markdown(
            '<div class="ic">⬆️</div><h4>Kéo thả ảnh bàn tay vào đây</h4>'
            '<p>hoặc click để chọn ảnh từ máy tính<br>Định dạng hỗ trợ: JPG, PNG, JPEG (Tối đa 10MB)</p>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader("upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)
        if uploaded is not None:
            save_hand_image(Image.open(uploaded).convert("RGB"), source="Tải lên")

    with col2:
        st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
        st.markdown(
            '<div class="ic">📷</div><h4>Sử dụng camera</h4>'
            '<p>Chụp ảnh bàn tay trực tiếp bằng webcam của bạn</p>',
            unsafe_allow_html=True,
        )
        use_cam = st.toggle("Mở camera", key="open_cam")
        if use_cam:
            cam = st.camera_input("camera", label_visibility="collapsed")
            if cam is not None:
                save_hand_image(Image.open(cam).convert("RGB"), source="Camera")
        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# STAT CARDS
# ============================================================
def render_stat_cards(results: dict):
    cols = st.columns(4, gap="medium")
    tags = {"success": "Rất tốt", "info": "Tốt", "warning": "Cần chú ý"}
    for col, aspect in zip(cols, ASPECTS):
        meta = ASPECT_META[aspect]
        r = results.get(aspect)
        score = r["score"] if r else 0
        tag = tags.get(r["tone"], "Tốt") if r else "—"
        with col:
            st.markdown(
                f"""
                <div class="card stat-card">
                    <div class="stat-ic" style="background:{meta['accent_bg']};color:{meta['accent']};">{meta['icon']}</div>
                    <div style="flex:1;">
                        <p class="stat-label">{meta['label']}</p>
                        <p class="stat-value" style="color:{meta['accent']};">{score}%</p>
                        <span class="stat-tag">{tag}</span>
                        <div class="stat-bar-bg"><div class="stat-bar-fill" style="width:{score}%;background:{meta['accent']};"></div></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_placeholder_stat_cards():
    cols = st.columns(4, gap="medium")
    for col, aspect in zip(cols, ASPECTS):
        meta = ASPECT_META[aspect]
        with col:
            st.markdown(
                f"""
                <div class="card stat-card">
                    <div class="stat-ic" style="background:{meta['accent_bg']};color:{meta['accent']};">{meta['icon']}</div>
                    <div style="flex:1;">
                        <p class="stat-label">{meta['label']}</p>
                        <p class="stat-value" style="color:#C9C9D6;">—</p>
                        <span class="stat-tag">Chưa phân tích</span>
                        <div class="stat-bar-bg"><div class="stat-bar-fill" style="width:0%;background:{meta['accent']};"></div></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# DETAIL TABS
# ============================================================
def render_tone_box(tone: str, text: str):
    if tone == "success":
        st.success(text)
    elif tone == "warning":
        st.warning(text)
    else:
        st.info(text)


def render_aspect_detail(aspect: str, result: dict):
    meta = ASPECT_META[aspect]
    st.markdown(
        f'<span class="line-badge" style="background:{meta["accent_bg"]};color:{meta["accent"]}">'
        f'{meta["palm_line"]} · {meta["palm_line_en"]}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(f"#### {meta['icon']} {result['verdict']}")
    render_tone_box(result["tone"], f"**Kết quả:** {result['label_vi']}")
    st.markdown(result["body"])
    st.markdown("**🔮 Điểm nổi bật**")
    for item in result["highlights"]:
        st.markdown(f"- {item}")
    st.markdown("**✨ Lời khuyên dành cho bạn**")
    for tip in result["tips"]:
        st.markdown(f"- {tip}")


def render_overview(results: dict):
    st.markdown(
        """
        <div class="insight-box">
            <div class="star">★</div>
            <p><strong>Tổng quan về bạn:</strong> Bạn là người có tư duy logic, sáng tạo và rất kiên trì trong công việc.
            Bạn có khả năng đạt được thành công nhờ sự nỗ lực không ngừng. Trong tình cảm, bạn chân thành và luôn
            quan tâm đến người khác. Giai đoạn sắp tới, bạn sẽ có nhiều cơ hội tốt trong sự nghiệp và tài vận.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_detail_tabs(results: dict):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("##### Phân tích chi tiết")
    tab_overview, tab_love, tab_career, tab_finance, tab_health = st.tabs(
        ["📋 Tổng quan", "❤️ Tình duyên", "🧠 Sự nghiệp", "💰 Tài vận", "🌱 Sức khỏe"]
    )
    with tab_overview:
        render_overview(results)
    with tab_love:
        render_aspect_detail("love", results["love"])
    with tab_career:
        render_aspect_detail("career", results["career"])
    with tab_finance:
        render_aspect_detail("finance", results["finance"])
    with tab_health:
        render_aspect_detail("health", results["health"])
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# TRAITS + HISTORY
# ============================================================
def render_traits_card():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("##### Tính cách nổi bật")
    traits = [
        ("Logic", "#3D7BF7", "#EAF1FE"), ("Sáng tạo", "#D88A1E", "#FCF1E3"),
        ("Kiên trì", "#2E8B6E", "#EAF6EF"), ("Tư duy tốt", "#7C5CFC", "#F1EEFE"),
        ("Cẩn thận", "#C44D6E", "#FDF0F3"),
    ]
    pills = "".join(
        f'<span class="trait-pill" style="background:{bg};color:{c}">{t}</span>' for t, c, bg in traits
    )
    st.markdown(f'<div class="traits-row">{pills}</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="insight-box">
            <div class="star">★</div>
            <p>Bạn là người có tư duy logic, sáng tạo và rất kiên trì trong công việc. Bạn có khả năng đạt được
            thành công nhờ sự nỗ lực không ngừng và luôn quan tâm đến người khác.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_history_card():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    h1, h2 = st.columns([3, 1])
    with h1:
        st.markdown("##### Lịch sử phân tích")
    with h2:
        st.button("Xem tất cả", key="btn_history_all", use_container_width=True)

    if not st.session_state.history:
        st.caption("Chưa có lịch sử phân tích nào.")
    else:
        rows = ""
        for item in st.session_state.history[:5]:
            rows += (
                f"<tr><td>{item['time']}</td><td>{item['source']}</td>"
                f"<td>Tình cảm: {item['love']}% · Sự nghiệp: {item['career']}%</td></tr>"
            )
        st.markdown(
            f"""
            <table class="hist-table">
                <tr><th>Thời gian</th><th>Nguồn</th><th>Kết quả</th></tr>
                {rows}
            </table>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# PAGE: DASHBOARD
# ============================================================
def render_dashboard_page(models_dict: dict):
    render_topbar()

    st.markdown('<div class="card">', unsafe_allow_html=True)
    handle_image_input()
    st.markdown("</div>", unsafe_allow_html=True)

    img = get_hand_image()
    results = st.session_state.reading_results if st.session_state.analyzed else None

    if results:
        render_stat_cards(results)
    else:
        render_placeholder_stat_cards()

    col_left, col_right = st.columns([1, 1.5], gap="large")

    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("##### Ảnh bàn tay của bạn")
        if img is not None:
            st.image(img, use_container_width=True)
            if not st.session_state.analyzed:
                st.button(
                    "✦ Xem Luận Giải Chỉ Tay", type="primary", use_container_width=True,
                    key="btn_analyze", on_click=run_analysis_callback,
                    args=(models_dict, st.session_state.get("_last_source", "Tải lên")),
                )
            else:
                bcol1, bcol2 = st.columns(2)
                with bcol1:
                    st.button(
                        "🔄 Luận giải lại", use_container_width=True, key="btn_reanalyze",
                        on_click=run_analysis_callback,
                        args=(models_dict, st.session_state.get("_last_source", "Tải lên")),
                    )
                with bcol2:
                    if st.button("↺ Ảnh khác", use_container_width=True, key="btn_clear"):
                        clear_session()
                        st.rerun()
        else:
            st.caption("Chưa có ảnh bàn tay nào. Hãy tải lên hoặc chụp ảnh ở trên.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        render_traits_card()

    if results:
        render_detail_tabs(results)

    render_history_card()


def render_simple_placeholder_page(title: str):
    st.markdown(f"<h2 style='color:#1A1A2E;'>{title}</h2>", unsafe_allow_html=True)
    st.markdown('<div class="card"><p style="color:#8C8DA3;">Tính năng đang được phát triển.</p></div>', unsafe_allow_html=True)


# ============================================================
# MAIN
# ============================================================
def main():
    try:
        init_session_state()

        st.set_page_config(
            page_title="PalmInsight AI",
            page_icon="🖐",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        inject_styles()
        render_sidebar()

        models_dict = load_all_models()
        if not models_dict:
            st.error("Ứng dụng tạm thời chưa sẵn sàng. Vui lòng kiểm tra lại thư mục /models.")
            st.stop()

        page = st.session_state.active_page
        if page == "Dashboard" or page == "Phân tích mới":
            render_dashboard_page(models_dict)
        elif page == "Lịch sử phân tích":
            render_topbar()
            render_history_card()
        else:
            render_topbar()
            render_simple_placeholder_page(page)

    except Exception as e:
        st.error(f"Lỗi khởi chạy ứng dụng: {e}")
        st.exception(e)
        st.stop()


if __name__ == "__main__":
    main()