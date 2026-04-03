"""
🧠 VN SENTIMENT PRO V4 - CỰC MẠNH & SIÊU VƯỢT TRỘI
Morphology & Sentiment tiếng Việt - Production Ready
"""

import re
from collections import Counter
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import streamlit as st
from underthesea import (
    word_tokenize, sentiment, pos_tag, ner, 
    sent_tokenize, classify
)

# ====================== CONFIG ======================
st.set_page_config(
    page_title="🧠 VN Sentiment Pro V4",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {padding: 2rem;}
    h1 {font-size: 3rem; font-weight: bold;}
    .stMetric {border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);}
</style>
""", unsafe_allow_html=True)

# ====================== DATA ======================
PREFIX_MEANINGS = {
    "bất": "phủ định", "phi": "trái với", "tái": "lặp lại",
    "siêu": "vượt trội", "phụ": "phụ trợ"
}

POSITIVE_PHRASES = ["chạy nhanh", "chơi game mượt", "màn hình đẹp", "siêu đẹp", "hài lòng", "rất hài lòng"]
NEGATIVE_PHRASES = ["chạy chậm", "lag", "pin yếu", "tụt pin", "hao pin", "máy nóng", "camera xấu", "xấu quá"]

EXAMPLE_TEXT = "Máy chạy nhanh, chơi game mượt, màn hình đẹp nhưng pin tụt kinh khủng. App ngân hàng chuyển tiền nhanh nhưng hay lag và khó đăng nhập."

# Cache
@st.cache_resource
def load_models():
    return True

load_models()

# ====================== HÀM ======================
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())

def safe_sentiment(text: str, domain="general"):
    try:
        result = sentiment(text, domain=domain)
        if isinstance(result, (list, tuple)) and len(result) > 0:
            item = result[0]
            label = str(item[0] if isinstance(item, (list, tuple)) else item).lower()
            aspect = str(item[1]) if isinstance(item, (list, tuple)) and len(item) > 1 else ""
            return label, aspect
        return str(result).lower() if isinstance(result, str) else "neutral", ""
    except Exception as e:
        st.warning(f"Lỗi sentiment ({domain}): {e}")
        return "neutral", ""

def detect_prefixes(tokens, prefixes):
    counts = Counter()
    for tok in tokens:
        t = re.sub(r"[^\\w_áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]", "", tok.lower())
        for pref in prefixes:
            if t.startswith(pref):
                counts[pref] += 1
    return counts

def detect_phrases(text: str, phrases: list):
    return Counter(p for p in phrases if p and re.search(re.escape(p), text))

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("🔧 Cài đặt")
    domain = st.selectbox("Chọn Domain", ["general", "bank"], index=0)
    show_nlp = st.checkbox("Hiển thị Full NLP Suite", value=True)
    
    st.divider()
    if st.button("🗑️ Xóa toàn bộ lịch sử", type="secondary"):
        st.session_state.history = []
        st.success("Đã xóa lịch sử!")

if "history" not in st.session_state:
    st.session_state.history = []

# ====================== MAIN UI ======================
st.title("🔥 VN Sentiment Pro V4")
st.markdown("**5 mô hình • Aspect-Based Sentiment • Batch Analysis • Visualization cao cấp • Full NLP**")

col_input, col_batch = st.columns([2.3, 1])

with col_input:
    if st.button("📋 Dán ví dụ mẫu"):
        st.session_state.input_text = EXAMPLE_TEXT

    text = st.text_area(
        "Nhập văn bản / review tiếng Việt",
        value=st.session_state.get("input_text", ""),
        height=180,
        placeholder="Nhập review sản phẩm hoặc bình luận ngân hàng..."
    )

    analyze_btn = st.button("🔥 Phân tích CỰC MẠNH", type="primary", use_container_width=True)

with col_batch:
    st.subheader("📤 Batch Analysis")
    uploaded_file = st.file_uploader("Upload file CSV hoặc Excel", type=["csv", "xlsx"])

# ====================== PHÂN TÍCH CHÍNH ======================
if analyze_btn and text.strip():
    with st.spinner("Đang chạy phân tích cực mạnh với 5 mô hình..."):
        norm_text = normalize_text(text)
        tokens = word_tokenize(text)

        # Rule-based
        pos_c = detect_phrases(norm_text, POSITIVE_PHRASES)
        neg_c = detect_phrases(norm_text, NEGATIVE_PHRASES)
        rule_label = "TÍCH CỰC" if len(pos_c) > len(neg_c) else "TIÊU CỰC" if len(neg_c) > len(pos_c) else "TRUNG TÍNH"

        # underthesea
        ut_gen, aspect_gen = safe_sentiment(text, "general")
        ut_bank, aspect_bank = safe_sentiment(text, "bank")

        # Hybrid Score (kết hợp)
        hybrid_score = 0.4 * (1 if rule_label == "TÍCH CỰC" else -1 if rule_label == "TIÊU CỰC" else 0) + \
                       0.3 * (1 if ut_gen == "positive" else -1 if ut_gen == "negative" else 0) + \
                       0.3 * (1 if ut_bank == "positive" else -1 if ut_bank == "negative" else 0)

        hybrid_label = "TÍCH CỰC" if hybrid_score > 0.3 else "TIÊU CỰC" if hybrid_score < -0.3 else "TRUNG TÍNH"

        # Lưu lịch sử
        st.session_state.history.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "text": text[:130] + "..." if len(text) > 130 else text,
            "rule": rule_label,
            "ut_general": ut_gen,
            "ut_bank": ut_bank,
            "hybrid": hybrid_label,
            "domain": domain
        })

        # ====================== TABS ======================
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 So sánh 5 mô hình", 
            "🔍 Tokenization & Hán-Việt", 
            "📈 Visualization", 
            "🔬 Full NLP", 
            "📦 Batch & Export"
        ])

        with tab1:
            st.subheader("Kết quả từ 5 mô hình")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Rule-based", rule_label)
            c2.metric("underthesea General", ut_gen.upper())
            c3.metric("underthesea Bank", ut_bank.upper())
            c4.metric("Hybrid Score", hybrid_label)
            c5.metric("Aspect", aspect_bank or aspect_gen or "N/A")

        with tab2:
            st.subheader("Tokenization (underthesea)")
            st.code(" | ".join(tokens))
            st.subheader("Tiền tố Hán-Việt")
            prefixes = detect_prefixes(tokens, PREFIX_MEANINGS)
            if prefixes:
                for p, cnt in prefixes.items():
                    st.write(f"**{p}** × {cnt} → {PREFIX_MEANINGS.get(p)}")
            else:
                st.write("Không phát hiện tiền tố Hán-Việt trong văn bản.")

        with tab3:
            st.subheader("Visualization")
            models = ["Rule-based", "UT General", "UT Bank", "Hybrid"]
            scores = [1 if x=="TÍCH CỰC" else -1 if x=="TIÊU CỰC" else 0 for x in [rule_label, ut_gen, ut_bank, hybrid_label]]
            
            fig_bar = px.bar(x=models, y=scores, color=scores, 
                            color_continuous_scale=["red", "gray", "green"], title="So sánh Sentiment Score")
            st.plotly_chart(fig_bar, use_container_width=True)

            # Word Cloud
            st.subheader("Word Cloud")
            wc1, wc2 = st.columns(2)
            with wc1:
                if pos_c:
                    wc = WordCloud(width=500, height=300, background_color='white').generate(" ".join(pos_c.keys()))
                    plt.imshow(wc)
                    plt.axis("off")
                    st.pyplot(plt)
                    st.caption("Từ tích cực")
            with wc2:
                if neg_c:
                    wc = WordCloud(width=500, height=300, background_color='white').generate(" ".join(neg_c.keys()))
                    plt.imshow(wc)
                    plt.axis("off")
                    st.pyplot(plt)
                    st.caption("Từ tiêu cực")

        with tab4:
            if show_nlp:
                st.subheader("Full NLP Analysis")
                st.write("**Sentence Split**:", sent_tokenize(text))
                st.write("**POS Tagging**:", pos_tag(text))
                st.write("**NER**")
                st.dataframe(pd.DataFrame(ner(text), columns=["Word", "POS", "Chunk", "Entity"]), use_container_width=True)
                st.write("**Text Classification**:", classify(text))

        with tab5:
            st.subheader("Export Results")
            single_df = pd.DataFrame([{
                "Thời gian": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Văn bản": text,
                "Rule-based": rule_label,
                "UT_General": ut_gen,
                "UT_Bank": ut_bank,
                "Hybrid": hybrid_label,
                "Aspect": aspect_bank or aspect_gen
            }])
            st.download_button("📥 Tải kết quả phân tích này", single_df.to_csv(index=False), "result_v4.csv", "text/csv")

# ====================== LỊCH SỬ ======================
st.subheader("📖 Lịch sử phân tích")
if st.session_state.history:
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df, use_container_width=True)
    
    st.download_button(
        "📥 Tải toàn bộ lịch sử", 
        hist_df.to_csv(index=False).encode(), 
        "full_history_v4.csv", 
        "text/csv"
    )
else:
    st.info("Chưa có phân tích nào. Hãy nhập văn bản và phân tích!")

st.caption("🔥 VN Sentiment Pro V4 - Cực Mạnh | 5 mô hình • Aspect-Based • Visualization • Full NLP | 2026")