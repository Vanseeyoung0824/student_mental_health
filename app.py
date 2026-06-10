import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import pandas as pd

# ──────────────────────────────────────────────
# 페이지 설정
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="학생 정신건강 분석",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# CSS 스타일
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Noto Sans KR', sans-serif;
    }

    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #6b7280;
        font-size: 1rem;
    }

    .result-card {
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        text-align: center;
        border: 2px solid;
    }
    .result-card.high-risk {
        background: #fff1f2;
        border-color: #fca5a5;
    }
    .result-card.mid-risk {
        background: #fffbeb;
        border-color: #fcd34d;
    }
    .result-card.low-risk {
        background: #f0fdf4;
        border-color: #86efac;
    }
    .result-card h2 {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .result-card .emoji {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }

    .info-box {
        background: #f8fafc;
        border-left: 4px solid #6366f1;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        font-size: 0.95rem;
        color: #374151;
    }

    .tip-card {
        background: white;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        border-left: 4px solid #6366f1;
    }

    .stSlider > div > div > div > div {
        background: #6366f1 !important;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1a1a2e;
    }

    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# 모델 로드
# ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    import os, glob
    pkl_files = glob.glob("*.pkl")
    if not pkl_files:
        st.error("모델 파일(.pkl)을 찾을 수 없습니다. app.py와 같은 폴더에 넣어주세요.")
        st.stop()
    return joblib.load(pkl_files[0])

model = load_model()

# 클러스터 센터 분석 (표준화된 값)
# Cluster 0: [-1.19,  1.26,  0.99] → 스트레스 낮음, 불안/우울 높음
# Cluster 1: [ 1.07, -1.00, -0.91] → 스트레스 높음, 불안/우울 낮음
# Cluster 2: [-0.11,  0.00,  0.11] → 전반적 중간

CLUSTER_INFO = {
    0: {
        "label": "정서적 어려움 주의군",
        "emoji": "😟",
        "color": "high-risk",
        "color_hex": "#ef4444",
        "description": "불안감과 우울감이 다소 높은 편입니다. 감정 조절과 정서적 지원이 필요할 수 있습니다.",
        "tips": [
            "📞 학교 상담 선생님 또는 Wee 클래스를 방문해 보세요",
            "🛌 규칙적인 수면 습관을 유지해 보세요 (7~9시간 권장)",
            "🚶 매일 30분 이상 가벼운 산책이나 운동을 해보세요",
            "📓 감정 일기를 써서 내 감정을 관찰해 보세요",
            "🤝 신뢰할 수 있는 친구나 가족에게 마음을 털어놓아 보세요",
        ],
    },
    1: {
        "label": "스트레스 주의군",
        "emoji": "😤",
        "color": "mid-risk",
        "color_hex": "#f59e0b",
        "description": "학업이나 환경적 스트레스가 높은 편입니다. 스트레스 관리 방법을 찾아보는 것이 도움이 됩니다.",
        "tips": [
            "📚 공부 시간과 휴식 시간을 균형 있게 나눠보세요",
            "🎯 목표를 작게 나눠 하나씩 달성하는 습관을 들여보세요",
            "🎵 좋아하는 음악 감상이나 취미 활동으로 스트레스를 해소해 보세요",
            "💬 선생님이나 부모님과 학업 고민을 솔직하게 이야기해 보세요",
            "🧘 심호흡이나 간단한 명상으로 긴장을 풀어보세요",
        ],
    },
    2: {
        "label": "안정군",
        "emoji": "😊",
        "color": "low-risk",
        "color_hex": "#22c55e",
        "description": "전반적으로 안정적인 정신건강 상태를 보이고 있습니다. 현재의 좋은 습관을 계속 유지하세요!",
        "tips": [
            "✅ 지금처럼 건강한 생활 습관을 유지하세요",
            "🌱 주변 친구들의 어려움에도 관심을 기울여 보세요",
            "📖 새로운 취미나 도전을 통해 긍정적인 경험을 쌓아보세요",
            "💪 운동과 충분한 수면으로 몸과 마음의 건강을 지켜요",
        ],
    },
}

# ──────────────────────────────────────────────
# 헤더
# ──────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🧠 학생 정신건강 분석</h1>
    <p>AI 모델이 나의 정신건강 상태를 분석하고 맞춤 도움말을 제공합니다</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ──────────────────────────────────────────────
# 입력 폼
# ──────────────────────────────────────────────
st.subheader("📋 현재 나의 상태를 알려주세요")

st.markdown("""
<div class="info-box">
아래 세 가지 항목에 대해 최근 2주를 기준으로 솔직하게 답해주세요.
점수가 높을수록 해당 증상이 심한 것을 의미합니다.
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    stress = st.slider(
        "😰 스트레스 수준",
        min_value=1,
        max_value=10,
        value=5,
        help="학업, 교우관계, 가정 등에서 느끼는 전반적인 스트레스 정도"
    )
    st.caption("1: 거의 없음 → 10: 매우 심함")

with col2:
    anxiety = st.slider(
        "😨 불안감",
        min_value=1,
        max_value=10,
        value=5,
        help="가슴이 두근거리거나 걱정이 많아지는 느낌의 정도"
    )
    st.caption("1: 거의 없음 → 10: 매우 심함")

with col3:
    depression = st.slider(
        "😔 우울감",
        min_value=1,
        max_value=10,
        value=5,
        help="의욕 저하, 슬픔, 무기력함 등의 정도"
    )
    st.caption("1: 거의 없음 → 10: 매우 심함")

st.markdown("<br>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# 분석 버튼
# ──────────────────────────────────────────────
analyze_btn = st.button("🔍 정신건강 상태 분석하기", use_container_width=True, type="primary")

if analyze_btn:
    # 표준화 (모델 학습 시 사용한 스케일 기준으로 변환)
    # 모델은 1~10 스케일의 데이터로 학습됐다고 가정, 평균 5, std 약 2.5~3
    raw = np.array([[stress, anxiety, depression]], dtype=float)

    # 간단한 표준화 (1~10 범위 기준, 평균 5.5, std 2.8)
    scaler_mean = np.array([5.5, 5.5, 5.5])
    scaler_std = np.array([2.8, 2.8, 2.8])
    scaled = (raw - scaler_mean) / scaler_std

    cluster = model.predict(scaled)[0]
    info = CLUSTER_INFO[cluster]

    st.markdown("---")
    st.subheader("📊 분석 결과")

    # 결과 카드
    st.markdown(f"""
    <div class="result-card {info['color']}">
        <div class="emoji">{info['emoji']}</div>
        <h2>{info['label']}</h2>
        <p style="color: #374151; font-size: 1rem;">{info['description']}</p>
    </div>
    """, unsafe_allow_html=True)

    # 점수 요약
    m1, m2, m3 = st.columns(3)
    m1.metric("😰 스트레스", f"{stress}/10")
    m2.metric("😨 불안감", f"{anxiety}/10")
    m3.metric("😔 우울감", f"{depression}/10")

    # 레이더 차트
    st.markdown("<br>", unsafe_allow_html=True)
    categories = ['스트레스', '불안감', '우울감']
    values = [stress, anxiety, depression]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor=info['color_hex'] + '33',
        line=dict(color=info['color_hex'], width=2),
        name='내 상태',
        marker=dict(size=8, color=info['color_hex']),
    ))
    fig.add_trace(go.Scatterpolar(
        r=[5, 5, 5, 5],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(156,163,175,0.1)',
        line=dict(color='#9ca3af', width=1, dash='dash'),
        name='평균 (5점)',
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], tickfont=dict(size=10)),
            angularaxis=dict(tickfont=dict(size=13, family='Noto Sans KR')),
        ),
        showlegend=True,
        legend=dict(font=dict(family='Noto Sans KR')),
        margin=dict(t=30, b=30),
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig, use_container_width=True)

    # 맞춤 도움말
    st.markdown("### 💡 맞춤 도움말")
    for tip in info['tips']:
        st.markdown(f'<div class="tip-card">{tip}</div>', unsafe_allow_html=True)

    # 위기상담 안내 (고위험군 또는 점수 높을 때)
    if cluster == 0 or (anxiety >= 8 or depression >= 8):
        st.markdown("<br>", unsafe_allow_html=True)
        st.error("""
**🚨 전문 상담이 필요하신가요?**

힘들 때 혼자 견디지 않아도 됩니다. 아래 기관에 연락하세요.

- **청소년 전화 1388** — 24시간 운영 (전화 · 문자)
- **자살예방 상담전화 1393** — 24시간 운영
- **Wee 클래스** — 학교 내 상담실
        """)

    st.markdown("""
    <div class="footer">
        본 결과는 AI 모델에 의한 참고 정보이며, 전문적인 의학 진단을 대체하지 않습니다.<br>
        정확한 진단과 치료는 전문 상담사나 의사에게 받으시기 바랍니다.
    </div>
    """, unsafe_allow_html=True)

else:
    # 사용 안내
    st.markdown("""
    <div class="info-box">
    📌 <strong>사용 방법</strong><br>
    위 세 가지 슬라이더를 조절하여 최근 2주간 나의 상태를 입력한 뒤,
    <strong>[정신건강 상태 분석하기]</strong> 버튼을 누르면 AI가 분석해 드립니다.
    </div>
    """, unsafe_allow_html=True)

    # 클러스터 설명 카드
    st.markdown("### 🔎 분석 유형 안내")
    cols = st.columns(3)
    for i, (cluster_id, info) in enumerate(CLUSTER_INFO.items()):
        with cols[i]:
            border_color = {"high-risk": "#fca5a5", "mid-risk": "#fcd34d", "low-risk": "#86efac"}[info["color"]]
            bg_color = {"high-risk": "#fff1f2", "mid-risk": "#fffbeb", "low-risk": "#f0fdf4"}[info["color"]]
            st.markdown(f"""
            <div style="background:{bg_color}; border:2px solid {border_color};
                        border-radius:12px; padding:1rem; text-align:center; height:140px;">
                <div style="font-size:2rem;">{info['emoji']}</div>
                <div style="font-weight:700; font-size:0.9rem; color:#1a1a2e; margin-top:0.3rem;">
                    {info['label']}
                </div>
                <div style="font-size:0.78rem; color:#6b7280; margin-top:0.3rem;">
                    {info['description'][:35]}...
                </div>
            </div>
            """, unsafe_allow_html=True)
