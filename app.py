import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# ── 페이지 설정 ──────────────────────────────────────────────
st.set_page_config(
    page_title="학생 스트레스 분석기",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 커스텀 CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700;900&family=Space+Grotesk:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Noto Sans KR', sans-serif;
}

/* 배경 */
.stApp {
    background: #0D0F1A;
    color: #E8E8F0;
}

/* 사이드바 */
section[data-testid="stSidebar"] {
    background: #13162A;
    border-right: 1px solid #1E2240;
}
section[data-testid="stSidebar"] * {
    color: #C8C8E0 !important;
}

/* 슬라이더 레이블 */
.stSlider label { color: #A0A0C0 !important; font-size: 0.85rem !important; }
.stSlider [data-baseweb="slider"] { margin-top: 4px; }

/* 헤더 */
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #FFFFFF;
    margin-bottom: 0;
    line-height: 1.1;
}
.hero-sub {
    font-size: 0.95rem;
    color: #6B6B9A;
    margin-top: 6px;
    margin-bottom: 32px;
}

/* 상태 카드 */
.status-card {
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    margin-bottom: 20px;
}
.status-good  { background: linear-gradient(135deg, #0A2A1F 0%, #0D3D2A 100%); border: 1.5px solid #1A6644; }
.status-warn  { background: linear-gradient(135deg, #2A1F00 0%, #3D3000 100%); border: 1.5px solid #806020; }
.status-danger{ background: linear-gradient(135deg, #2A0A0A 0%, #3D0D0D 100%); border: 1.5px solid #802020; }

.status-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.status-score {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 4.2rem;
    font-weight: 700;
    letter-spacing: -0.04em;
    line-height: 1;
}
.status-name {
    font-size: 1.1rem;
    font-weight: 500;
    margin-top: 10px;
}
.status-emoji { font-size: 2rem; margin-bottom: 6px; }

/* 조언 박스 */
.advice-box {
    background: #13162A;
    border-radius: 12px;
    padding: 18px 22px;
    margin-top: 8px;
    border-left: 3px solid;
    font-size: 0.88rem;
    line-height: 1.7;
    color: #C0C0D8;
}

/* 지표 카드 */
.metric-row { display: flex; gap: 14px; margin-bottom: 14px; }
.metric-card {
    background: #13162A;
    border: 1px solid #1E2240;
    border-radius: 12px;
    padding: 16px 20px;
    flex: 1;
    min-width: 0;
}
.metric-card-label { font-size: 0.78rem; color: #6B6B9A; margin-bottom: 4px; }
.metric-card-value { font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 600; }

/* 섹션 타이틀 */
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #4A4A7A;
    margin-bottom: 16px;
    margin-top: 24px;
}

/* Plotly 컨테이너 */
.plot-container { border-radius: 14px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ── 스트레스 계산 함수 ───────────────────────────────────────
def calc_stress(sns_h, study_h, exercise_h):
    """
    SNS 사용 많을수록 + 공부시간 극단값일수록 + 운동 적을수록 스트레스 증가.
    0 ~ 100 점 반환.
    """
    # SNS: 0시간=0점, 8시간+=40점 (선형 클램프)
    sns_score = min(sns_h / 8 * 40, 40)

    # 공부: 0시간=20점 패널티, 4~6시간=0점, 10시간+=30점
    if study_h < 2:
        study_score = 20
    elif study_h <= 6:
        study_score = max(0, (6 - study_h) / 4 * 15)
    else:
        study_score = min((study_h - 6) / 4 * 30, 30)

    # 운동: 0시간=30점, 2시간+=0점
    exercise_score = max(0, (2 - exercise_h) / 2 * 30)

    raw = sns_score + study_score + exercise_score
    return round(min(raw, 100))


def get_level(score):
    if score < 40:
        return "양호", "#22C97A", "status-good", "😊", "good"
    elif score < 70:
        return "주의", "#F0B429", "status-warn", "😟", "warn"
    else:
        return "위험", "#F05252", "status-danger", "😰", "danger"


def get_advice(score, sns_h, study_h, exercise_h, level_key):
    advices = []
    if sns_h > 4:
        advices.append(f"📱 SNS 사용시간이 {sns_h}시간으로 길어요. 하루 2시간 이하를 목표로 줄여보세요.")
    if study_h > 7:
        advices.append(f"📚 공부시간이 {study_h}시간으로 과도할 수 있어요. 뽀모도로 기법으로 집중+휴식을 반복하세요.")
    if study_h < 2:
        advices.append("📖 공부시간이 너무 짧아요. 작은 목표부터 시작해 집중 시간을 늘려보세요.")
    if exercise_h < 0.5:
        advices.append("🏃 운동을 거의 하지 않고 있어요. 하루 30분 산책만으로도 스트레스가 크게 줄어요.")

    if level_key == "good":
        advices.append("✅ 현재 생활 패턴을 잘 유지하고 있어요. 꾸준히 지속해보세요!")
    elif level_key == "warn":
        advices.append("⚠️ 스트레스가 쌓이고 있어요. 수면 7~8시간 확보와 취미 활동을 더해보세요.")
    else:
        advices.append("🚨 번아웃 위험 신호예요. 지금 당장 휴식을 취하고 신뢰할 수 있는 사람과 이야기해보세요.")

    return advices


# ── 사이드바: 입력 ───────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ 오늘 나의 하루")
    st.markdown("---")

    sns_h = st.slider(
        "📱 SNS 사용시간",
        min_value=0.0, max_value=12.0, value=3.0, step=0.5,
        format="%.1f시간"
    )
    study_h = st.slider(
        "📚 공부시간",
        min_value=0.0, max_value=14.0, value=5.0, step=0.5,
        format="%.1f시간"
    )
    exercise_h = st.slider(
        "🏃 운동시간",
        min_value=0.0, max_value=4.0, value=0.5, step=0.25,
        format="%.2f시간"
    )

    total_h = sns_h + study_h + exercise_h
    st.markdown("---")
    st.caption(f"입력된 총 시간: **{total_h:.1f}시간** / 24시간")

    st.markdown("---")
    st.caption("ℹ️ 이 도구는 참고용입니다. 심각한 스트레스는 전문가 상담을 권장합니다.")


# ── 메인 영역 ────────────────────────────────────────────────
score = calc_stress(sns_h, study_h, exercise_h)
level_name, level_color, level_class, level_emoji, level_key = get_level(score)
advices = get_advice(score, sns_h, study_h, exercise_h, level_key)

# 헤더
st.markdown('<div class="hero-title">학생 스트레스 분석기</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">SNS · 공부 · 운동 패턴으로 오늘의 스트레스를 진단해보세요</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1.6], gap="large")

# ── 왼쪽: 상태 카드 + 조언 ───────────────────────────────────
with col_left:
    st.markdown(
        f"""
        <div class="status-card {level_class}">
            <div class="status-emoji">{level_emoji}</div>
            <div class="status-label" style="color:{level_color}">스트레스 지수</div>
            <div class="status-score" style="color:{level_color}">{score}</div>
            <div class="status-name" style="color:{level_color}">{level_name}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    border_colors = {"good": "#1A6644", "warn": "#806020", "danger": "#802020"}
    for adv in advices:
        st.markdown(
            f'<div class="advice-box" style="border-color:{border_colors[level_key]}">{adv}</div>',
            unsafe_allow_html=True,
        )

    # 입력 요약 미니 카드
    st.markdown('<div class="section-title">오늘 입력값 요약</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("📱 SNS", f"{sns_h}h")
    with c2:
        st.metric("📚 공부", f"{study_h}h")
    with c3:
        st.metric("🏃 운동", f"{exercise_h}h")


# ── 오른쪽: 차트들 ──────────────────────────────────────────
with col_right:
    # 1) 게이지 차트
    st.markdown('<div class="section-title">스트레스 게이지</div>', unsafe_allow_html=True)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        number={"font": {"size": 52, "color": level_color, "family": "Space Grotesk"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#2A2A4A",
                     "tickfont": {"color": "#4A4A7A", "size": 11}},
            "bar": {"color": level_color, "thickness": 0.22},
            "bgcolor": "#13162A",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40],  "color": "#0A2A1F"},
                {"range": [40, 70], "color": "#2A1F00"},
                {"range": [70, 100],"color": "#2A0A0A"},
            ],
            "threshold": {
                "line": {"color": "#FFFFFF", "width": 2},
                "thickness": 0.75,
                "value": score,
            },
        },
    ))
    fig_gauge.update_layout(
        height=230,
        margin=dict(l=24, r=24, t=16, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#E8E8F0",
    )
    st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

    # 2) 레이더 차트 – 생활 균형
    st.markdown('<div class="section-title">생활 균형 레이더</div>', unsafe_allow_html=True)

    # 정규화 (각 항목의 '이상적' 기준 대비 점수)
    sns_norm   = max(0, 100 - min(sns_h / 8 * 100, 100))    # 적을수록 좋음
    study_norm = 100 - abs(study_h - 5) / 5 * 60            # 4~6h 최적
    study_norm = max(0, min(study_norm, 100))
    exer_norm  = min(exercise_h / 2 * 100, 100)             # 많을수록 좋음
    sleep_est  = max(0, min((24 - sns_h - study_h - exercise_h) * 0.45, 100))  # 추정 여유 시간

    categories = ["SNS 절제", "공부 균형", "운동", "여유 시간"]
    values     = [sns_norm, study_norm, exer_norm, sleep_est]

    fig_radar = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor=f"rgba({int(level_color[1:3],16)},{int(level_color[3:5],16)},{int(level_color[5:7],16)},0.18)",
        line=dict(color=level_color, width=2),
        marker=dict(color=level_color, size=6),
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="#13162A",
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color="#4A4A7A", size=10),
                            gridcolor="#1E2240", linecolor="#1E2240"),
            angularaxis=dict(tickfont=dict(color="#A0A0C0", size=12), gridcolor="#1E2240", linecolor="#1E2240"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=16, b=16),
        height=280,
        showlegend=False,
    )
    st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})


# ── 하단: 시뮬레이션 히트맵 ─────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">SNS vs 운동 — 스트레스 지형도 (공부시간 고정: {:.0f}h)'.format(study_h) + '</div>', unsafe_allow_html=True)

sns_vals  = np.arange(0, 12.5, 0.5)
exer_vals = np.arange(0, 4.25, 0.25)
Z = np.array([[calc_stress(s, study_h, e) for s in sns_vals] for e in exer_vals])

fig_heat = go.Figure(go.Heatmap(
    z=Z,
    x=sns_vals,
    y=exer_vals,
    colorscale=[
        [0.0,  "#0A2A1F"],
        [0.4,  "#22C97A"],
        [0.4,  "#2A1F00"],
        [0.7,  "#F0B429"],
        [0.7,  "#2A0A0A"],
        [1.0,  "#F05252"],
    ],
    zmin=0, zmax=100,
    colorbar=dict(
        title="스트레스",
        titlefont=dict(color="#A0A0C0"),
        tickfont=dict(color="#A0A0C0"),
        tickvals=[20, 55, 85],
        ticktext=["양호", "주의", "위험"],
        bgcolor="#13162A",
        bordercolor="#1E2240",
    ),
    hovertemplate="SNS %{x}h · 운동 %{y}h<br>스트레스: %{z}<extra></extra>",
))

# 현재 입력값 마커
fig_heat.add_trace(go.Scatter(
    x=[sns_h], y=[min(exercise_h, 4.0)],
    mode="markers",
    marker=dict(size=14, color="#FFFFFF", symbol="circle-open", line=dict(width=2.5, color="#FFFFFF")),
    name="현재 위치",
    hovertemplate=f"📍 현재 위치<br>SNS {sns_h}h · 운동 {exercise_h}h<extra></extra>",
))

fig_heat.update_layout(
    xaxis=dict(title="SNS 사용시간 (h)", title_font_color="#A0A0C0", tickfont=dict(color="#6B6B9A"),
               gridcolor="#1E2240", zerolinecolor="#1E2240"),
    yaxis=dict(title="운동시간 (h)", title_font_color="#A0A0C0", tickfont=dict(color="#6B6B9A"),
               gridcolor="#1E2240", zerolinecolor="#1E2240"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0D0F1A",
    margin=dict(l=20, r=20, t=10, b=20),
    height=340,
    legend=dict(font=dict(color="#A0A0C0"), bgcolor="rgba(0,0,0,0)"),
)

st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

st.markdown(
    "<div style='text-align:center;color:#3A3A6A;font-size:0.78rem;margin-top:24px;'>"
    "⬤ 흰 원이 현재 입력값 위치 &nbsp;|&nbsp; 초록=양호(0~39) · 노랑=주의(40~69) · 빨강=위험(70~100)"
    "</div>",
    unsafe_allow_html=True,
)
