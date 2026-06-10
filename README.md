# 🧠 학생 정신건강 분석 앱

학생의 스트레스·불안·우울 수준을 입력받아 AI KMeans 모델로 정신건강 군집을 분류하고, 맞춤형 도움말을 제공하는 Streamlit 웹 앱입니다.

---

## 📁 파일 구조

```
student_mental_health_app/
├── app.py                        # 메인 Streamlit 앱
├── 학생정신건강_분析모델.pkl        # 학습된 KMeans 모델
├── requirements.txt              # Python 패키지 목록
└── README.md
```

---

## 🚀 Streamlit Cloud 배포 방법

### 1단계 — GitHub 레포 준비

1. GitHub에서 새 레포지터리 생성 (예: `student-mental-health`)
2. 이 폴더의 파일을 모두 업로드:
   - `app.py`
   - `requirements.txt`
   - `학생정신건강_분析모델.pkl`
   - `README.md`

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_ID/student-mental-health.git
git push -u origin main
```

### 2단계 — Streamlit Cloud 배포

1. [share.streamlit.io](https://share.streamlit.io) 접속
2. **New app** 클릭
3. GitHub 레포 연결:
   - Repository: `YOUR_ID/student-mental-health`
   - Branch: `main`
   - Main file path: `app.py`
4. **Deploy** 클릭 → 자동 빌드 후 URL 생성

---

## 💻 로컬 실행 방법

```bash
pip install -r requirements.txt
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속

---

## 🤖 모델 정보

| 항목 | 내용 |
|------|------|
| 알고리즘 | KMeans (sklearn 1.6.1) |
| 클러스터 수 | 3개 |
| 입력 피처 | 스트레스, 불안감, 우울감 (1~10점) |
| 군집 결과 | 정서적 어려움 주의군 / 스트레스 주의군 / 안정군 |

---

## ⚠️ 주의사항

본 앱의 분석 결과는 **참고용 정보**이며, 전문적인 의학·심리 진단을 대체하지 않습니다.
