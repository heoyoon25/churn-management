import streamlit as st

st.set_page_config(
    page_title="DCX - 이탈 고객 방지",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 상태 초기화
defaults = {
    'df': None,
    'processed_df': None,
    'X_train': None,
    'X_test': None,
    'y_train': None,
    'y_test': None,
    'selected_features': None,
    'target': None,
    'lr_model': None,
    'dt_model': None,
    'lr_results': None,
    'dt_results': None
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# 사이드바
with st.sidebar:
    st.title("📊 DCX 메뉴")
    st.markdown("---")
    st.page_link("main.py",                label="🏠 메인")
    st.page_link("pages/1_main.py",        label="📂 데이터 업로드")
    st.page_link("pages/2_explore.py",     label="🔍 데이터 탐색")
    st.page_link("pages/3_preprocess.py",  label="⚙️ 데이터 전처리")
    st.page_link("pages/4_model.py",       label="🤖 연구 모형")
    st.page_link("pages/5_results.py",     label="📈 연구 결과")

st.title("📊 DCX - 이탈 고객 방지")
st.markdown("#### 고객 이탈 예측 머신러닝 분석 플랫폼")
st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.info("🔍 **데이터 탐색**\n\n변수 분포 및 관계를 시각화합니다.")
with col2:
    st.success("⚙️ **데이터 전처리**\n\n결측치·이상치 처리 및 피처 선택을 합니다.")
with col3:
    st.warning("🤖 **모델 학습 & 평가**\n\nLogistic Regression, Decision Tree를 비교합니다.")

st.markdown("""
| 페이지 | 설명 |
|--------|------|
| 📂 데이터 업로드 | CSV/Excel 파일 업로드 |
| 🔍 데이터 탐색 | 데이터 시각화 및 탐색 |
| ⚙️ 데이터 전처리 | 전처리 및 Feature Selection |
| 🤖 연구 모형 | 모델 학습 |
| 📈 연구 결과 | 성능 비교 분석 |
""")
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="데이터 업로드", page_icon="📂", layout="wide")

with st.sidebar:
    st.title("📊 DCX 메뉴")
    st.markdown("---")
    st.page_link("main.py",                label="🏠 메인")
    st.page_link("pages/1_main.py",        label="📂 데이터 업로드")
    st.page_link("pages/2_explore.py",     label="🔍 데이터 탐색")
    st.page_link("pages/3_preprocess.py",  label="⚙️ 데이터 전처리")
    st.page_link("pages/4_model.py",       label="🤖 연구 모형")
    st.page_link("pages/5_results.py",     label="📈 연구 결과")

# 세션 상태 초기화
defaults = {
    'df': None, 'processed_df': None,
    'X_train': None, 'X_test': None,
    'y_train': None, 'y_test': None,
    'selected_features': None, 'target': None,
    'lr_model': None, 'dt_model': None,
    'lr_results': None, 'dt_results': None
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

st.title("📂 데이터 업로드")
st.markdown("---")

uploaded_file = st.file_uploader(
    "CSV 또는 Excel 파일을 업로드하세요",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            try:
                df = pd.read_csv(uploaded_file, encoding="utf-8")
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="cp949")
        else:
            df = pd.read_excel(uploaded_file)

        st.session_state.df = df
        st.session_state.processed_df = df.copy()
        st.success(f"✅ 업로드 성공: {uploaded_file.name}")

        c1, c2, c3 = st.columns(3)
        c1.metric("행 수", f"{df.shape[0]:,}")
        c2.metric("열 수", f"{df.shape[1]:,}")
        c3.metric("결측치", f"{df.isnull().sum().sum():,}")

        st.dataframe(df.head(10), use_container_width=True)

        col_info = pd.DataFrame({
            "컬럼명": df.columns,
            "타입": df.dtypes.values,
            "결측치 수": df.isnull().sum().values,
            "고유값 수": df.nunique().values
        })
        st.dataframe(col_info, use_container_width=True)

    except Exception as e:
        st.error(f"❌ 오류: {e}")

else:
    if st.session_state.df is not None:
        st.info("✅ 데이터가 이미 로드되어 있습니다.")
        st.dataframe(st.session_state.df.head(5), use_container_width=True)
    else:
        st.markdown("""
        <div style='text-align:center; padding:60px;
                    border:2px dashed #ccc; border-radius:10px; color:#888;'>
            <h3>파일을 업로드해주세요</h3>
            <p>CSV 또는 Excel 파일 지원</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("샘플 데이터로 시작하기")
    if st.button("샘플 데이터 불러오기", type="primary"):
        np.random.seed(42)
        n = 500
        sample_df = pd.DataFrame({
            "고객ID":       range(1, n+1),
            "나이":         np.random.randint(20, 70, n),
            "성별":         np.random.choice(["남", "여"], n),
            "사용기간(월)": np.random.randint(1, 60, n),
            "월평균요금":   np.random.randint(10000, 100000, n),
            "서비스만족도": np.random.randint(1, 6, n),
            "문의횟수":     np.random.randint(0, 20, n),
            "이탈여부":     np.random.choice([0, 1], n, p=[0.7, 0.3])
        })
        st.session_state.df = sample_df
        st.session_state.processed_df = sample_df.copy()
        st.success("✅ 샘플 데이터 로드 완료!")
        st.rerun()
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

st.set_page_config(page_title="데이터 탐색", page_icon="🔍", layout="wide")

with st.sidebar:
    st.title("📊 DCX 메뉴")
    st.markdown("---")
    st.page_link("main.py",                label="🏠 메인")
    st.page_link("pages/1_main.py",        label="📂 데이터 업로드")
    st.page_link("pages/2_explore.py",     label="🔍 데이터 탐색")
    st.page_link("pages/3_preprocess.py",  label="⚙️ 데이터 전처리")
    st.page_link("pages/4_model.py",       label="🤖 연구 모형")
    st.page_link("pages/5_results.py",     label="📈 연구 결과")

st.title("🔍 데이터 탐색")
st.markdown("---")

if st.session_state.get("df") is None:
    st.warning("⚠️ 먼저 데이터 업로드 페이지에서 데이터를 업로드해주세요.")
    st.stop()

df = st.session_state.df

# 기본 정보
st.subheader("데이터 기본 정보")
c1, c2, c3, c4 = st.columns(4)
c1.metric("행 수", f"{df.shape[0]:,}")
c2.metric("열 수", f"{df.shape[1]:,}")
c3.metric("결측치", f"{df.isnull().sum().sum():,}")
c4.metric("수치형 변수", len(df.select_dtypes(include="number").columns))

st.markdown("---")

# 변수 목록
st.subheader("변수 목록 및 타입")
col_left, col_right = st.columns([1, 2])

with col_left:
    type_df = pd.DataFrame({
        "변수명": df.columns.tolist(),
        "타입": [str(t) for t in df.dtypes.tolist()],
        "결측치": df.isnull().sum().tolist(),
        "고유값": df.nunique().tolist()
    })
    st.dataframe(type_df, use_container_width=True, height=300)

with col_right:
    st.markdown("**기술 통계량 (수치형)**")
    st.dataframe(
        df.describe().T.style.format("{:.2f}"),
        use_container_width=True, height=300
    )

st.markdown("---")

# 시각화
st.subheader("데이터 시각화")
all_cols = df.columns.tolist()
cat_cols = df.select_dtypes(exclude="number").columns.tolist()

col1, col2, col3 = st.columns(3)
with col1:
    x_col = st.selectbox("X축 변수", all_cols)
with col2:
    y_col = st.selectbox("Y축 변수", all_cols, index=min(1, len(all_cols)-1))
with col3:
    chart_type = st.selectbox(
        "그래프 유형",
        ["Histogram", "Box Plot", "Scatter Plot", "Bar Chart", "Line Chart"]
    )

color_col = st.selectbox("색상 구분 변수 (선택)", ["없음"] + all_cols)
color_arg = None if color_col == "없음" else color_col

if st.button("그래프 생성", type="primary"):
    try:
        if chart_type == "Histogram":
            fig = px.histogram(df, x=x_col, color=color_arg,
                               template="plotly_white", opacity=0.75)
        elif chart_type == "Box Plot":
            fig = px.box(df, x=x_col, y=y_col, color=color_arg,
                         template="plotly_white")
        elif chart_type == "Scatter Plot":
            fig = px.scatter(df, x=x_col, y=y_col, color=color_arg,
                             template="plotly_white", opacity=0.7)
        elif chart_type == "Bar Chart":
            if x_col in cat_cols:
                bar_data = df.groupby(x_col)[y_col].mean().reset_index()
                fig = px.bar(bar_data, x=x_col, y=y_col,
                             template="plotly_white")
            else:
                fig = px.bar(df, x=x_col, y=y_col, color=color_arg,
                             template="plotly_white")
        elif chart_type == "Line Chart":
            fig = px.line(df.sort_values(x_col), x=x_col, y=y_col,
                          color=color_arg, template="plotly_white")

        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ 그래프 오류: {e}")

# 상관관계 히트맵
st.markdown("---")
st.subheader("상관관계 히트맵")
num_cols = df.select_dtypes(include="number").columns.tolist()
if len(num_cols) >= 2:
    corr = df[num_cols].corr()
    fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto",
                         color_continuous_scale="RdBu_r",
                         template="plotly_white")
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, use_container_width=True)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="데이터 전처리", page_icon="⚙️", layout="wide")

with st.sidebar:
    st.title("📊 DCX 메뉴")
    st.markdown("---")
    st.page_link("main.py",                label="🏠 메인")
    st.page_link("pages/1_main.py",        label="📂 데이터 업로드")
    st.page_link("pages/2_explore.py",     label="🔍 데이터 탐색")
    st.page_link("pages/3_preprocess.py",  label="⚙️ 데이터 전처리")
    st.page_link("pages/4_model.py",       label="🤖 연구 모형")
    st.page_link("pages/5_results.py",     label="📈 연구 결과")

st.title("⚙️ 데이터 전처리 / Feature Selection / Data Partitioning")
st.markdown("---")

if st.session_state.get("df") is None:
    st.warning("⚠️ 먼저 데이터 업로드 페이지에서 데이터를 업로드해주세요.")
    st.stop()

if st.session_state.get("processed_df") is None:
    st.session_state.processed_df = st.session_state.df.copy()

# ── 1. 데이터 전처리 ──────────────────────────────────
st.subheader("1. 데이터 전처리")
df = st.session_state.processed_df

c1, c2, c3 = st.columns(3)
c1.metric("행 수", df.shape[0])
c2.metric("열 수", df.shape[1])
c3.metric("결측치", df.isnull().sum().sum())

# 결측치
with st.expander("결측치 처리", expanded=True):
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        st.success("✅ 결측치 없음")
    else:
        st.dataframe(pd.DataFrame({
            "결측치 수": missing,
            "비율(%)": (missing/len(df)*100).round(2)
        }), use_container_width=True)

        method = st.radio("처리 방법",
            ["평균값 대체(수치형)", "중앙값 대체(수치형)",
             "최빈값 대체(범주형)", "행 삭제"], horizontal=True)

        if st.button("결측치 처리 실행"):
            tmp = st.session_state.processed_df.copy()
            num_c = tmp.select_dtypes(include="number").columns
            cat_c = tmp.select_dtypes(exclude="number").columns
            if method == "평균값 대체(수치형)":
                tmp[num_c] = tmp[num_c].fillna(tmp[num_c].mean())
            elif method == "중앙값 대체(수치형)":
                tmp[num_c] = tmp[num_c].fillna(tmp[num_c].median())
            elif method == "최빈값 대체(범주형)":
                for col in cat_c:
                    tmp[col] = tmp[col].fillna(tmp[col].mode()[0])
            else:
                tmp = tmp.dropna()
            st.session_state.processed_df = tmp
            st.success("✅ 완료!")
            st.rerun()

# 이상치
with st.expander("이상치 처리 (IQR)", expanded=False):
    num_cols = st.session_state.processed_df.select_dtypes(
        include="number").columns.tolist()
    if not num_cols:
        st.info("수치형 변수 없음")
    else:
        out_cols = st.multiselect("처리할 변수", num_cols, default=num_cols[:3])
        out_method = st.radio("방법",
            ["IQR 클리핑(상·하한 대체)", "IQR 행 삭제"], horizontal=True)

        if out_cols:
            rows = []
            for col in out_cols:
                Q1 = st.session_state.processed_df[col].quantile(0.25)
                Q3 = st.session_state.processed_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lo, hi = Q1-1.5*IQR, Q3+1.5*IQR
                n = ((st.session_state.processed_df[col]<lo) |
                     (st.session_state.processed_df[col]>hi)).sum()
                rows.append({"변수": col, "하한": round(lo,2),
                             "상한": round(hi,2), "이상치 수": n})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        if st.button("이상치 처리 실행"):
            tmp = st.session_state.processed_df.copy()
            for col in out_cols:
                Q1 = tmp[col].quantile(0.25)
                Q3 = tmp[col].quantile(0.75)
                IQR = Q3 - Q1
                lo, hi = Q1-1.5*IQR, Q3+1.5*IQR
                if out_method == "IQR 클리핑(상·하한 대체)":
                    tmp[col] = tmp[col].clip(lo, hi)
                else:
                    tmp = tmp[(tmp[col]>=lo) & (tmp[col]<=hi)]
            st.session_state.processed_df = tmp
            st.success(f"✅ 완료! 행 수: {len(tmp):,}")
            st.rerun()

# 원핫 인코딩
with st.expander("원핫 인코딩", expanded=False):
    cat_cols = st.session_state.processed_df.select_dtypes(
        exclude="number").columns.tolist()
    if not cat_cols:
        st.success("✅ 범주형 변수 없음")
    else:
        enc_cols = st.multiselect("인코딩할 변수", cat_cols, default=cat_cols)
        if st.button("원핫 인코딩 실행"):
            tmp = st.session_state.processed_df.copy()
            tmp = pd.get_dummies(tmp, columns=enc_cols, drop_first=False)
            bool_c = tmp.select_dtypes(include="bool").columns
            tmp[bool_c] = tmp[bool_c].astype(int)
            st.session_state.processed_df = tmp
            st.success(f"✅ 완료! 열 수: {tmp.shape[1]}")
            st.rerun()

if st.button("전처리 초기화 (원본 복원)", type="secondary"):
    st.session_state.processed_df = st.session_state.df.copy()
    st.success("✅ 원본 복원 완료")
    st.rerun()

st.markdown("**현재 데이터 미리보기**")
st.dataframe(st.session_state.processed_df.head(5), use_container_width=True)

# ── 2. Feature Selection ──────────────────────────────
st.markdown("---")
st.subheader("2. Feature Selection")

cur_df   = st.session_state.processed_df
all_cols = cur_df.columns.tolist()

col1, col2 = st.columns(2)
with col1:
    target_col = st.selectbox("종속변수 (Y)", all_cols,
                               index=len(all_cols)-1)
with col2:
    feat_opts = [c for c in all_cols if c != target_col]
    sel_feats = st.multiselect("독립변수 (X)", feat_opts, default=feat_opts)

if sel_feats:
    st.info(f"독립변수 {len(sel_feats)}개 선택됨 | 종속변수: {target_col}")
    val_counts = cur_df[target_col].value_counts().reset_index()
    val_counts.columns = [target_col, "count"]
    fig = px.bar(val_counts, x=target_col, y="count",
                 title=f"종속변수 '{target_col}' 분포",
                 template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# ── 3. Data Partitioning ──────────────────────────────
st.markdown("---")
st.subheader("3. Data Partitioning")

col1, col2 = st.columns(2)
with col1:
    split = st.radio("Train : Test 비율", ["7 : 3", "8 : 2"], horizontal=True)
    test_size = 0.3 if split == "7 : 3" else 0.2
with col2:
    seed = st.number_input("Random Seed", 0, 9999, 42)

train_p = int(split.split(":")[0].strip()) * 10
test_p  = int(split.split(":")[1].strip()) * 10
st.markdown(f"""
<div style='display:flex;height:30px;border-radius:8px;overflow:hidden;margin:10px 0;'>
    <div style='width:{train_p}%;background:#4CAF50;display:flex;
                align-items:center;justify-content:center;
                color:white;font-weight:bold;'>Train {train_p}%</div>
    <div style='width:{test_p}%;background:#F44336;display:flex;
                align-items:center;justify-content:center;
                color:white;font-weight:bold;'>Test {test_p}%</div>
</div>
""", unsafe_allow_html=True)

if st.button("데이터 분할 실행", type="primary",
             disabled=not bool(sel_feats)):
    try:
        X = cur_df[sel_feats]
        y = cur_df[target_col]
        non_num = X.select_dtypes(exclude="number").columns.tolist()
        if non_num:
            st.error(f"수치형이 아닌 변수: {non_num} → 원핫 인코딩 필요")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=seed, stratify=y)
            st.session_state.X_train = X_train
            st.session_state.X_test  = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test  = y_test
            st.session_state.selected_features = sel_feats
            st.session_state.target  = target_col
            st.success("✅ 분할 완료!")
            r1,r2,r3,r4 = st.columns(4)
            r1.metric("Train", f"{len(X_train):,}")
            r2.metric("Test",  f"{len(X_test):,}")
            r3.metric("Train%", f"{len(X_train)/len(X)*100:.1f}%")
            r4.metric("Test%",  f"{len(X_test)/len(X)*100:.1f}%")
    except Exception as e:
        st.error(f"❌ 오류: {e}")

if st.session_state.get("X_train") is not None:
    st.success(
        f"✅ 분할 완료 | Train: {len(st.session_state.X_train):,} / "
        f"Test: {len(st.session_state.X_test):,}"
    )
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

st.set_page_config(page_title="연구 모형", page_icon="🤖", layout="wide")

with st.sidebar:
    st.title("📊 DCX 메뉴")
    st.markdown("---")
    st.page_link("main.py",                label="🏠 메인")
    st.page_link("pages/1_main.py",        label="📂 데이터 업로드")
    st.page_link("pages/2_explore.py",     label="🔍 데이터 탐색")
    st.page_link("pages/3_preprocess.py",  label="⚙️ 데이터 전처리")
    st.page_link("pages/4_model.py",       label="🤖 연구 모형")
    st.page_link("pages/5_results.py",     label="📈 연구 결과")

st.title("🤖 연구 모형")
st.markdown("---")

if st.session_state.get("X_train") is None:
    st.warning("⚠️ 먼저 데이터 전처리 페이지에서 데이터 분할을 완료해주세요.")
    st.stop()

X_train = st.session_state.X_train
X_test  = st.session_state.X_test
y_train = st.session_state.y_train
y_test  = st.session_state.y_test

# ── Logistic Regression ───────────────────────────────
st.subheader("1. Logistic Regression")

with st.expander("하이퍼파라미터 설정", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        lr_C = st.select_slider("규제 강도 C",
            options=[0.001,0.01,0.1,1.0,10.0,100.0], value=1.0)
    with c2:
        lr_iter = st.slider("최대 반복 횟수", 100, 2000, 1000, 100)
    with c3:
        lr_solver = st.selectbox("Solver", ["lbfgs","liblinear","saga"])

if st.button("Logistic Regression 학습", type="primary", key="btn_lr"):
    with st.spinner("학습 중..."):
        try:
            model = LogisticRegression(C=lr_C, max_iter=lr_iter,
                                       solver=lr_solver, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            st.session_state.lr_model = model
            st.session_state.lr_results = {
                "accuracy":  accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred,
                                             average="weighted", zero_division=0),
                "recall":    recall_score(y_test, y_pred,
                                          average="weighted", zero_division=0),
                "f1":        f1_score(y_test, y_pred,
                                      average="weighted", zero_division=0),
                "y_pred": y_pred, "y_prob": y_prob,
                "cm": confusion_matrix(y_test, y_pred)
            }
            st.success("✅ 학습 완료!")
        except Exception as e:
            st.error(f"❌ {e}")

if st.session_state.get("lr_results"):
    r = st.session_state.lr_results
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Accuracy",  f"{r['accuracy']:.4f}")
    m2.metric("Precision", f"{r['precision']:.4f}")
    m3.metric("Recall",    f"{r['recall']:.4f}")
    m4.metric("F1-Score",  f"{r['f1']:.4f}")

    col1, col2 = st.columns(2)
    with col1:
        labels = [str(c) for c in sorted(y_test.unique())]
        fig_cm = ff.create_annotated_heatmap(
            r["cm"], x=labels, y=labels,
            colorscale="Blues", showscale=True)
        fig_cm.update_layout(title="Confusion Matrix - LR",
                             xaxis_title="예측", yaxis_title="실제", height=350)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        m = st.session_state.lr_model
        if len(m.classes_) == 2:
            coef_df = pd.DataFrame({
                "변수": X_train.columns,
                "계수": m.coef_[0]
            }).sort_values("계수", key=abs, ascending=False)
            fig = px.bar(coef_df, x="계수", y="변수", orientation="h",
                         color="계수", color_continuous_scale="RdBu",
                         template="plotly_white",
                         title="LR 계수 (변수 중요도)")
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

# ── Decision Tree ─────────────────────────────────────
st.markdown("---")
st.subheader("2. Decision Tree")

with st.expander("하이퍼파라미터 설정", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        dt_depth = st.slider("최대 깊이", 1, 20, 5)
    with c2:
        dt_min   = st.slider("최소 분할 샘플 수", 2, 50, 2)
    with c3:
        dt_crit  = st.selectbox("분할 기준", ["gini","entropy"])

if st.button("Decision Tree 학습", type="primary", key="btn_dt"):
    with st.spinner("학습 중..."):
        try:
            model = DecisionTreeClassifier(
                max_depth=dt_depth, min_samples_split=dt_min,
                criterion=dt_crit, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            st.session_state.dt_model = model
            st.session_state.dt_results = {
                "accuracy":  accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred,
                                             average="weighted", zero_division=0),
                "recall":    recall_score(y_test, y_pred,
                                          average="weighted", zero_division=0),
                "f1":        f1_score(y_test, y_pred,
                                      average="weighted", zero_division=0),
                "y_pred": y_pred, "y_prob": y_prob,
                "cm": confusion_matrix(y_test, y_pred)
            }
            st.success("✅ 학습 완료!")
        except Exception as e:
            st.error(f"❌ {e}")

if st.session_state.get("dt_results"):
    r = st.session_state.dt_results
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Accuracy",  f"{r['accuracy']:.4f}")
    m2.metric("Precision", f"{r['precision']:.4f}")
    m3.metric("Recall",    f"{r['recall']:.4f}")
    m4.metric("F1-Score",  f"{r['f1']:.4f}")

    col1, col2 = st.columns(2)
    with col1:
        labels = [str(c) for c in sorted(y_test.unique())]
        fig_cm = ff.create_annotated_heatmap(
            r["cm"], x=labels, y=labels,
            colorscale="Greens", showscale=True)
        fig_cm.update_layout(title="Confusion Matrix - DT",
                             xaxis_title="예측", yaxis_title="실제", height=350)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        m = st.session_state.dt_model
        fi_df = pd.DataFrame({
            "변수": X_train.columns,
            "중요도": m.feature_importances_
        }).sort_values("중요도", ascending=False).head(15)
        fig = px.bar(fi_df, x="중요도", y="변수", orientation="h",
                     color="중요도", color_continuous_scale="Greens",
                     template="plotly_white", title="DT Feature Importance")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Decision Tree 시각화**")
    vis_depth = st.slider("시각화 깊이", 1, min(5, dt_depth), 3)
    fig_tree, ax = plt.subplots(figsize=(20, 8))
    plot_tree(st.session_state.dt_model,
              feature_names=X_train.columns.tolist(),
              class_names=[str(c) for c in sorted(y_test.unique())],
              filled=True, rounded=True, fontsize=9,
              max_depth=vis_depth, ax=ax)
    st.pyplot(fig_tree)
    plt.close()
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_curve, auc

st.set_page_config(page_title="연구 결과", page_icon="📈", layout="wide")

with st.sidebar:
    st.title("📊 DCX 메뉴")
    st.markdown("---")
    st.page_link("main.py",                label="🏠 메인")
    st.page_link("pages/1_main.py",        label="📂 데이터 업로드")
    st.page_link("pages/2_explore.py",     label="🔍 데이터 탐색")
    st.page_link("pages/3_preprocess.py",  label="⚙️ 데이터 전처리")
    st.page_link("pages/4_model.py",       label="🤖 연구 모형")
    st.page_link("pages/5_results.py",     label="📈 연구 결과")

st.title("📈 연구 결과")
st.markdown("---")

lr_done = st.session_state.get("lr_results") is not None
dt_done = st.session_state.get("dt_results") is not None

if not lr_done and not dt_done:
    st.warning("⚠️ 먼저 연구 모형 페이지에서 모델을 학습해주세요.")
    st.stop()

y_test = st.session_state.y_test

# ── 성능 지표 비교 ────────────────────────────────────
st.subheader("1. 모델 성능 지표 비교")

rows = []
if lr_done:
    r = st.session_state.lr_results
    rows.append({"모델":"Logistic Regression",
                 "Accuracy": round(r["accuracy"],4),
                 "Precision":round(r["precision"],4),
                 "Recall":   round(r["recall"],4),
                 "F1-Score": round(r["f1"],4)})
if dt_done:
    r = st.session_state.dt_results
    rows.append({"모델":"Decision Tree",
                 "Accuracy": round(r["accuracy"],4),
                 "Precision":round(r["precision"],4),
                 "Recall":   round(r["recall"],4),
                 "F1-Score": round(r["f1"],4)})

metrics_df = pd.DataFrame(rows).set_index("모델")

def highlight_max(s):
    return ["background-color:#d4edda;font-weight:bold"
            if v == s.max() else "" for v in s]

st.dataframe(
    metrics_df.style.apply(highlight_max).format("{:.4f}"),
    use_container_width=True
)

# 레이더 차트
st.markdown("**성능 지표 레이더 차트**")
cats = ["Accuracy","Precision","Recall","F1-Score"]
fig_radar = go.Figure()
colors = ["#1f77b4","#2ca02c"]
for i, (idx, row) in enumerate(metrics_df.iterrows()):
    vals = row[cats].tolist()
    fig_radar.add_trace(go.Scatterpolar(
        r=vals+[vals[0]], theta=cats+[cats[0]],
        fill="toself", name=idx, line_color=colors[i]
    ))
fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0,1])),
    height=450, template="plotly_white",
    title="모델 성능 비교 레이더 차트"
)
st.plotly_chart(fig_radar, use_container_width=True)

# 막대 그래프
melted = metrics_df.reset_index().melt(
    id_vars="모델", var_name="지표", value_name="값")
fig_bar = px.bar(melted, x="지표", y="값", color="모델",
                 barmode="group", template="plotly_white",
                 color_discrete_sequence=["#1f77b4","#2ca02c"],
                 text_auto=".4f", title="지표별 비교")
fig_bar.update_layout(height=400, yaxis_range=[0,1.1])
fig_bar.update_traces(textposition="outside")
st.plotly_chart(fig_bar, use_container_width=True)

# ── ROC Curve ─────────────────────────────────────────
st.markdown("---")
st.subheader("2. ROC Curve 비교")

if len(y_test.unique()) != 2:
    st.warning("ROC Curve는 이진 분류에서만 지원됩니다.")
else:
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=[0,1], y=[0,1], mode="lines", name="Random (AUC=0.50)",
        line=dict(dash="dash", color="gray", width=1.5)
    ))

    auc_rows = []
    if lr_done:
        fpr, tpr, _ = roc_curve(y_test,
                                 st.session_state.lr_results["y_prob"])
        roc_auc = auc(fpr, tpr)
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"Logistic Regression (AUC={roc_auc:.4f})",
            line=dict(color="#1f77b4", width=2.5)
        ))
        auc_rows.append({"모델":"Logistic Regression","AUC":round(roc_auc,4)})

    if dt_done:
        fpr, tpr, _ = roc_curve(y_test,
                                 st.session_state.dt_results["y_prob"])
        roc_auc = auc(fpr, tpr)
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"Decision Tree (AUC={roc_auc:.4f})",
            line=dict(color="#2ca02c", width=2.5)
        ))
        auc_rows.append({"모델":"Decision Tree","AUC":round(roc_auc,4)})

    fig_roc.update_layout(
        title="ROC Curve 비교",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white", height=500,
        xaxis=dict(range=[0,1]), yaxis=dict(range=[0,1.02])
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    auc_df = pd.DataFrame(auc_rows).set_index("모델")
    st.dataframe(
        auc_df.style.highlight_max(color="#d4edda").format("{:.4f}"),
        use_container_width=True
    )

# ── 종합 요약 ─────────────────────────────────────────
st.markdown("---")
st.subheader("3. 종합 분석 요약")

if len(rows) == 2:
    best = metrics_df["F1-Score"].idxmax()
    st.markdown(f"""
    <div style='background:#f0f8ff;padding:20px;
                border-radius:10px;border-left:5px solid #1f77b4;'>
        <h4>🏆 분석 결과 요약</h4>
        <ul>
            <li>최고 Accuracy: <b>{metrics_df['Accuracy'].idxmax()}</b>
                ({metrics_df['Accuracy'].max():.4f})</li>
            <li>최고 F1-Score: <b>{best}</b>
                ({metrics_df['F1-Score'].max():.4f})</li>
            <li>권장 모델: <span style='color:#e74c3c;font-weight:bold;'>
                {best}</span></li>
        </ul>
        <p style='color:#666;font-size:0.9em;'>
        ※ 이탈 고객 예측에서는 Recall이 높은 모델이 실제 이탈 고객을
        더 잘 탐지합니다.
        </p>
    </div>
    """, unsafe_allow_html=True)

# 다운로드
st.markdown("---")
st.subheader("예측 결과 다운로드")
result_df = pd.DataFrame({"실제값": y_test.values})
if lr_done:
    result_df["LR_예측값"]  = st.session_state.lr_results["y_pred"]
    result_df["LR_예측확률"] = st.session_state.lr_results["y_prob"].round(4)
if dt_done:
    result_df["DT_예측값"]  = st.session_state.dt_results["y_pred"]
    result_df["DT_예측확률"] = st.session_state.dt_results["y_prob"].round(4)

st.dataframe(result_df.head(10), use_container_width=True)
csv = result_df.to_csv(index=False, encoding="utf-8-sig")
st.download_button("📥 CSV 다운로드", data=csv,
                   file_name="churn_results.csv",
                   mime="text/csv", type="primary")
