# 이모지 제거한 안전한 파일명
pages/
├── 1_main.py
├── 2_data_explore.py
├── 3_preprocessing.py
├── 4_model.py
└── 5_results.py

import streamlit as st

st.set_page_config(
    page_title="DCX - 이탈 고객 방지",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 상태 초기화
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = None
if 'target' not in st.session_state:
    st.session_state.target = None
if 'lr_model' not in st.session_state:
    st.session_state.lr_model = None
if 'dt_model' not in st.session_state:
    st.session_state.dt_model = None
if 'lr_results' not in st.session_state:
    st.session_state.lr_results = None
if 'dt_results' not in st.session_state:
    st.session_state.dt_results = None

st.title("📊 DCX - 이탈 고객 방지")
st.markdown("---")
st.markdown("""
### 👋 환영합니다!
이 시스템은 고객 이탈을 예측하고 방지하기 위한 머신러닝 분석 플랫폼입니다.

**사이드바 메뉴**를 통해 각 페이지로 이동하세요:

| 페이지 | 설명 |
|--------|------|
| 🏠 메인 | 데이터 업로드 |
| 🔍 데이터 탐색 | 데이터 시각화 및 탐색 |
| ⚙️ 데이터 전처리 | 전처리 및 Feature Selection |
| 🤖 연구 모형 | 모델 학습 |
| 📈 연구 결과 | 성능 비교 분석 |
""")
import streamlit as st
import pandas as pd

st.set_page_config(page_title="메인", page_icon="🏠", layout="wide")

# 세션 상태 초기화 보장
for key, default in {
    'df': None, 'processed_df': None,
    'X_train': None, 'X_test': None,
    'y_train': None, 'y_test': None,
    'selected_features': None, 'target': None,
    'lr_model': None, 'dt_model': None,
    'lr_results': None, 'dt_results': None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── 헤더 ──────────────────────────────────────────────
st.title("📊 DCX - 이탈 고객 방지")
st.markdown("#### 고객 이탈 예측 머신러닝 분석 플랫폼")
st.markdown("---")

# ── 소개 카드 ─────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    st.info("🔍 **데이터 탐색**\n\n변수 분포 및 관계를 시각화합니다.")
with col2:
    st.success("⚙️ **데이터 전처리**\n\n결측치·이상치 처리 및 피처 선택을 합니다.")
with col3:
    st.warning("🤖 **모델 학습 & 평가**\n\nLogistic Regression, Decision Tree를 비교합니다.")

st.markdown("---")

# ── 데이터 업로드 ─────────────────────────────────────
st.subheader("📂 데이터 업로드")
st.markdown("분석할 CSV 또는 Excel 파일을 업로드하세요.")

uploaded_file = st.file_uploader(
    "파일을 드래그하거나 클릭하여 업로드",
    type=["csv", "xlsx", "xls"],
    help="CSV 또는 Excel 형식의 파일을 지원합니다."
)

if uploaded_file is not None:
    try:
        # 파일 형식에 따라 읽기
        if uploaded_file.name.endswith(".csv"):
            # 인코딩 자동 감지
            try:
                df = pd.read_csv(uploaded_file, encoding="utf-8")
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="cp949")
        else:
            df = pd.read_excel(uploaded_file)

        st.session_state.df = df
        st.session_state.processed_df = df.copy()  # 전처리용 복사본

        st.success(f"✅ 파일 업로드 성공! **{uploaded_file.name}**")

        # ── 데이터 미리보기 ───────────────────────────
        st.markdown("---")
        st.subheader("📋 데이터 미리보기")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("📏 행 수", f"{df.shape[0]:,}")
        col_b.metric("📐 열 수", f"{df.shape[1]:,}")
        col_c.metric("❓ 결측치 수", f"{df.isnull().sum().sum():,}")

        st.dataframe(df.head(10), use_container_width=True)

        # ── 컬럼 정보 ─────────────────────────────────
        st.subheader("🗂️ 컬럼 정보")
        col_info = pd.DataFrame({
            "컬럼명": df.columns,
            "데이터 타입": df.dtypes.values,
            "결측치 수": df.isnull().sum().values,
            "결측치 비율(%)": (df.isnull().sum().values / len(df) * 100).round(2),
            "고유값 수": df.nunique().values
        })
        st.dataframe(col_info, use_container_width=True)

    except Exception as e:
        st.error(f"❌ 파일 읽기 오류: {e}")

else:
    # 업로드 전 안내
    if st.session_state.df is not None:
        st.info("✅ 이미 데이터가 로드되어 있습니다. 다른 파일을 업로드하면 교체됩니다.")
        st.dataframe(st.session_state.df.head(5), use_container_width=True)
    else:
        st.markdown("""
        <div style='text-align:center; padding:60px; 
                    border:2px dashed #ccc; border-radius:10px; color:#888;'>
            <h3>📁 파일을 업로드해주세요</h3>
            <p>CSV 또는 Excel 파일을 지원합니다</p>
        </div>
        """, unsafe_allow_html=True)

        # ── 샘플 데이터 제공 ──────────────────────────
        st.markdown("---")
        st.subheader("🧪 샘플 데이터로 시작하기")
        if st.button("📥 샘플 데이터 불러오기", type="primary"):
            import numpy as np
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
            st.success("✅ 샘플 데이터가 로드되었습니다!")
            st.rerun()
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="데이터 탐색", page_icon="🔍", layout="wide")

st.title("🔍 데이터 탐색")
st.markdown("---")

# ── 데이터 확인 ───────────────────────────────────────
if st.session_state.get("df") is None:
    st.warning("⚠️ 먼저 **메인 페이지**에서 데이터를 업로드해주세요.")
    st.stop()

df = st.session_state.df

# ── 1. 기본 정보 ──────────────────────────────────────
st.subheader("📊 데이터 기본 정보")

col1, col2, col3, col4 = st.columns(4)
col1.metric("📏 행 수",    f"{df.shape[0]:,}")
col2.metric("📐 열 수",    f"{df.shape[1]:,}")
col3.metric("❓ 결측치",   f"{df.isnull().sum().sum():,}")
col4.metric("🔢 수치형 변수", f"{len(df.select_dtypes(include='number').columns)}")

# ── 2. 변수 목록 & 타입 ───────────────────────────────
st.markdown("---")
st.subheader("🗂️ 변수 목록 및 타입")

col_left, col_right = st.columns([1, 2])

with col_left:
    type_df = pd.DataFrame({
        "변수명":      df.columns.tolist(),
        "타입":        [str(t) for t in df.dtypes.tolist()],
        "결측치 수":   df.isnull().sum().tolist(),
        "고유값 수":   df.nunique().tolist()
    })
    st.dataframe(type_df, use_container_width=True, height=300)

with col_right:
    st.markdown("**📈 기술 통계량 (수치형 변수)**")
    st.dataframe(
        df.describe().T.style.format("{:.2f}"),
        use_container_width=True,
        height=300
    )

# ── 3. 시각화 ─────────────────────────────────────────
st.markdown("---")
st.subheader("📉 데이터 시각화")

all_cols  = df.columns.tolist()
num_cols  = df.select_dtypes(include="number").columns.tolist()
cat_cols  = df.select_dtypes(exclude="number").columns.tolist()

col_opt1, col_opt2, col_opt3 = st.columns(3)

with col_opt1:
    x_col = st.selectbox("📌 X축 변수", all_cols, key="x_col")
with col_opt2:
    y_col = st.selectbox("📌 Y축 변수", all_cols,
                         index=min(1, len(all_cols)-1), key="y_col")
with col_opt3:
    chart_type = st.selectbox(
        "📊 그래프 유형",
        ["Histogram", "Box Plot", "Scatter Plot", "Bar Chart", "Line Chart"],
        key="chart_type"
    )

# 색상 구분 변수 (선택)
color_col = st.selectbox(
    "🎨 색상 구분 변수 (선택)",
    ["없음"] + all_cols,
    key="color_col"
)
color_arg = None if color_col == "없음" else color_col

# ── 그래프 생성 ───────────────────────────────────────
if st.button("📊 그래프 생성", type="primary"):
    try:
        fig = None

        if chart_type == "Histogram":
            fig = px.histogram(
                df, x=x_col, color=color_arg,
                title=f"Histogram - {x_col}",
                template="plotly_white",
                barmode="overlay", opacity=0.75
            )

        elif chart_type == "Box Plot":
            fig = px.box(
                df, x=x_col, y=y_col, color=color_arg,
                title=f"Box Plot - {x_col} vs {y_col}",
                template="plotly_white"
            )

        elif chart_type == "Scatter Plot":
            fig = px.scatter(
                df, x=x_col, y=y_col, color=color_arg,
                title=f"Scatter Plot - {x_col} vs {y_col}",
                template="plotly_white", opacity=0.7
            )

        elif chart_type == "Bar Chart":
            if x_col in cat_cols:
                bar_data = df.groupby(x_col)[y_col].mean().reset_index()
                fig = px.bar(
                    bar_data, x=x_col, y=y_col, color=color_arg,
                    title=f"Bar Chart - {x_col} vs {y_col} (평균)",
                    template="plotly_white"
                )
            else:
                fig = px.bar(
                    df, x=x_col, y=y_col, color=color_arg,
                    title=f"Bar Chart - {x_col} vs {y_col}",
                    template="plotly_white"
                )

        elif chart_type == "Line Chart":
            fig = px.line(
                df.sort_values(x_col), x=x_col, y=y_col, color=color_arg,
                title=f"Line Chart - {x_col} vs {y_col}",
                template="plotly_white"
            )

        if fig:
            fig.update_layout(height=500, font=dict(size=13))
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ 그래프 생성 오류: {e}")

# ── 4. 상관관계 히트맵 ────────────────────────────────
st.markdown("---")
st.subheader("🌡️ 상관관계 히트맵 (수치형 변수)")

if len(num_cols) >= 2:
    corr = df[num_cols].corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r",
        title="변수 간 상관관계",
        template="plotly_white"
    )
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.info("수치형 변수가 2개 이상이어야 상관관계 히트맵을 표시할 수 있습니다.")
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="데이터 전처리", page_icon="⚙️", layout="wide")

st.title("⚙️ 데이터 전처리 / Feature Selection / Data Partitioning")
st.markdown("---")

# ── 데이터 확인 ───────────────────────────────────────
if st.session_state.get("df") is None:
    st.warning("⚠️ 먼저 **메인 페이지**에서 데이터를 업로드해주세요.")
    st.stop()

# processed_df 초기화
if st.session_state.get("processed_df") is None:
    st.session_state.processed_df = st.session_state.df.copy()

df = st.session_state.processed_df

# ════════════════════════════════════════════════════════
# SECTION 1 : 데이터 전처리
# ════════════════════════════════════════════════════════
st.subheader("🧹 1. 데이터 전처리")

# 현재 상태 요약
c1, c2, c3 = st.columns(3)
c1.metric("📏 행 수",    df.shape[0])
c2.metric("📐 열 수",    df.shape[1])
c3.metric("❓ 결측치 수", df.isnull().sum().sum())

st.markdown("---")

# ── 1-1. 결측치 처리 ──────────────────────────────────
with st.expander("❓ 결측치 처리", expanded=True):
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        st.success("✅ 결측치가 없습니다.")
    else:
        st.dataframe(
            pd.DataFrame({"결측치 수": missing,
                          "비율(%)": (missing/len(df)*100).round(2)}),
            use_container_width=True
        )

        miss_method = st.radio(
            "처리 방법 선택",
            ["평균값으로 대체 (수치형)", "중앙값으로 대체 (수치형)",
             "최빈값으로 대체 (범주형)", "결측치 행 삭제"],
            horizontal=True
        )

        if st.button("결측치 처리 실행", key="btn_missing"):
            df_temp = st.session_state.processed_df.copy()
            num_cols = df_temp.select_dtypes(include="number").columns
            cat_cols = df_temp.select_dtypes(exclude="number").columns

            if miss_method == "평균값으로 대체 (수치형)":
                df_temp[num_cols] = df_temp[num_cols].fillna(
                    df_temp[num_cols].mean())
            elif miss_method == "중앙값으로 대체 (수치형)":
                df_temp[num_cols] = df_temp[num_cols].fillna(
                    df_temp[num_cols].median())
            elif miss_method == "최빈값으로 대체 (범주형)":
                for col in cat_cols:
                    df_temp[col] = df_temp[col].fillna(
                        df_temp[col].mode()[0])
            else:
                df_temp = df_temp.dropna()

            st.session_state.processed_df = df_temp
            st.success(f"✅ 결측치 처리 완료! 남은 결측치: "
                       f"{df_temp.isnull().sum().sum()}")
            st.rerun()

# ── 1-2. 이상치 처리 ──────────────────────────────────
with st.expander("⚠️ 이상치 처리 (IQR 방법)", expanded=False):
    num_cols = st.session_state.processed_df.select_dtypes(
        include="number").columns.tolist()

    if not num_cols:
        st.info("수치형 변수가 없습니다.")
    else:
        outlier_cols = st.multiselect(
            "이상치를 처리할 변수 선택", num_cols, default=num_cols[:3]
        )
        outlier_method = st.radio(
            "처리 방법",
            ["IQR 기반 클리핑 (상·하한 대체)", "IQR 기반 행 삭제"],
            horizontal=True
        )

        if outlier_cols:
            # 이상치 현황 미리보기
            preview_rows = []
            for col in outlier_cols:
                Q1 = st.session_state.processed_df[col].quantile(0.25)
                Q3 = st.session_state.processed_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
                n_out = ((st.session_state.processed_df[col] < lower) |
                         (st.session_state.processed_df[col] > upper)).sum()
                preview_rows.append(
                    {"변수": col, "하한": round(lower,2),
                     "상한": round(upper,2), "이상치 수": n_out}
                )
            st.dataframe(pd.DataFrame(preview_rows), use_container_width=True)

        if st.button("이상치 처리 실행", key="btn_outlier"):
            df_temp = st.session_state.processed_df.copy()
            for col in outlier_cols:
                Q1 = df_temp[col].quantile(0.25)
                Q3 = df_temp[col].quantile(0.75)
                IQR = Q3 - Q1
                lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
                if outlier_method == "IQR 기반 클리핑 (상·하한 대체)":
                    df_temp[col] = df_temp[col].clip(lower, upper)
                else:
                    df_temp = df_temp[
                        (df_temp[col] >= lower) & (df_temp[col] <= upper)
                    ]
            st.session_state.processed_df = df_temp
            st.success(f"✅ 이상치 처리 완료! 현재 행 수: {len(df_temp):,}")
            st.rerun()

# ── 1-3. 원핫 인코딩 ──────────────────────────────────
with st.expander("🔤 원핫 인코딩 (범주형 → 수치형)", expanded=False):
    cat_cols = st.session_state.processed_df.select_dtypes(
        exclude="number").columns.tolist()

    if not cat_cols:
        st.success("✅ 범주형 변수가 없습니다. (이미 인코딩 완료)")
    else:
        st.write("**범주형 변수 목록:**", cat_cols)
        encode_cols = st.multiselect(
            "인코딩할 변수 선택", cat_cols, default=cat_cols
        )

        if st.button("원핫 인코딩 실행", key="btn_ohe"):
            df_temp = st.session_state.processed_df.copy()
            df_temp = pd.get_dummies(df_temp, columns=encode_cols,
                                     drop_first=False)
            # bool → int 변환
            bool_cols = df_temp.select_dtypes(include="bool").columns
            df_temp[bool_cols] = df_temp[bool_cols].astype(int)
            st.session_state.processed_df = df_temp
            st.success(f"✅ 원핫 인코딩 완료! "
                       f"열 수: {df_temp.shape[1]} (이전: "
                       f"{st.session_state.df.shape[1]})")
            st.rerun()

# 전처리 초기화 버튼
st.markdown("---")
if st.button("🔄 전처리 초기화 (원본 복원)", type="secondary"):
    st.session_state.processed_df = st.session_state.df.copy()
    st.success("✅ 원본 데이터로 복원되었습니다.")
    st.rerun()

# 전처리 결과 미리보기
st.markdown("**📋 현재 데이터 미리보기**")
st.dataframe(st.session_state.processed_df.head(5), use_container_width=True)

# ════════════════════════════════════════════════════════
# SECTION 2 : Feature Selection
# ════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("🎯 2. Feature Selection")

current_df = st.session_state.processed_df
all_cols   = current_df.columns.tolist()

col_fs1, col_fs2 = st.columns(2)

with col_fs1:
    target_col = st.selectbox(
        "🎯 종속변수 (Y) 선택",
        all_cols,
        index=len(all_cols)-1,   # 기본값: 마지막 열
        help="예측하려는 목표 변수를 선택하세요."
    )

with col_fs2:
    feature_options = [c for c in all_cols if c != target_col]
    selected_features = st.multiselect(
        "📌 독립변수 (X) 선택",
        feature_options,
        default=feature_options,
        help="모델 학습에 사용할 변수를 선택하세요."
    )

if selected_features:
    st.info(f"✅ 선택된 독립변수 **{len(selected_features)}개**: "
            f"{', '.join(selected_features)}")
    st.info(f"🎯 종속변수: **{target_col}**")

    # 종속변수 분포 확인
    import plotly.express as px
    val_counts = current_df[target_col].value_counts().reset_index()
    val_counts.columns = [target_col, "count"]
    fig_target = px.bar(
        val_counts, x=target_col, y="count",
        title=f"종속변수 '{target_col}' 분포",
        template="plotly_white", color=target_col
    )
    st.plotly_chart(fig_target, use_container_width=True)
else:
    st.warning("⚠️ 독립변수를 1개 이상 선택해주세요.")

# ════════════════════════════════════════════════════════
# SECTION 3 : Data Partitioning
# ════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("✂️ 3. Data Partitioning")

col_dp1, col_dp2 = st.columns(2)

with col_dp1:
    split_ratio = st.radio(
        "Train : Test 비율 선택",
        ["7 : 3", "8 : 2"],
        horizontal=True
    )
    test_size = 0.3 if split_ratio == "7 : 3" else 0.2

with col_dp2:
    random_seed = st.number_input(
        "Random Seed", min_value=0, max_value=9999, value=42
    )

# 비율 시각화
train_pct = int(split_ratio.split(":")[0].strip())
test_pct  = int(split_ratio.split(":")[1].strip())

st.markdown(f"""
<div style='display:flex; height:30px; border-radius:8px; overflow:hidden; margin:10px 0;'>
    <div style='width:{train_pct*10}%; background:#4CAF50; 
                display:flex; align-items:center; justify-content:center; 
                color:white; font-weight:bold;'>
        Train {train_pct*10}%
    </div>
    <div style='width:{test_pct*10}%; background:#F44336; 
                display:flex; align-items:center; justify-content:center; 
                color:white; font-weight:bold;'>
        Test {test_pct*10}%
    </div>
</div>
""", unsafe_allow_html=True)

# 분할 실행
if st.button("✂️ 데이터 분할 실행", type="primary",
             disabled=not bool(selected_features)):
    try:
        X = current_df[selected_features]
        y = current_df[target_col]

        # 수치형 확인
        non_num = X.select_dtypes(exclude="number").columns.tolist()
        if non_num:
            st.error(f"❌ 수치형이 아닌 변수가 있습니다: {non_num}\n"
                     "원핫 인코딩 후 다시 시도해주세요.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size,
                random_state=random_seed, stratify=y
            )

            st.session_state.X_train = X_train
            st.session_state.X_test  = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test  = y_test
            st.session_state.selected_features = selected_features
            st.session_state.target  = target_col

            st.success("✅ 데이터 분할 완료!")

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Train 샘플", f"{len(X_train):,}")
            r2.metric("Test 샘플",  f"{len(X_test):,}")
            r3.metric("Train 비율", f"{len(X_train)/len(X)*100:.1f}%")
            r4.metric("Test 비율",  f"{len(X_test)/len(X)*100:.1f}%")

    except Exception as e:
        st.error(f"❌ 데이터 분할 오류: {e}")

# 분할 완료 상태 표시
if st.session_state.get("X_train") is not None:
    st.success(
        f"✅ 데이터 분할 완료 상태 | "
        f"Train: {len(st.session_state.X_train):,}행 / "
        f"Test: {len(st.session_state.X_test):,}행"
    )
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

st.set_page_config(page_title="연구 모형", page_icon="🤖", layout="wide")

st.title("🤖 연구 모형")
st.markdown("---")

# ── 사전 조건 확인 ────────────────────────────────────
if st.session_state.get("X_train") is None:
    st.warning("⚠️ 먼저 **데이터 전처리 페이지**에서 데이터 분할을 완료해주세요.")
    st.stop()

X_train = st.session_state.X_train
X_test  = st.session_state.X_test
y_train = st.session_state.y_train
y_test  = st.session_state.y_test

# ════════════════════════════════════════════════════════
# MODEL 1 : Logistic Regression
# ════════════════════════════════════════════════════════
st.subheader("📘 1. Logistic Regression")

with st.expander("⚙️ 하이퍼파라미터 설정", expanded=True):
    col_lr1, col_lr2, col_lr3 = st.columns(3)
    with col_lr1:
        lr_C = st.select_slider(
            "규제 강도 C (클수록 규제 약함)",
            options=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], value=1.0
        )
    with col_lr2:
        lr_max_iter = st.slider("최대 반복 횟수", 100, 2000, 1000, 100)
    with col_lr3:
        lr_solver = st.selectbox(
            "Solver", ["lbfgs", "liblinear", "saga"]
        )

if st.button("🚀 Logistic Regression 학습", type="primary", key="btn_lr"):
    with st.spinner("모델 학습 중..."):
        try:
            lr_model = LogisticRegression(
                C=lr_C, max_iter=lr_max_iter,
                solver=lr_solver, random_state=42
            )
            lr_model.fit(X_train, y_train)
            y_pred_lr = lr_model.predict(X_test)
            y_prob_lr = lr_model.predict_proba(X_test)[:, 1]

            results = {
                "accuracy":  accuracy_score(y_test, y_pred_lr),
                "precision": precision_score(y_test, y_pred_lr,
                                             average="weighted",
                                             zero_division=0),
                "recall":    recall_score(y_test, y_pred_lr,
                                          average="weighted",
                                          zero_division=0),
                "f1":        f1_score(y_test, y_pred_lr,
                                      average="weighted",
                                      zero_division=0),
                "y_pred":    y_pred_lr,
                "y_prob":    y_prob_lr,
                "cm":        confusion_matrix(y_test, y_pred_lr)
            }
            st.session_state.lr_model   = lr_model
            st.session_state.lr_results = results
            st.success("✅ Logistic Regression 학습 완료!")

        except Exception as e:
            st.error(f"❌ 학습 오류: {e}")

# LR 결과 표시
if st.session_state.get("lr_results"):
    res = st.session_state.lr_results
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy",  f"{res['accuracy']:.4f}")
    m2.metric("Precision", f"{res['precision']:.4f}")
    m3.metric("Recall",    f"{res['recall']:.4f}")
    m4.metric("F1-Score",  f"{res['f1']:.4f}")

    # 혼동 행렬
    col_cm, col_coef = st.columns(2)
    with col_cm:
        st.markdown("**혼동 행렬**")
        cm = res["cm"]
        labels = [str(c) for c in sorted(y_test.unique())]
        fig_cm = ff.create_annotated_heatmap(
            cm, x=labels, y=labels,
            colorscale="Blues", showscale=True
        )
        fig_cm.update_layout(
            title="Confusion Matrix - Logistic Regression",
            xaxis_title="예측값", yaxis_title="실제값", height=350
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_coef:
        st.markdown("**변수 중요도 (계수)**")
        lr_model = st.session_state.lr_model
        if len(lr_model.classes_) == 2:
            coef_df = pd.DataFrame({
                "변수": X_train.columns,
                "계수": lr_model.coef_[0]
            }).sort_values("계수", key=abs, ascending=False)
            fig_coef = px.bar(
                coef_df, x="계수", y="변수", orientation="h",
                title="Logistic Regression 계수",
                template="plotly_white", color="계수",
                color_continuous_scale="RdBu"
            )
            fig_coef.update_layout(height=350)
            st.plotly_chart(fig_coef, use_container_width=True)

# ════════════════════════════════════════════════════════
# MODEL 2 : Decision Tree
# ════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("🌳 2. Decision Tree")

with st.expander("⚙️ 하이퍼파라미터 설정", expanded=True):
    col_dt1, col_dt2, col_dt3 = st.columns(3)
    with col_dt1:
        dt_max_depth = st.slider("최대 깊이 (max_depth)", 1, 20, 5)
    with col_dt2:
        dt_min_samples = st.slider("최소 분할 샘플 수", 2, 50, 2)
    with col_dt3:
        dt_criterion = st.selectbox("분할 기준", ["gini", "entropy"])

if st.button("🚀 Decision Tree 학습", type="primary", key="btn_dt"):
    with st.spinner("모델 학습 중..."):
        try:
            dt_model = DecisionTreeClassifier(
                max_depth=dt_max_depth,
                min_samples_split=dt_min_samples,
                criterion=dt_criterion,
                random_state=42
            )
            dt_model.fit(X_train, y_train)
            y_pred_dt = dt_model.predict(X_test)
            y_prob_dt = dt_model.predict_proba(X_test)[:, 1]

            results = {
                "accuracy":  accuracy_score(y_test, y_pred_dt),
                "precision": precision_score(y_test, y_pred_dt,
                                             average="weighted",
                                             zero_division=0),
                "recall":    recall_score(y_test, y_pred_dt,
                                          average="weighted",
                                          zero_division=0),
                "f1":        f1_score(y_test, y_pred_dt,
                                      average="weighted",
                                      zero_division=0),
                "y_pred":    y_pred_dt,
                "y_prob":    y_prob_dt,
                "cm":        confusion_matrix(y_test, y_pred_dt)
            }
            st.session_state.dt_model   = dt_model
            st.session_state.dt_results = results
            st.success("✅ Decision Tree 학습 완료!")

        except Exception as e:
            st.error(f"❌ 학습 오류: {e}")

# DT 결과 표시
if st.session_state.get("dt_results"):
    res = st.session_state.dt_results
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy",  f"{res['accuracy']:.4f}")
    m2.metric("Precision", f"{res['precision']:.4f}")
    m3.metric("Recall",    f"{res['recall']:.4f}")
    m4.metric("F1-Score",  f"{res['f1']:.4f}")

    col_cm2, col_fi = st.columns(2)
    with col_cm2:
        st.markdown("**혼동 행렬**")
        cm = res["cm"]
        labels = [str(c) for c in sorted(y_test.unique())]
        fig_cm2 = ff.create_annotated_heatmap(
            cm, x=labels, y=labels,
            colorscale="Greens", showscale=True
        )
        fig_cm2.update_layout(
            title="Confusion Matrix - Decision Tree",
            xaxis_title="예측값", yaxis_title="실제값", height=350
        )
        st.plotly_chart(fig_cm2, use_container_width=True)

    with col_fi:
        st.markdown("**변수 중요도 (Feature Importance)**")
        dt_model = st.session_state.dt_model
        fi_df = pd.DataFrame({
            "변수": X_train.columns,
            "중요도": dt_model.feature_importances_
        }).sort_values("중요도", ascending=False).head(15)
        fig_fi = px.bar(
            fi_df, x="중요도", y="변수", orientation="h",
            title="Decision Tree Feature Importance",
            template="plotly_white", color="중요도",
            color_continuous_scale="Greens"
        )
        fig_fi.update_layout(height=350)
        st.plotly_chart(fig_fi, use_container_width=True)

    # 트리 시각화
    st.markdown("**🌳 Decision Tree 시각화**")
    max_vis_depth = st.slider("시각화 깊이", 1, min(5, dt_max_depth), 3)
    fig_tree, ax = plt.subplots(figsize=(20, 8))
    plot_tree(
        st.session_state.dt_model,
        feature_names=X_train.columns.tolist(),
        class_names=[str(c) for c in sorted(y_test.unique())],
        filled=True, rounded=True, fontsize=9,
        max_depth=max_vis_depth, ax=ax
    )
    st.pyplot(fig_tree)
    plt.close()
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_curve, auc

st.set_page_config(page_title="연구 결과", page_icon="📈", layout="wide")

st.title("📈 연구 결과")
st.markdown("---")

# ── 사전 조건 확인 ────────────────────────────────────
lr_done = st.session_state.get("lr_results") is not None
dt_done = st.session_state.get("dt_results") is not None

if not lr_done and not dt_done:
    st.warning("⚠️ 먼저 **연구 모형 페이지**에서 모델을 학습해주세요.")
    st.stop()

y_test = st.session_state.y_test

# ════════════════════════════════════════════════════════
# SECTION 1 : 성능 지표 비교
# ════════════════════════════════════════════════════════
st.subheader("📊 1. 모델 성능 지표 비교")

# 성능 테이블 구성
metrics_data = []
if lr_done:
    r = st.session_state.lr_results
    metrics_data.append({
        "모델": "Logistic Regression",
        "Accuracy":  round(r["accuracy"],  4),
        "Precision": round(r["precision"], 4),
        "Recall":    round(r["recall"],    4),
        "F1-Score":  round(r["f1"],        4),
    })
if dt_done:
    r = st.session_state.dt_results
    metrics_data.append({
        "모델": "Decision Tree",
        "Accuracy":  round(r["accuracy"],  4),
        "Precision": round(r["precision"], 4),
        "Recall":    round(r["recall"],    4),
        "F1-Score":  round(r["f1"],        4),
    })

metrics_df = pd.DataFrame(metrics_data).set_index("모델")

# 스타일 적용 (최고값 강조)
def highlight_max(s):
    is_max = s == s.max()
    return ["background-color: #d4edda; font-weight:bold"
            if v else "" for v in is_max]

st.dataframe(
    metrics_df.style.apply(highlight_max, axis=0).format("{:.4f}"),
    use_container_width=True
)

# ── 레이더 차트 ───────────────────────────────────────
st.markdown("**🕸️ 성능 지표 레이더 차트**")
categories = ["Accuracy", "Precision", "Recall", "F1-Score"]

fig_radar = go.Figure()
colors = ["#1f77b4", "#2ca02c"]

for i, row in metrics_df.iterrows():
    vals = row[categories].tolist()
    fig_radar.add_trace(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=categories + [categories[0]],
        fill="toself", name=i,
        line_color=colors[metrics_df.index.tolist().index(i)]
    ))

fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    showlegend=True, height=450,
    title="모델 성능 비교 레이더 차트",
    template="plotly_white"
)
st.plotly_chart(fig_radar, use_container_width=True)

# ── 지표별 막대 그래프 ────────────────────────────────
st.markdown("**📊 지표별 비교 막대 그래프**")
melted = metrics_df.reset_index().melt(
    id_vars="모델", var_name="지표", value_name="값"
)
fig_bar = px.bar(
    melted, x="지표", y="값", color="모델",
    barmode="group", template="plotly_white",
    title="모델별 성능 지표 비교",
    color_discrete_sequence=["#1f77b4", "#2ca02c"],
    text_auto=".4f"
)
fig_bar.update_layout(height=400, yaxis_range=[0, 1.05])
fig_bar.update_traces(textposition="outside")
st.plotly_chart(fig_bar, use_container_width=True)

# ════════════════════════════════════════════════════════
# SECTION 2 : ROC Curve 비교
# ════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("📉 2. ROC Curve 비교")

# 이진 분류 여부 확인
unique_classes = sorted(y_test.unique())
is_binary = len(unique_classes) == 2

if not is_binary:
    st.warning("⚠️ ROC Curve는 이진 분류(클래스 2개)에서만 지원됩니다.")
else:
    fig_roc = go.Figure()
    # 기준선
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines", name="Random (AUC=0.50)",
        line=dict(dash="dash", color="gray", width=1.5)
    ))

    roc_summary = []

    if lr_done:
        y_prob_lr = st.session_state.lr_results["y_prob"]
        fpr, tpr, _ = roc_curve(y_test, y_prob_lr)
        roc_auc = auc(fpr, tpr)
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"Logistic Regression (AUC={roc_auc:.4f})",
            line=dict(color="#1f77b4", width=2.5)
        ))
        roc_summary.append(
            {"모델": "Logistic Regression", "AUC": round(roc_auc, 4)}
        )

    if dt_done:
        y_prob_dt = st.session_state.dt_results["y_prob"]
        fpr, tpr, _ = roc_curve(y_test, y_prob_dt)
        roc_auc = auc(fpr, tpr)
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"Decision Tree (AUC={roc_auc:.4f})",
            line=dict(color="#2ca02c", width=2.5)
        ))
        roc_summary.append(
            {"모델": "Decision Tree", "AUC": round(roc_auc, 4)}
        )

    fig_roc.update_layout(
        title="ROC Curve 비교",
        xaxis_title="False Positive Rate (FPR)",
        yaxis_title="True Positive Rate (TPR)",
        template="plotly_white",
        height=500,
        legend=dict(x=0.6, y=0.1),
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.02])
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    # AUC 요약 테이블
    st.markdown("**AUC 요약**")
    auc_df = pd.DataFrame(roc_summary).set_index("모델")
    st.dataframe(
        auc_df.style.highlight_max(color="#d4edda").format("{:.4f}"),
        use_container_width=True
    )

# ════════════════════════════════════════════════════════
# SECTION 3 : 종합 분석 요약
# ════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("📝 3. 종합 분석 요약")

if len(metrics_data) == 2:
    best_model = metrics_df["F1-Score"].idxmax()
    best_acc   = metrics_df["Accuracy"].idxmax()
    best_f1    = metrics_df["F1-Score"].max()

    st.markdown(f"""
    <div style='background:#f0f8ff; padding:20px; 
                border-radius:10px; border-left:5px solid #1f77b4;'>
        <h4>🏆 분석 결과 요약</h4>
        <ul>
            <li><b>최고 Accuracy 모델:</b> {best_acc} 
                ({metrics_df.loc[best_acc, 'Accuracy']:.4f})</li>
            <li><b>최고 F1-Score 모델:</b> {best_model} 
                ({best_f1:.4f})</li>
            <li><b>권장 모델:</b> <span style='color:#e74c3c; font-weight:bold;'>
                {best_model}</span> 
                (F1-Score 기준)</li>
        </ul>
        <p style='color:#666; font-size:0.9em;'>
            ※ 이탈 고객 예측에서는 Recall(재현율)이 높은 모델이 
            실제 이탈 고객을 더 잘 잡아낼 수 있습니다.
        </p>
    </div>
    """, unsafe_allow_html=True)

elif len(metrics_data) == 1:
    model_name = metrics_data[0]["모델"]
    st.info(f"현재 **{model_name}** 모델만 학습되었습니다. "
            "두 모델을 모두 학습하면 비교 분석이 가능합니다.")

# ── 예측 결과 다운로드 ────────────────────────────────
st.markdown("---")
st.subheader("💾 예측 결과 다운로드")

if lr_done or dt_done:
    result_df = pd.DataFrame({"실제값": y_test.values})

    if lr_done:
        result_df["LR_예측값"] = st.session_state.lr_results["y_pred"]
        result_df["LR_예측확률"] = st.session_state.lr_results["y_prob"].round(4)
    if dt_done:
        result_df["DT_예측값"] = st.session_state.dt_results["y_pred"]
        result_df["DT_예측확률"] = st.session_state.dt_results["y_prob"].round(4)

    st.dataframe(result_df.head(10), use_container_width=True)

    csv = result_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="📥 예측 결과 CSV 다운로드",
        data=csv,
        file_name="churn_prediction_results.csv",
        mime="text/csv",
        type="primary"
    )
