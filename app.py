import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pickle
import requests
from io import BytesIO

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MU Canteen Rush Predictor",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #1a1d27 0%, #12151f 100%);
        border-right: 1px solid #2a2d3e;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #1a1d27;
        border: 1px solid #2a2d3e;
        border-radius: 12px;
        padding: 16px 20px;
    }

    /* Rush level badge */
    .rush-badge {
        display: inline-block;
        padding: 10px 28px;
        border-radius: 50px;
        font-size: 1.3rem;
        font-weight: 700;
        letter-spacing: 1px;
        margin-top: 8px;
    }
    .rush-low    { background: #0d2b1a; color: #4ade80; border: 2px solid #22c55e; }
    .rush-medium { background: #2d1f08; color: #fb923c; border: 2px solid #f97316; }
    .rush-high   { background: #2b0d0d; color: #f87171; border: 2px solid #ef4444; }

    /* Prediction card */
    .pred-card {
        background: #1a1d27;
        border: 1px solid #2a2d3e;
        border-radius: 16px;
        padding: 24px 28px;
        text-align: center;
        margin-bottom: 20px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: #1a1d27;
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #888;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: #2a2d3e !important;
        color: #fff !important;
    }

    /* Section headers */
    .section-header {
        font-size: 1rem;
        font-weight: 600;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin: 20px 0 10px;
        padding-bottom: 6px;
        border-bottom: 1px solid #2a2d3e;
    }

    /* Divider */
    hr { border-color: #2a2d3e !important; }

    /* Plot background fix */
    .stPlotlyChart, [data-testid="stImage"] { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MATPLOTLIB DARK THEME
# ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#1a1d27",
    "axes.facecolor":    "#1a1d27",
    "axes.edgecolor":    "#2a2d3e",
    "axes.labelcolor":   "#aaa",
    "xtick.color":       "#aaa",
    "ytick.color":       "#aaa",
    "text.color":        "#ddd",
    "grid.color":        "#2a2d3e",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

PALETTE = {"Low": "#22c55e", "Medium": "#f97316", "High": "#ef4444"}

# ─────────────────────────────────────────────
#  CACHED DATA & MODEL LOADERS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="📦 Loading dataset...")
def load_data():
    url = "https://raw.githubusercontent.com/bloomingcodeby-iffath/MU_Canteen_Rush_Predictor/refs/heads/main/canteen_rush_data.csv"
    return pd.read_csv(url)

@st.cache_resource(show_spinner="🤖 Loading model...")
def load_model():
    model_url = "https://raw.githubusercontent.com/bloomingcodeby-iffath/MU_Canteen_Rush_Predictor/refs/heads/main/rf_model.pkl"
    r = requests.get(model_url)
    return pickle.loads(r.content)

@st.cache_data(show_spinner=False)
def get_encodings(_df):
    day_map     = {d: i for i, d in enumerate(sorted(_df["Day"].unique()))}
    weather_map = {w: i for i, w in enumerate(sorted(_df["Weather"].unique()))}
    return day_map, weather_map

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def get_rush_level(students: float) -> str:
    if students < 20:   return "Low"
    elif students < 40: return "Medium"
    else:               return "High"

def rush_badge(level: str) -> str:
    cls = f"rush-{level.lower()}"
    return f'<div class="rush-badge {cls}">🚦 {level.upper()} RUSH</div>'

# ─────────────────────────────────────────────
#  LOAD
# ─────────────────────────────────────────────
df = load_data()
rf = load_model()
day_map, weather_map = get_encodings(df)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍽️ Canteen Rush\n**Predictor**")
    st.markdown('<div class="section-header">Input Parameters</div>', unsafe_allow_html=True)

    time      = st.slider("⏰ Time (Hour)", 9, 17, 12)
    lunch     = st.radio("🥗 Lunch Period?", [0, 1],
                         format_func=lambda x: "Yes" if x == 1 else "No",
                         horizontal=True)
    day       = st.selectbox("📅 Day", sorted(df["Day"].unique()))
    weather   = st.selectbox("🌤️ Weather", sorted(df["Weather"].unique()))

    st.markdown("---")
    st.caption("Model: Random Forest Regressor\nData: MU Canteen Dataset")

# ─────────────────────────────────────────────
#  PREDICTION
# ─────────────────────────────────────────────
X_input = pd.DataFrame(
    [[time, lunch, day_map[day], weather_map[weather]]],
    columns=["Time", "Lunch_Time", "Day_Encoded", "Weather_Encoded"],
)
pred  = rf.predict(X_input)[0]
rush  = get_rush_level(pred)

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("# 🍽️ MU Canteen Rush Predictor")
st.markdown("Predict how crowded the canteen will be based on time, weather, and day.")
st.markdown("---")

# ─────────────────────────────────────────────
#  PREDICTION RESULT ROW
# ─────────────────────────────────────────────
col1, col2, col3, col4 = st.columns([1.4, 1, 1, 1])

with col1:
    st.markdown(f"""
    <div class="pred-card">
        <div style="color:#888; font-size:.85rem; margin-bottom:4px;">PREDICTED STUDENTS</div>
        <div style="font-size:3rem; font-weight:800; color:#fff;">{int(pred)}</div>
        {rush_badge(rush)}
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.metric("⏰ Time", f"{time}:00")
with col3:
    st.metric("🌤️ Weather", weather)
with col4:
    st.metric("📅 Day", day)

st.markdown("---")

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Rush Overview", "🌦️ Feature Impact", "🔗 Patterns & Correlation"])

# ── TAB 1 : Rush Overview ──────────────────
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Rush Level vs Avg Students")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        order = ["Low", "Medium", "High"]
        colors = [PALETTE[r] for r in order]
        sns.barplot(x="Rush_Level", y="Students", data=df,
                    order=order, palette=PALETTE, ax=ax)
        ax.set_xlabel("Rush Level"); ax.set_ylabel("Avg Students")
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with c2:
        st.markdown("#### Rush Level Distribution")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        counts = df["Rush_Level"].value_counts().reindex(order)
        wedges, texts, autotexts = ax.pie(
            counts,
            labels=counts.index,
            autopct="%1.1f%%",
            colors=[PALETTE[r] for r in counts.index],
            startangle=140,
            wedgeprops={"linewidth": 2, "edgecolor": "#0f1117"},
        )
        for t in texts + autotexts:
            t.set_color("#ddd")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    c3, c4 = st.columns(2)

    with c3:
        st.markdown("#### Students Over Time (Line)")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        avg_time = df.groupby("Time")["Students"].mean().reset_index()
        ax.plot(avg_time["Time"], avg_time["Students"],
                marker="o", color="#818cf8", linewidth=2, markersize=5)
        ax.fill_between(avg_time["Time"], avg_time["Students"],
                         alpha=0.15, color="#818cf8")
        ax.set_xlabel("Hour"); ax.set_ylabel("Avg Students")
        ax.grid(True)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with c4:
        st.markdown("#### Students vs Time (Scatter)")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        rush_colors = df["Rush_Level"].map(PALETTE)
        ax.scatter(df["Time"], df["Students"],
                   c=rush_colors, alpha=0.7, edgecolors="none", s=40)
        ax.set_xlabel("Hour"); ax.set_ylabel("Students")
        handles = [plt.Line2D([0], [0], marker="o", color="w",
                               markerfacecolor=v, label=k, markersize=8)
                   for k, v in PALETTE.items()]
        ax.legend(handles=handles, framealpha=0.2, labelcolor="white")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

# ── TAB 2 : Feature Impact ─────────────────
with tab2:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Weather vs Students (Box)")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.boxplot(x="Weather", y="Students", data=df,
                    palette="cool", ax=ax,
                    medianprops={"color": "#f97316", "linewidth": 2})
        ax.set_xlabel("Weather"); ax.set_ylabel("Students")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with c2:
        st.markdown("#### Weather — Average Students")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.barplot(x="Weather", y="Students", data=df,
                    estimator="mean", palette="cool", ax=ax)
        ax.set_xlabel("Weather"); ax.set_ylabel("Avg Students")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    c3, c4 = st.columns(2)

    with c3:
        st.markdown("#### Lunch Period vs Students")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.barplot(x="Lunch_Time", y="Students", data=df,
                    palette=["#818cf8", "#f472b6"], ax=ax)
        ax.set_xticklabels(["Non-Lunch", "Lunch"])
        ax.set_xlabel(""); ax.set_ylabel("Avg Students")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with c4:
        st.markdown("#### Lunch Period Distribution")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        lc = df.groupby("Lunch_Time")["Students"].sum()
        ax.pie(lc, labels=["Non-Lunch", "Lunch"],
               autopct="%1.1f%%",
               colors=["#818cf8", "#f472b6"],
               startangle=90,
               wedgeprops={"linewidth": 2, "edgecolor": "#0f1117"})
        for t in ax.texts: t.set_color("#ddd")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

# ── TAB 3 : Patterns & Correlation ────────
with tab3:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Time × Rush Level (Stacked Bar)")
        fig, ax = plt.subplots(figsize=(5, 3.8))
        rush_time = (df.groupby(["Time", "Rush_Level"])["Students"]
                       .sum().unstack(fill_value=0))
        for col in ["Low", "Medium", "High"]:
            if col not in rush_time.columns:
                rush_time[col] = 0
        rush_time[["Low", "Medium", "High"]].plot(
            kind="bar", stacked=True, ax=ax,
            color=[PALETTE["Low"], PALETTE["Medium"], PALETTE["High"]],
            edgecolor="none",
        )
        ax.set_xlabel("Hour"); ax.set_ylabel("Total Students")
        ax.legend(title="Rush Level", framealpha=0.2, labelcolor="white")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with c2:
        st.markdown("#### Time × Rush Level (Heatmap)")
        fig, ax = plt.subplots(figsize=(5, 3.8))
        hm = df.pivot_table(index="Time", columns="Rush_Level",
                            values="Students", aggfunc="sum")
        sns.heatmap(hm, annot=True, fmt=".0f",
                    cmap="YlOrRd", ax=ax,
                    linewidths=0.5, linecolor="#0f1117",
                    cbar_kws={"shrink": 0.8})
        ax.set_xlabel("Rush Level"); ax.set_ylabel("Hour")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown("#### Feature Correlation Heatmap")
    df["Rush_Level_Encoded"] = df["Rush_Level"].map({"Low": 0, "Medium": 1, "High": 2})
    corr = df[["Time", "Lunch_Time", "Students", "Rush_Level_Encoded"]].corr()
    fig, ax = plt.subplots(figsize=(6, 4))
    mask = (corr.abs() < 0.01) & (corr != 1.0)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                ax=ax, linewidths=0.5, linecolor="#0f1117",
                vmin=-1, vmax=1,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Pearson Correlation", color="#aaa", pad=10)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("#### Pairplot (by Rush Level)")
    with st.spinner("Rendering pairplot…"):
        pair_fig = sns.pairplot(
            df[["Time", "Lunch_Time", "Students", "Rush_Level"]],
            hue="Rush_Level",
            palette=PALETTE,
            plot_kws={"alpha": 0.6, "edgecolor": "none"},
            diag_kind="kde",
        )
        pair_fig.fig.patch.set_facecolor("#1a1d27")
        for ax_ in pair_fig.axes.flatten():
            if ax_:
                ax_.set_facecolor("#1a1d27")
        st.pyplot(pair_fig.fig); plt.close()
