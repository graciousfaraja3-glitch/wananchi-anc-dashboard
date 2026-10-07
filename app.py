import base64
import html
import os
import re
import warnings
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import statsmodels.api as sm
import statsmodels.formula.api as smf
import streamlit as st

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# BRAND PALETTE (red, blue, green)
# ============================================================

BLUE = "#1F4E9C"
BLUE_DARK = "#14346B"
BLUE_LIGHT = "#EAF0FA"
RED = "#C8102E"
GREEN = "#2A7F3B"
TEXT = "#27323F"
MUTED = "#5A6675"
GREY = "#8A94A3"
FONT = "Poppins"

COLORWAY = [BLUE, RED, GREEN, "#6C8FD0", "#E36A7C", "#7CC48A"]

DATA_FILE = "wananchi_anc_data.csv"

REQUIRED_COLUMNS = [
    "Year",
    "Age",
    "Visit Count",
    "Systolic BP",
    "Diastolic BP",
    "Weight (kg)",
    "Gestation Weeks",
]

AGE_ORDER = ["Under 20", "20-24", "25-29", "30-34", "35 and above", "Unknown"]
TRIMESTER_ORDER = [
    "First (up to 13 wks)",
    "Second (14-27 wks)",
    "Third (28-42 wks)",
    "Over 42 wks (check)",
    "Unknown",
]
MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

pio.templates["wananchi"] = go.layout.Template(
    layout=dict(
        font=dict(family=f"{FONT}, sans-serif", color=TEXT, size=13),
        colorway=COLORWAY,
        title=dict(font=dict(color=BLUE_DARK, size=16)),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(gridcolor="#E6EAF0", linecolor="#C9D1DC"),
        yaxis=dict(gridcolor="#E6EAF0", linecolor="#C9D1DC"),
    )
)
pio.templates.default = "wananchi"


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Wananchi ANC Monitoring",
    layout="wide"
)


# ============================================================
# STYLING
# ============================================================

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

/* Font: applied to text elements only so icon fonts are left intact */
html, body, .stApp, .stMarkdown, p, li, label, h1, h2, h3, h4, h5, h6,
button, input, textarea, select, th, td,
[data-testid="stMetricLabel"], [data-testid="stMetricValue"],
[data-baseweb="tab"], [data-baseweb="select"] {
    font-family: '__FONT__', sans-serif !important;
}
[data-testid="stIconMaterial"], .material-symbols-rounded,
[class*="material-symbols"] {
    font-family: 'Material Symbols Rounded' !important;
}

/* Base colours: always dark text on light backgrounds */
.stApp { background-color: #F7F9FC; color: __TEXT__; }
.stApp p, .stApp li, .stApp label { color: __TEXT__; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
    color: __MUTED__ !important;
}
.block-container { padding-top: 2.2rem; }

header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }

/* Hide the sidebar collapse arrow on desktop only.
   On phones it stays, otherwise the filters cannot be opened. */
@media (min-width: 992px) {
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
    }
}

/* Title banner */
.brand-banner {
    background: __BLUE__;
    border-bottom: 5px solid __RED__;
    border-radius: 8px;
    padding: 20px 26px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 20px;
}
.brand-banner .brand-logo {
    background: #FFFFFF;
    border-radius: 6px;
    padding: 6px 10px;
    height: 58px;
    width: auto;
}
.brand-banner h1 {
    color: #FFFFFF !important;
    font-weight: 600;
    font-size: 1.7rem;
    margin: 0;
    padding: 0;
    border: none;
}
.brand-banner p {
    color: #E4ECFA !important;
    margin: 6px 0 0 0;
    font-size: 0.92rem;
    font-weight: 300;
}

/* Headings */
h1, h2, h3, h4 { color: __BLUE_DARK__; font-weight: 600; }
h2 {
    border-bottom: 3px solid __GREEN__;
    padding-bottom: 6px;
    display: inline-block;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #DDE4EE;
    border-left: 5px solid __BLUE__;
    border-radius: 8px;
    padding: 12px 14px;
    box-shadow: 0 1px 3px rgba(20, 52, 107, 0.06);
}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {
    color: __MUTED__ !important;
    font-weight: 500;
    white-space: normal;
}
[data-testid="stMetricValue"] { color: __BLUE_DARK__; font-weight: 600; }
[data-testid="stMetricValue"] > div {
    white-space: normal;
    overflow: visible;
    text-overflow: clip;
    line-height: 1.25;
}

/* Result card for the individual calculator */
.result-card {
    background: #FFFFFF;
    border: 1px solid #DDE4EE;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.result-label { color: __MUTED__; font-size: 0.85rem; font-weight: 500; }
.result-value { font-size: 1.45rem; font-weight: 600; line-height: 1.3; }
.result-sub { color: __TEXT__; font-size: 0.95rem; margin-top: 4px; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    border-bottom: 2px solid #DDE4EE;
    overflow-x: auto;
    flex-wrap: nowrap;
}
.stTabs [data-baseweb="tab"] {
    white-space: nowrap;
    padding: 10px 16px;
}
.stTabs [data-baseweb="tab"] p {
    color: #3D4A5C;
    font-weight: 500;
    font-size: 0.92rem;
}
.stTabs [aria-selected="true"] p { color: __BLUE__; font-weight: 600; }
.stTabs [data-baseweb="tab-highlight"] {
    background-color: __RED__ !important;
    height: 3px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 4px solid __BLUE__;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: __BLUE_DARK__;
    border-bottom: none;
    font-size: 1.05rem;
    display: block;
}
.sidebar-section {
    background: __BLUE_LIGHT__;
    border-left: 4px solid __GREEN__;
    border-radius: 6px;
    padding: 8px 12px;
    margin: 16px 0 8px 0;
    color: __BLUE_DARK__;
    font-weight: 600;
    font-size: 0.95rem;
}

/* Inputs: light background, dark text, in any browser theme */
[data-baseweb="input"], [data-baseweb="base-input"],
[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border-color: #B8C4D6 !important;
}
input, textarea {
    color: __TEXT__ !important;
    -webkit-text-fill-color: __TEXT__ !important;
}
[data-baseweb="popover"] ul, [data-baseweb="menu"] {
    background-color: #FFFFFF !important;
}
[data-baseweb="popover"] li, [data-baseweb="popover"] li *,
[data-baseweb="menu"] li, [data-baseweb="menu"] li * {
    color: __TEXT__ !important;
}
[data-testid="stNumberInput"] button {
    background-color: __BLUE_LIGHT__ !important;
    color: __BLUE_DARK__ !important;
}
[data-testid="stSlider"] [data-testid="stThumbValue"] {
    color: __BLUE_DARK__;
    font-weight: 600;
}
span[data-baseweb="tag"] {
    background-color: __BLUE__ !important;
}
span[data-baseweb="tag"], span[data-baseweb="tag"] span {
    color: #FFFFFF !important;
}
span[data-baseweb="tag"] svg { fill: #FFFFFF !important; }

/* Buttons: white text on solid colour, always legible */
.stButton > button, .stFormSubmitButton > button {
    background-color: __RED__;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    padding: 0.55rem 1rem;
    width: 100%;
}
.stButton > button p, .stFormSubmitButton > button p {
    color: #FFFFFF !important;
    font-weight: 600;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    background-color: #A30D25;
    border: none;
}
.stDownloadButton > button {
    background-color: __BLUE__;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    padding: 0.55rem 1rem;
    width: 100%;
}
.stDownloadButton > button p { color: #FFFFFF !important; font-weight: 600; }
.stDownloadButton > button:hover {
    background-color: __BLUE_DARK__;
    border: none;
}

/* Tables and alerts */
[data-testid="stDataFrame"] {
    border: 1px solid #DDE4EE;
    border-radius: 6px;
}
[data-testid="stAlert"] { border-radius: 6px; }

/* Phones and small tablets */
@media (max-width: 768px) {
    .block-container {
        padding-left: 0.9rem;
        padding-right: 0.9rem;
        padding-top: 2.8rem;
    }
    .brand-banner {
        padding: 14px 16px;
        flex-direction: column;
        align-items: flex-start;
        gap: 10px;
    }
    .brand-banner .brand-logo { height: 44px; }
    .brand-banner h1 { font-size: 1.2rem; }
    .brand-banner p { font-size: 0.82rem; }
    h2 { font-size: 1.3rem; }
    [data-testid="stMetricValue"] { font-size: 1.25rem; }
    .stTabs [data-baseweb="tab"] { padding: 8px 10px; }
}
</style>
"""

for token, value in {
    "__FONT__": FONT,
    "__TEXT__": TEXT,
    "__MUTED__": MUTED,
    "__BLUE__": BLUE,
    "__BLUE_DARK__": BLUE_DARK,
    "__BLUE_LIGHT__": BLUE_LIGHT,
    "__RED__": RED,
    "__GREEN__": GREEN,
}.items():
    CSS = CSS.replace(token, value)

st.markdown(CSS, unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================

def _st_version():
    match = re.match(r"(\d+)\.(\d+)", st.__version__)
    return (int(match.group(1)), int(match.group(2))) if match else (0, 0)


NEW_WIDTH_API = _st_version() >= (1, 50)


def show_chart(fig, key=None):
    """Draw a Plotly chart at full width on any Streamlit version."""
    fig.update_layout(
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="left",
            x=0,
            title_text=""
        ),
    )
    if NEW_WIDTH_API:
        st.plotly_chart(fig, width="stretch", key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, key=key)


def show_table(data):
    """Draw a table at full width, without the row index."""
    if NEW_WIDTH_API:
        st.dataframe(data, width="stretch", hide_index=True)
    else:
        st.dataframe(data, use_container_width=True, hide_index=True)


def category_color(name):
    text = str(name).lower()
    if "high" in text:
        return RED
    if "moderate" in text or "medium" in text:
        return BLUE
    if "low" in text:
        return GREEN
    return GREY


def category_rank(name):
    text = str(name).lower()
    if "high" in text:
        return 2
    if "moderate" in text or "medium" in text:
        return 1
    if "low" in text:
        return 0
    return 3


def trimester_label(weeks):
    if pd.isna(weeks):
        return "Unknown"
    if weeks > 42:
        return "Over 42 wks (check)"
    if weeks <= 13:
        return "First (up to 13 wks)"
    if weeks <= 27:
        return "Second (14-27 wks)"
    return "Third (28-42 wks)"


def extract_month(data):
    """Return a numeric month (1-12) Series if a date or month column exists."""
    for col in data.columns:
        name = col.lower()
        if "date" not in name and "month" not in name:
            continue
        series = data[col]
        if pd.api.types.is_numeric_dtype(series):
            valid = series.dropna()
            if len(valid) > 0 and valid.between(1, 12).all():
                return series.astype(float)
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(series, errors="coerce")
        if parsed.notna().mean() >= 0.5:
            return parsed.dt.month.astype(float)
        names = {m: i + 1 for i, m in enumerate(MONTH_NAMES)}
        mapped = series.astype(str).str[:3].str.title().map(names)
        if mapped.notna().mean() >= 0.5:
            return mapped.astype(float)
    return None


def year_labels(data):
    """Copy of the data with Year as text so charts use distinct colours."""
    out = data.copy()
    out["Year"] = out["Year"].astype(str)
    return out


def summarise(data):
    """Headline measures used in comparisons and the summary report."""
    if len(data) == 0:
        return {}
    elevated = (data["Systolic BP"] >= 140) | (data["Diastolic BP"] >= 90)
    return {
        "Records": int(len(data)),
        "Mean age (years)": data["Age"].mean(),
        "Under 20 (%)": 100 * (data["Age"] < 20).mean(),
        "35 and above (%)": 100 * (data["Age"] >= 35).mean(),
        "Mean gestation (weeks)": data["Gestation Weeks"].mean(),
        "Mean ANC visits": data["Visit Count"].mean(),
        "4 or more visits (%)": 100 * (data["Visit Count"] >= 4).mean(),
        "Elevated BP (%)": 100 * elevated.mean(),
        "Mean weight (kg)": data["Weight (kg)"].mean(),
    }


def indicator_table(data, targets):
    """Programme indicators per year, compared with the chosen targets."""
    rows = []
    for year, group in data.groupby("Year"):
        anc4 = 100 * (group["Visit Count"] >= 4).mean()

        first = group[
            (group["Visit Count"] == 1) & group["Gestation Weeks"].notna()
        ]
        early = (
            100 * (first["Gestation Weeks"] <= 12).mean()
            if len(first) > 0 else np.nan
        )

        elevated = 100 * (
            (group["Systolic BP"] >= 140) | (group["Diastolic BP"] >= 90)
        ).mean()

        values = {
            "ANC 4 or more visits (%)": anc4,
            "First visit at 12 weeks or earlier (%)": early,
            "Elevated BP records (%)": elevated,
        }

        for name, value in values.items():
            target, higher_is_better = targets[name]
            if pd.isna(value):
                status = "No data"
            elif higher_is_better:
                status = "Meets target" if value >= target else "Below target"
            else:
                status = "Within limit" if value <= target else "Above limit"
            rows.append({
                "Indicator": name,
                "Year": int(year),
                "Value (%)": None if pd.isna(value) else round(value, 1),
                "Target (%)": target,
                "Status": status,
            })
    return pd.DataFrame(rows)


def share_by(data, column):
    table = data.groupby(["Year", column]).size().reset_index(name="Records")
    table["Share (%)"] = (
        100 * table["Records"] /
        table.groupby("Year")["Records"].transform("sum")
    ).round(1)
    return table


def assess(age, parity, gestation, visits, systolic, diastolic):
    """Exploratory monitoring score used by the sidebar calculator."""
    score = 0
    flags = []

    if age < 20:
        score += 1
        flags.append("Age below 20")
    elif age >= 35:
        score += 1
        flags.append("Age 35 or above")

    if systolic >= 140:
        score += 2
        flags.append("Elevated systolic BP")

    if diastolic >= 90:
        score += 2
        flags.append("Elevated diastolic BP")

    if parity >= 4:
        score += 1
        flags.append("High parity")

    if visits <= 1:
        score += 1
        flags.append("Low recorded ANC visits")

    if gestation > 42:
        flags.append("Implausible gestational age")

    if score >= 4:
        category = "High monitoring concern"
    elif score >= 2:
        category = "Moderate monitoring concern"
    else:
        category = "Lower monitoring concern"

    return score, category, flags


def logo_html():
    for path in ["wananchi_logo.png", "logo.png", "wananchi_logo.jpg", "logo.jpg"]:
        if os.path.exists(path):
            mime = "image/png" if path.endswith(".png") else "image/jpeg"
            with open(path, "rb") as handle:
                encoded = base64.b64encode(handle.read()).decode()
            return (
                f'<img class="brand-logo" '
                f'src="data:{mime};base64,{encoded}" '
                f'alt="Wananchi Hospital logo">'
            )
    return ""


REPORT_CSS = (
    "body{font-family:Arial,Helvetica,sans-serif;color:#27323F;margin:32px;}"
    "h1{color:#fff;background:#1F4E9C;border-bottom:5px solid #C8102E;"
    "padding:16px 20px;font-size:22px;}"
    "h2{color:#14346B;border-bottom:3px solid #2A7F3B;padding-bottom:4px;"
    "font-size:16px;margin-top:26px;}"
    "table.t{border-collapse:collapse;width:100%;font-size:13px;}"
    "table.t th{background:#1F4E9C;color:#fff;text-align:left;padding:7px 9px;}"
    "table.t td{border-bottom:1px solid #DDE4EE;padding:6px 9px;}"
    ".note{background:#EAF0FA;border-left:5px solid #1F4E9C;padding:10px 14px;"
    "font-size:12px;margin-top:22px;}"
    "@media print{body{margin:12mm;}"
    "h1,table.t th{-webkit-print-color-adjust:exact;print-color-adjust:exact;}}"
)


def build_report(data, filters_text, indicators):
    summary = summarise(data)
    summary_df = pd.DataFrame({
        "Measure": list(summary.keys()),
        "Value": [
            round(v, 1) if isinstance(v, float) else v
            for v in summary.values()
        ],
    })

    categories = data["Monitoring_Category"].value_counts().reset_index()
    categories.columns = ["Monitoring category", "Records"]
    categories["Share (%)"] = (
        100 * categories["Records"] / categories["Records"].sum()
    ).round(1)

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Wananchi ANC summary report</title>",
        f"<style>{REPORT_CSS}</style></head><body>",
        "<h1>Wananchi Hospital ANC Summary Report</h1>",
        f"<p>Generated on {date.today().isoformat()}.</p>",
        f"<p><b>Filters applied:</b> {html.escape(filters_text)}</p>",
        "<h2>Headline measures</h2>",
        summary_df.to_html(index=False, border=0, classes="t", escape=True),
        "<h2>Programme indicators</h2>",
        indicators.to_html(index=False, border=0, classes="t", escape=True)
        if len(indicators) > 0 else "<p>No indicator data.</p>",
        "<h2>Monitoring categories</h2>",
        categories.to_html(index=False, border=0, classes="t", escape=True),
        "<div class='note'>This is an exploratory summary of routinely "
        "recorded ANC data. The monitoring framework is not a validated "
        "clinical diagnostic or adverse-outcome prediction tool. Targets "
        "shown are the values set in the dashboard at the time of export.</div>",
        "</body></html>",
    ]
    return "".join(parts)


# ============================================================
# OPTIONAL PASSWORD (set app_password in .streamlit/secrets.toml)
# ============================================================

def require_password():
    try:
        expected = st.secrets["app_password"]
    except Exception:
        return

    if st.session_state.get("authenticated"):
        return

    st.markdown(
        '<div class="brand-banner"><div><h1>Wananchi ANC Monitoring</h1>'
        '<p>Please sign in to view the dashboard.</p></div></div>',
        unsafe_allow_html=True
    )
    with st.form("login"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")

    if submitted:
        if password == expected:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


require_password()


# ============================================================
# LOAD AND PREPARE DATA
# ============================================================

@st.cache_data
def load_data(path, modified_time):
    # modified_time is part of the cache key, so replacing the CSV
    # refreshes the dashboard without restarting the app.
    return pd.read_csv(path)


@st.cache_data
def prepare_data(raw):
    data = raw.copy()

    data["Year"] = pd.to_numeric(data["Year"], errors="coerce")
    data = data.dropna(subset=["Year"]).copy()
    data["Year"] = data["Year"].astype(int)
    # Convert numerical ANC variables to numeric
numeric_columns = [
    "Age",
    "Visit Count",
    "Systolic BP",
    "Diastolic BP",
    "Weight (kg)",
    "Gestation Weeks",
    "Gravidity",
]

for col in numeric_columns:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    age_group = pd.cut(
        data["Age"],
        bins=[0, 20, 25, 30, 35, 200],
        labels=["Under 20", "20-24", "25-29", "30-34", "35 and above"],
        right=False
    ).astype(object)
    data["Age Group"] = age_group.where(age_group.notna(), "Unknown")

    data["Trimester"] = data["Gestation Weeks"].map(trimester_label)

    month = extract_month(data)
    if month is not None:
        data["Month"] = month

    if "Monitoring_Category" not in data.columns:
        parity = (
            data["Parity_clean"]
            if "Parity_clean" in data.columns
            else pd.Series(0, index=data.index)
        )
        score = (
            ((data["Age"] < 20) | (data["Age"] >= 35)).astype(int)
            + 2 * (data["Systolic BP"] >= 140).astype(int)
            + 2 * (data["Diastolic BP"] >= 90).astype(int)
            + (parity >= 4).astype(int)
            + (data["Visit Count"] <= 1).astype(int)
        )
        data["Monitoring_Category"] = np.select(
            [score >= 4, score >= 2],
            ["High monitoring concern", "Moderate monitoring concern"],
            default="Lower monitoring concern"
        )
    else:
        data["Monitoring_Category"] = (
            data["Monitoring_Category"].fillna("Not assigned")
        )
    return data
if not os.path.exists(DATA_FILE):
    st.error(
        f"The data file '{DATA_FILE}' was not found. Place it in the same "
        "folder as app.py and reload the page."
    )
    st.stop()

try:
    raw_df = load_data(DATA_FILE, os.path.getmtime(DATA_FILE))
except Exception as error:
    st.error(f"The data file could not be read: {error}")
    st.stop()

missing_columns = [c for c in REQUIRED_COLUMNS if c not in raw_df.columns]
if missing_columns:
    st.error(
        "The data file is missing required column(s): "
        + ", ".join(missing_columns)
    )
    st.stop()

df = prepare_data(raw_df)
original_columns = [c for c in raw_df.columns if c in df.columns]

years = sorted(df["Year"].unique())
YEAR_COLORS = {
    str(y): COLORWAY[i % len(COLORWAY)] for i, y in enumerate(years)
}
YEAR_SEQUENCE = [YEAR_COLORS[str(y)] for y in years]


# ============================================================
# TITLE
# ============================================================

st.markdown(
    f"""
    <div class="brand-banner">
        {logo_html()}
        <div>
            <h1>Wananchi Hospital ANC Monitoring Dashboard</h1>
            <p>Exploratory summary of routinely recorded antenatal care (ANC)
            data, January to August 2021 and January to August 2026.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.info(
    "The ANC monitoring framework is exploratory and is not a validated "
    "clinical diagnostic or adverse-outcome prediction tool."
)


# ============================================================
# SIDEBAR: FILTERS
# ============================================================

st.sidebar.header("Dashboard Filters")

selected_years = st.sidebar.multiselect(
    "Year",
    years,
    default=years
)

age_options = [a for a in AGE_ORDER if a in set(df["Age Group"])]
selected_ages = st.sidebar.multiselect(
    "Age group",
    age_options,
    default=age_options
)

tri_options = [t for t in TRIMESTER_ORDER if t in set(df["Trimester"])]
selected_tris = st.sidebar.multiselect(
    "Gestation stage",
    tri_options,
    default=tri_options,
    help="First trimester: up to 13 weeks. Second: 14 to 27 weeks. "
         "Third: 28 to 42 weeks."
)

category_options = sorted(
    df["Monitoring_Category"].unique(), key=category_rank
)
selected_categories = st.sidebar.multiselect(
    "Monitoring category",
    category_options,
    default=category_options
)

visit_min = int(df["Visit Count"].min()) if df["Visit Count"].notna().any() else 0
visit_max = int(df["Visit Count"].max()) if df["Visit Count"].notna().any() else 0
if visit_min < visit_max:
    visit_range = st.sidebar.slider(
        "ANC visit count",
        min_value=visit_min,
        max_value=visit_max,
        value=(visit_min, visit_max)
    )
else:
    visit_range = (visit_min, visit_max)

base_mask = (
    df["Age Group"].isin(selected_ages)
    & df["Trimester"].isin(selected_tris)
    & df["Monitoring_Category"].isin(selected_categories)
    & (
        df["Visit Count"].between(visit_range[0], visit_range[1])
        | df["Visit Count"].isna()
    )
)

# base_df: all filters except year (used for year-to-year comparison)
base_df = df[base_mask].copy()
filtered_df = base_df[base_df["Year"].isin(selected_years)].copy()

filters_text = (
    f"Years {', '.join(str(y) for y in selected_years) or 'none'}; "
    f"age groups {len(selected_ages)} of {len(age_options)}; "
    f"gestation stages {len(selected_tris)} of {len(tri_options)}; "
    f"monitoring categories {len(selected_categories)} of {len(category_options)}; "
    f"visit count {visit_range[0]} to {visit_range[1]}"
)


# ============================================================
# SIDEBAR: INDIVIDUAL MONITORING CALCULATOR
# ============================================================

st.sidebar.markdown(
    '<div class="sidebar-section">Individual Monitoring Calculator</div>',
    unsafe_allow_html=True
)

st.sidebar.caption(
    "Enter example ANC characteristics, then press the button. "
    "The result appears in the ANC Monitoring tab."
)

calc_age = st.sidebar.number_input(
    "Age", min_value=10, max_value=60, value=25,
    help="Maternal age in years."
)
calc_parity = st.sidebar.number_input(
    "Parity", min_value=0, max_value=15, value=1,
    help="Number of previous births."
)
calc_gestation = st.sidebar.number_input(
    "Gestation Weeks", min_value=0, max_value=60, value=24,
    help="Weeks of pregnancy at this visit."
)
calc_visits = st.sidebar.number_input(
    "ANC Visit Count", min_value=0, max_value=50, value=2,
    help="Number of ANC visits recorded so far."
)
calc_weight = st.sidebar.number_input(
    "Weight (kg)", min_value=20.0, max_value=250.0, value=65.0,
    help="Shown for context only. Weight does not add monitoring points."
)
calc_systolic = st.sidebar.number_input(
    "Systolic BP", min_value=50, max_value=250, value=120,
    help="Upper blood pressure reading (mmHg)."
)
calc_diastolic = st.sidebar.number_input(
    "Diastolic BP", min_value=30, max_value=150, value=80,
    help="Lower blood pressure reading (mmHg)."
)

# Live input checks
if calc_age < 15 or calc_age > 49:
    st.sidebar.warning("Age is outside the usual 15 to 49 range.")
if calc_gestation > 42:
    st.sidebar.warning("Gestation above 42 weeks is not plausible.")
if calc_diastolic >= calc_systolic:
    st.sidebar.warning("Diastolic BP should be lower than systolic BP.")
if calc_weight < 30 or calc_weight > 150:
    st.sidebar.warning("Weight is outside the usual 30 to 150 kg range.")

if st.sidebar.button("Assess Monitoring Level"):
    score, category, flags = assess(
        calc_age, calc_parity, calc_gestation,
        calc_visits, calc_systolic, calc_diastolic
    )
    st.session_state["calc_result"] = {
        "score": score,
        "category": category,
        "flags": flags,
        "inputs": (
            f"Age {calc_age}, parity {calc_parity}, "
            f"gestation {calc_gestation} weeks, visits {calc_visits}, "
            f"weight {calc_weight:g} kg, BP {calc_systolic}/{calc_diastolic}"
        ),
    }


# ============================================================
# STOP IF FILTERS REMOVE EVERYTHING
# ============================================================

if filtered_df.empty:
    st.warning(
        "No records match the current filters. Widen the filters in the "
        "sidebar to see results."
    )
    st.stop()


# ============================================================
# KPI CARDS
# ============================================================

elevated_mask = (
    (filtered_df["Systolic BP"] >= 140) | (filtered_df["Diastolic BP"] >= 90)
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "ANC Records",
        f"{len(filtered_df):,}",
        help="Number of records that match the filters."
    )

with col2:
    mean_age = filtered_df["Age"].mean()
    st.metric(
        "Mean Age",
        f"{mean_age:.1f} years" if not pd.isna(mean_age) else "N/A"
    )

with col3:
    mean_visits = filtered_df["Visit Count"].mean()
    st.metric(
        "Mean ANC Visits",
        f"{mean_visits:.1f}" if not pd.isna(mean_visits) else "N/A"
    )

with col4:
    st.metric(
        "Elevated BP Records",
        f"{int(elevated_mask.sum()):,}",
        help="Systolic BP of 140 or more, or diastolic BP of 90 or more."
    )

st.caption(
    f"Showing {len(filtered_df):,} of {len(df):,} records. "
    "Use the sidebar to change the filters."
)


# ============================================================
# TABS
# ============================================================

(
    tab_overview, tab_compare, tab_indicators, tab_monitoring,
    tab_bp, tab_quality, tab_regression, tab_data, tab_guide
) = st.tabs([
    "Overview",
    "Year Comparison",
    "Indicators and Trends",
    "ANC Monitoring",
    "Elevated BP",
    "Data Quality",
    "Age Regression",
    "Data and Downloads",
    "Guide"
])

plot_df = year_labels(filtered_df)

# Default indicator targets (editable in the Indicators tab)
targets = {
    "ANC 4 or more visits (%)": (60.0, True),
    "First visit at 12 weeks or earlier (%)": (50.0, True),
    "Elevated BP records (%)": (10.0, False),
}


# ============================================================
# TAB: OVERVIEW
# ============================================================

with tab_overview:

    st.header("Overview")

    st.subheader("Age Distribution")

    fig_age = px.histogram(
        plot_df,
        x="Age",
        color="Year",
        nbins=20,
        barmode="overlay",
        opacity=0.75,
        title="Age Distribution by Year",
        color_discrete_map=YEAR_COLORS
    )
    show_chart(fig_age, key="ov_age")

    st.subheader("Gestational Age")

    fig_gestation = px.histogram(
        plot_df,
        x="Gestation Weeks",
        color="Year",
        nbins=20,
        barmode="overlay",
        opacity=0.75,
        title="Gestational Age Distribution",
        color_discrete_map=YEAR_COLORS
    )
    show_chart(fig_gestation, key="ov_gest")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Age Groups")
        age_share = share_by(plot_df, "Age Group")
        fig_age_group = px.bar(
            age_share,
            x="Age Group",
            y="Share (%)",
            color="Year",
            barmode="group",
            text_auto=".1f",
            title="Share of Records by Age Group",
            category_orders={"Age Group": AGE_ORDER},
            color_discrete_map=YEAR_COLORS
        )
        show_chart(fig_age_group, key="ov_agegroup")

    with col2:
        st.subheader("Gestation Stage")
        tri_share = share_by(plot_df, "Trimester")
        fig_tri = px.bar(
            tri_share,
            x="Trimester",
            y="Share (%)",
            color="Year",
            barmode="group",
            text_auto=".1f",
            title="Share of Records by Gestation Stage",
            category_orders={"Trimester": TRIMESTER_ORDER},
            color_discrete_map=YEAR_COLORS
        )
        show_chart(fig_tri, key="ov_tri")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ANC Visit Count")

        visit_counts = (
            plot_df
            .groupby(["Year", "Visit Count"])
            .size()
            .reset_index(name="Records")
        )

        fig_visits = px.bar(
            visit_counts,
            x="Visit Count",
            y="Records",
            color="Year",
            barmode="group",
            title="ANC Visits by Year",
            color_discrete_map=YEAR_COLORS
        )
        show_chart(fig_visits, key="ov_visits")

    with col2:
        st.subheader("Weight Distribution")

        fig_weight = px.box(
            plot_df,
            x="Year",
            y="Weight (kg)",
            color="Year",
            points="outliers",
            title="Weight Distribution by Year",
            color_discrete_map=YEAR_COLORS
        )
        fig_weight.update_layout(showlegend=False)
        show_chart(fig_weight, key="ov_weight")

    st.subheader("Blood Pressure")

    bp_df = plot_df[
        ["Year", "Systolic BP", "Diastolic BP"]
    ].melt(
        id_vars="Year",
        var_name="Blood Pressure",
        value_name="Value"
    )

    fig_bp = px.box(
        bp_df,
        x="Year",
        y="Value",
        color="Blood Pressure",
        title="Blood Pressure Distribution by Year",
        color_discrete_map={"Systolic BP": RED, "Diastolic BP": BLUE}
    )
    show_chart(fig_bp, key="ov_bp")


# ============================================================
# TAB: YEAR COMPARISON
# ============================================================

with tab_compare:

    st.header("Year Comparison")

    st.markdown(
        "Compares two years using all sidebar filters except Year. "
        "Positive change means the comparison year is higher."
    )

    compare_years = sorted(base_df["Year"].unique())

    if len(compare_years) < 2:
        st.info(
            "At least two years with records are needed for a comparison. "
            "Widen the other filters if one year is empty."
        )
    else:
        col1, col2 = st.columns(2)
        with col1:
            year_a = st.selectbox(
                "Baseline year", compare_years, index=0
            )
        with col2:
            year_b = st.selectbox(
                "Comparison year", compare_years, index=len(compare_years) - 1
            )

        if year_a == year_b:
            st.warning("Choose two different years to compare.")
        else:
            data_a = base_df[base_df["Year"] == year_a]
            data_b = base_df[base_df["Year"] == year_b]
            sum_a = summarise(data_a)
            sum_b = summarise(data_b)

            rows = []
            for measure in sum_a:
                a_val = sum_a[measure]
                b_val = sum_b[measure]
                change = b_val - a_val
                rows.append({
                    "Measure": measure,
                    str(year_a): round(a_val, 1),
                    str(year_b): round(b_val, 1),
                    "Change": round(change, 1),
                    "Direction": (
                        "Higher" if change > 0
                        else "Lower" if change < 0
                        else "No change"
                    ),
                })

            show_table(pd.DataFrame(rows))

            pct_measures = [
                "Under 20 (%)",
                "35 and above (%)",
                "4 or more visits (%)",
                "Elevated BP (%)",
            ]
            pct_rows = []
            for measure in pct_measures:
                pct_rows.append({
                    "Measure": measure, "Year": str(year_a),
                    "Percent": sum_a[measure]
                })
                pct_rows.append({
                    "Measure": measure, "Year": str(year_b),
                    "Percent": sum_b[measure]
                })

            fig_pct = px.bar(
                pd.DataFrame(pct_rows),
                x="Measure",
                y="Percent",
                color="Year",
                barmode="group",
                text_auto=".1f",
                title="Key Percentages by Year",
                color_discrete_map=YEAR_COLORS
            )
            show_chart(fig_pct, key="cmp_pct")

            col1, col2 = st.columns(2)
            pair_df = year_labels(
                base_df[base_df["Year"].isin([year_a, year_b])]
            )

            with col1:
                fig_cmp_age = px.bar(
                    share_by(pair_df, "Age Group"),
                    x="Age Group",
                    y="Share (%)",
                    color="Year",
                    barmode="group",
                    text_auto=".1f",
                    title="Age Group Mix",
                    category_orders={"Age Group": AGE_ORDER},
                    color_discrete_map=YEAR_COLORS
                )
                show_chart(fig_cmp_age, key="cmp_age")

            with col2:
                fig_cmp_tri = px.bar(
                    share_by(pair_df, "Trimester"),
                    x="Trimester",
                    y="Share (%)",
                    color="Year",
                    barmode="group",
                    text_auto=".1f",
                    title="Gestation Stage Mix",
                    category_orders={"Trimester": TRIMESTER_ORDER},
                    color_discrete_map=YEAR_COLORS
                )
                show_chart(fig_cmp_tri, key="cmp_tri")


# ============================================================
# TAB: INDICATORS AND TRENDS
# ============================================================

with tab_indicators:

    st.header("Indicators and Trends")

    st.subheader("Programme Indicators Against Targets")

    st.caption(
        "The default targets below are placeholders. Set them to your "
        "programme or Ministry of Health targets. 'First visit at 12 weeks "
        "or earlier' uses records where the visit count is 1."
    )

    tcol1, tcol2, tcol3 = st.columns(3)

    with tcol1:
        target_anc4 = st.number_input(
            "Target: ANC 4 or more visits (%)",
            min_value=0.0, max_value=100.0, value=60.0, step=1.0
        )
    with tcol2:
        target_early = st.number_input(
            "Target: first visit by 12 weeks (%)",
            min_value=0.0, max_value=100.0, value=50.0, step=1.0
        )
    with tcol3:
        target_bp = st.number_input(
            "Limit: elevated BP records (%)",
            min_value=0.0, max_value=100.0, value=10.0, step=1.0
        )

    targets = {
        "ANC 4 or more visits (%)": (target_anc4, True),
        "First visit at 12 weeks or earlier (%)": (target_early, True),
        "Elevated BP records (%)": (target_bp, False),
    }

    indicators = indicator_table(filtered_df, targets)

    show_table(indicators)

    chart_cols = st.columns(len(targets))
    for col, (name, (target_value, _)) in zip(chart_cols, targets.items()):
        subset = indicators[indicators["Indicator"] == name].dropna(
            subset=["Value (%)"]
        )
        fig = go.Figure()
        for _, row in subset.iterrows():
            fig.add_bar(
                x=[str(row["Year"])],
                y=[row["Value (%)"]],
                name=str(row["Year"]),
                marker_color=YEAR_COLORS.get(str(row["Year"]), BLUE),
                text=[f"{row['Value (%)']:.1f}%"],
                textposition="outside"
            )
        fig.add_hline(
            y=target_value,
            line_dash="dash",
            line_color=TEXT,
            annotation_text=f"Target {target_value:g}%",
            annotation_font_color=TEXT,
            annotation_position="top left"
        )
        top = max(
            [target_value] + list(subset["Value (%)"])
        ) if len(subset) > 0 else target_value
        fig.update_yaxes(range=[0, top * 1.3 + 5], title_text="Percent")
        fig.update_layout(title=name, showlegend=False, title_font_size=13)
        with col:
            show_chart(fig, key=f"ind_{name}")

    st.subheader("Records by Month")

    if "Month" in filtered_df.columns and filtered_df["Month"].notna().any():
        month_data = year_labels(filtered_df.dropna(subset=["Month"]))
        month_counts = (
            month_data
            .groupby(["Year", "Month"])
            .size()
            .reset_index(name="Records")
            .sort_values("Month")
        )
        month_counts["Month name"] = month_counts["Month"].astype(int).map(
            lambda i: MONTH_NAMES[i - 1]
        )
        fig_month = px.line(
            month_counts,
            x="Month name",
            y="Records",
            color="Year",
            markers=True,
            title="ANC Records per Month",
            category_orders={"Month name": MONTH_NAMES},
            color_discrete_map=YEAR_COLORS
        )
        show_chart(fig_month, key="ind_month")
    else:
        st.info(
            "No date or month column was found in the dataset, so monthly "
            "trends cannot be shown. Add a column named 'Date' or 'Month' "
            "to the CSV to switch this chart on."
        )


# ============================================================
# TAB: ANC MONITORING
# ============================================================

with tab_monitoring:

    st.header("Exploratory ANC Monitoring Framework")

    st.markdown(
        """
        The framework assigns monitoring points to selected characteristics
        recorded in the ANC register. It is intended for exploratory
        monitoring only and does not replace clinical assessment.
        Green marks lower concern, blue moderate concern and red high concern.
        """
    )

    category_counts = (
        filtered_df["Monitoring_Category"]
        .value_counts()
        .reset_index()
    )
    category_counts.columns = ["Monitoring Category", "Records"]
    category_counts["Order"] = category_counts["Monitoring Category"].map(
        category_rank
    )
    category_counts = category_counts.sort_values("Order")

    fig_monitoring = px.bar(
        category_counts,
        x="Monitoring Category",
        y="Records",
        color="Monitoring Category",
        text_auto=True,
        title="Monitoring Categories",
        category_orders={
            "Monitoring Category": list(category_counts["Monitoring Category"])
        },
        color_discrete_map={
            c: category_color(c) for c in category_counts["Monitoring Category"]
        }
    )
    fig_monitoring.update_layout(showlegend=False)
    show_chart(fig_monitoring, key="mon_cat")

    st.subheader("Monitoring Summary by Year")

    monitoring_summary = (
        filtered_df
        .groupby(["Year", "Monitoring_Category"])
        .size()
        .reset_index(name="Records")
    )
    monitoring_summary["Share of year (%)"] = (
        100 * monitoring_summary["Records"] /
        monitoring_summary.groupby("Year")["Records"].transform("sum")
    ).round(1)
    monitoring_summary["Order"] = monitoring_summary[
        "Monitoring_Category"
    ].map(category_rank)
    monitoring_summary = monitoring_summary.sort_values(
        ["Year", "Order"]
    ).drop(columns="Order")

    show_table(monitoring_summary)

    st.subheader("Individual Monitoring Result")

    result = st.session_state.get("calc_result")

    if result is None:
        st.info(
            "Enter ANC characteristics in the sidebar and press "
            "'Assess Monitoring Level' to see the result here."
        )
    else:
        color = category_color(result["category"])
        st.markdown(
            f"""
            <div class="result-card" style="border-left: 8px solid {color};">
                <div class="result-label">Monitoring category</div>
                <div class="result-value" style="color: {color};">
                    {html.escape(result['category'])}
                </div>
                <div class="result-sub">
                    Monitoring score: <b>{result['score']}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.caption(f"Inputs used: {result['inputs']}")

        if result["flags"]:
            st.write("**Monitoring flags:**")
            for flag in result["flags"]:
                st.warning(flag)
        else:
            st.success(
                "No monitoring flags were triggered by this framework."
            )

    st.caption(
        "Note: Weight is displayed as contextual ANC information but is "
        "not currently assigned points in this exploratory scoring rule."
    )


# ============================================================
# TAB: ELEVATED BP
# ============================================================

with tab_bp:

    st.header("Elevated Blood Pressure")

    st.markdown(
        "Elevated means systolic BP of 140 or more, or diastolic BP of 90 "
        "or more. Use the controls to list and sort the matching records."
    )

    sys_high = filtered_df["Systolic BP"] >= 140
    dia_high = filtered_df["Diastolic BP"] >= 90
    any_high = sys_high | dia_high

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Elevated records", f"{int(any_high.sum()):,}")
    with col2:
        st.metric(
            "Share of records",
            f"{100 * any_high.mean():.1f}%"
        )
    with col3:
        st.metric("Systolic only", f"{int((sys_high & ~dia_high).sum()):,}")
    with col4:
        st.metric("Diastolic only", f"{int((~sys_high & dia_high).sum()):,}")

    st.write("")

    bp_rates = (
        year_labels(filtered_df.assign(Elevated=any_high))
        .groupby(["Age Group", "Year"])["Elevated"]
        .mean()
        .mul(100)
        .round(1)
        .reset_index(name="Elevated BP (%)")
    )
    fig_bp_age = px.bar(
        bp_rates,
        x="Age Group",
        y="Elevated BP (%)",
        color="Year",
        barmode="group",
        text_auto=".1f",
        title="Elevated BP by Age Group and Year",
        category_orders={"Age Group": AGE_ORDER},
        color_discrete_map=YEAR_COLORS
    )
    show_chart(fig_bp_age, key="bp_age")

    st.subheader("Matching Records")

    col1, col2 = st.columns(2)
    with col1:
        bp_type = st.selectbox(
            "Show",
            [
                "Any elevated reading",
                "Both elevated",
                "Systolic only",
                "Diastolic only",
            ]
        )
    with col2:
        sort_by = st.selectbox(
            "Sort by",
            [
                "Systolic BP (highest first)",
                "Diastolic BP (highest first)",
                "Age (oldest first)",
                "Gestation Weeks (latest first)",
            ]
        )

    if bp_type == "Both elevated":
        type_mask = sys_high & dia_high
    elif bp_type == "Systolic only":
        type_mask = sys_high & ~dia_high
    elif bp_type == "Diastolic only":
        type_mask = ~sys_high & dia_high
    else:
        type_mask = any_high

    sort_column = {
        "Systolic BP (highest first)": "Systolic BP",
        "Diastolic BP (highest first)": "Diastolic BP",
        "Age (oldest first)": "Age",
        "Gestation Weeks (latest first)": "Gestation Weeks",
    }[sort_by]

    display_columns = [
        c for c in [
            "Year", "Age", "Age Group", "Gestation Weeks", "Visit Count",
            "Systolic BP", "Diastolic BP", "Monitoring_Category"
        ] if c in filtered_df.columns
    ]

    bp_records = (
        filtered_df.loc[type_mask, display_columns]
        .sort_values(sort_column, ascending=False)
    )

    st.caption(f"{len(bp_records):,} record(s) shown.")
    show_table(bp_records)

    st.download_button(
        "Download these records (CSV)",
        bp_records.to_csv(index=False).encode("utf-8"),
        file_name="wananchi_elevated_bp_records.csv",
        mime="text/csv",
        key="dl_bp"
    )


# ============================================================
# TAB: DATA QUALITY
# ============================================================

with tab_quality:

    st.header("Data Quality Assessment")

    st.markdown(
        """
        This section describes completeness, duplication and selected
        validity checks in the analytical dataset. Flags are intended for
        data-quality investigation and do not automatically imply that
        individual records are incorrect.
        """
    )

    quality_df_source = filtered_df[original_columns]

    total_records = len(quality_df_source)
    duplicate_records = quality_df_source.duplicated().sum()
    total_missing = quality_df_source.isna().sum().sum()
    total_cells = quality_df_source.shape[0] * quality_df_source.shape[1]
    completeness = (
        100 * (1 - total_missing / total_cells) if total_cells > 0 else 0
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Records", f"{total_records:,}")
    with col2:
        st.metric("Duplicate Rows", int(duplicate_records))
    with col3:
        st.metric("Missing Cells", int(total_missing))
    with col4:
        st.metric("Overall Completeness", f"{completeness:.1f}%")

    st.subheader("Missing Values by Variable")

    missing_table = pd.DataFrame({
        "Variable": quality_df_source.columns,
        "Missing Values": quality_df_source.isna().sum().values
    })
    missing_table["Missing (%)"] = (
        missing_table["Missing Values"] / max(len(quality_df_source), 1) * 100
    ).round(2)
    missing_table = missing_table.sort_values("Missing (%)", ascending=False)

    fig_missing = px.bar(
        missing_table,
        x="Variable",
        y="Missing (%)",
        title="Percentage of Missing Values",
        color_discrete_sequence=[RED]
    )
    fig_missing.update_layout(xaxis_tickangle=-45)
    show_chart(fig_missing, key="dq_missing")

    show_table(missing_table)

    st.subheader("Duplicate Records")

    if duplicate_records > 0:
        st.warning(f"{duplicate_records} duplicate row(s) detected.")
    else:
        st.success("No exact duplicate rows detected in the selected data.")

    st.subheader("Selected Validity Checks")

    checks = [
        ("Age below 15 or above 49", "Age",
         lambda s: (s < 15) | (s > 49)),
        ("Gestation above 42 weeks", "Gestation Weeks",
         lambda s: s > 42),
        ("Weight below 30 kg or above 150 kg", "Weight (kg)",
         lambda s: (s < 30) | (s > 150)),
        ("Systolic BP below 70 or above 200", "Systolic BP",
         lambda s: (s < 70) | (s > 200)),
        ("Diastolic BP below 40 or above 120", "Diastolic BP",
         lambda s: (s < 40) | (s > 120)),
    ]

    quality_rows = []
    flagged_mask = pd.Series(False, index=filtered_df.index)
    for label, column, rule in checks:
        if column in filtered_df.columns:
            mask = rule(filtered_df[column]).fillna(False)
            flagged_mask = flagged_mask | mask
            quality_rows.append({
                "Check": label,
                "Records Flagged": int(mask.sum())
            })

    show_table(pd.DataFrame(quality_rows))

    st.info(
        "Validity thresholds shown here are screening thresholds for "
        "data-quality review. They should not be interpreted as clinical "
        "diagnostic thresholds."
    )

    st.download_button(
        "Download flagged records (CSV)",
        filtered_df.loc[flagged_mask, original_columns]
        .to_csv(index=False).encode("utf-8"),
        file_name="wananchi_flagged_records.csv",
        mime="text/csv",
        key="dl_flagged"
    )


# ============================================================
# TAB: AGE REGRESSION
# ============================================================

@st.cache_data(show_spinner=False)
def run_regression(data):
    terms = []
    if "Parity_clean" in data.columns:
        terms.append("Parity_clean")
    if "Gravidity" in data.columns:
        terms.append("Gravidity")
    terms += [
        "Q('Gestation Weeks')",
        "Q('Visit Count')",
        "Q('Weight (kg)')",
        "Q('Systolic BP')",
        "Q('Diastolic BP')",
        "Year_2026",
    ]

    needed = [
        c for c in [
            "Age", "Parity_clean", "Gravidity", "Gestation Weeks",
            "Visit Count", "Weight (kg)", "Systolic BP", "Diastolic BP", "Year"
        ] if c in data.columns
    ]

    reg_df = data[needed].dropna().copy()
    if len(reg_df) < 20:
        return None

    reg_df["Year_2026"] = (reg_df["Year"] == 2026).astype(int)

    formula = "Age ~ " + " + ".join(terms)
    model = smf.ols(formula, data=reg_df).fit()

    predictions = model.predict(reg_df)
    actual = reg_df["Age"]
    errors = np.abs(actual - predictions)

    return {
        "n": len(reg_df),
        "actual": actual,
        "predictions": predictions,
        "resid": model.resid,
        "params": model.params,
        "pvalues": model.pvalues,
        "summary": model.summary().as_text(),
        "mae": mean_absolute_error(actual, predictions),
        "rmse": float(np.sqrt(mean_squared_error(actual, predictions))),
        "r2": r2_score(actual, predictions),
        "within": {k: float((errors <= k).mean() * 100) for k in (2, 3, 5, 10)},
        "mape": float(np.mean(np.abs((actual - predictions) / actual)) * 100),
    }


with tab_regression:

    st.header("Exploratory Multiple Linear Regression")

    st.markdown(
        """
        This analysis models recorded maternal age using selected ANC
        characteristics. It is an exploratory statistical analysis and
        should not be interpreted as a clinical risk-prediction model.
        The model is fitted on the full dataset and does not change with
        the sidebar filters.
        """
    )

    try:
        reg = run_regression(df)
    except Exception as error:
        reg = None
        st.error(f"Regression analysis could not be displayed: {error}")

    if reg is None:
        st.info(
            "There are not enough complete records to fit the regression."
        )
    else:
        st.subheader("Regression Performance")
        st.caption(f"Fitted on {reg['n']:,} complete records.")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MAE", f"{reg['mae']:.2f} years")
        with col2:
            st.metric("RMSE", f"{reg['rmse']:.2f} years")
        with col3:
            st.metric("R²", f"{reg['r2']:.3f}")

        st.write("")

        col1, col2, col3, col4 = st.columns(4)
        for col, k in zip([col1, col2, col3, col4], (2, 3, 5, 10)):
            with col:
                st.metric(f"Within ±{k} years", f"{reg['within'][k]:.1f}%")

        st.write("")
        st.metric("MAPE", f"{reg['mape']:.2f}%")

        st.info(
            f"""
            **Interpretation:** The exploratory model has an R² of
            {reg['r2']:.3f}, meaning that approximately {reg['r2'] * 100:.1f}%
            of the variation in recorded age is explained by the included
            predictors in this fitted dataset. The MAE is approximately
            {reg['mae']:.2f} years.

            The percentages reported above describe the proportion of
            observations whose predicted age falls within the stated
            error range. They should not be interpreted as clinical
            prediction accuracy.
            """
        )

        st.subheader("Actual vs Predicted Age")

        prediction_df = pd.DataFrame({
            "Actual Age": reg["actual"],
            "Predicted Age": reg["predictions"]
        })
        fig_prediction = px.scatter(
            prediction_df,
            x="Actual Age",
            y="Predicted Age",
            title="Actual vs Predicted Age",
            color_discrete_sequence=[BLUE]
        )
        low = min(prediction_df["Actual Age"].min(),
                  prediction_df["Predicted Age"].min())
        high = max(prediction_df["Actual Age"].max(),
                   prediction_df["Predicted Age"].max())
        fig_prediction.add_shape(
            type="line", x0=low, y0=low, x1=high, y1=high,
            line=dict(color=RED, width=2, dash="dash")
        )
        show_chart(fig_prediction, key="reg_pred")

        st.subheader("Residual Analysis")

        residual_df = pd.DataFrame({
            "Fitted Age": reg["predictions"],
            "Residual": reg["actual"] - reg["predictions"]
        })
        fig_residual = px.scatter(
            residual_df,
            x="Fitted Age",
            y="Residual",
            title="Residuals vs Fitted Values",
            color_discrete_sequence=[GREEN]
        )
        fig_residual.add_hline(y=0, line_color=RED, line_width=2)
        show_chart(fig_residual, key="reg_resid")

        st.subheader("Normal Q-Q Plot")

        qq_fig = sm.qqplot(reg["resid"], line="45", fit=True)
        qq_fig.set_facecolor("white")
        qq_ax = qq_fig.axes[0]
        qq_ax.set_facecolor("white")
        qq_ax.title.set_color(BLUE_DARK)
        for line in qq_ax.get_lines():
            if line.get_linestyle() == "None" or line.get_marker() != "None":
                line.set_markerfacecolor(BLUE)
                line.set_markeredgecolor(BLUE)
            else:
                line.set_color(RED)
        st.pyplot(qq_fig, clear_figure=True)

        st.subheader("Regression Coefficients")

        coefficient_table = pd.DataFrame({
            "Variable": reg["params"].index,
            "Coefficient": reg["params"].values.round(4),
            "P-value": reg["pvalues"].values.round(4)
        })
        show_table(coefficient_table)

        if st.checkbox("Show full regression model summary"):
            st.text(reg["summary"])

        st.warning(
            """
            This regression uses Age as the dependent variable because
            the available ANC register does not contain a reliably linked
            validated maternal or neonatal adverse-outcome variable.
            Therefore, the model should be presented as exploratory
            statistical analysis rather than clinical risk prediction.
            """
        )


# ============================================================
# TAB: DATA AND DOWNLOADS
# ============================================================

with tab_data:

    st.header("Data and Downloads")

    final_indicators = indicator_table(filtered_df, targets)

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "Download filtered data (CSV)",
            filtered_df.to_csv(index=False).encode("utf-8"),
            file_name="wananchi_anc_filtered.csv",
            mime="text/csv",
            key="dl_data"
        )

    with col2:
        st.download_button(
            "Download summary report (HTML)",
            build_report(
                filtered_df, filters_text, final_indicators
            ).encode("utf-8"),
            file_name="wananchi_anc_summary_report.html",
            mime="text/html",
            key="dl_report"
        )

    st.caption(
        "Open the summary report in a browser and use Print, then Save as "
        "PDF, for a one-page handout. Each chart also has a camera icon "
        "when you hover over it, which saves it as an image."
    )

    st.subheader("Summary Statistics")

    numeric_summary = (
        filtered_df[
            [c for c in [
                "Age", "Parity_clean", "Gravidity", "Gestation Weeks",
                "Visit Count", "Weight (kg)", "Systolic BP", "Diastolic BP"
            ] if c in filtered_df.columns]
        ]
        .describe()
        .T
        .reset_index()
        .rename(columns={"index": "Variable"})
        .round(2)
    )
    show_table(numeric_summary)

    st.subheader("Record-Level Data")

    st.caption(
        "Record-level data can contain sensitive patient information. "
        "Only show it when you need it, and do not share it outside the "
        "hospital team."
    )

    if st.checkbox("Show record-level data", value=False):
        show_table(filtered_df)

    st.caption(
        "The dataset excludes Serial Number, HIV Status, TB Screening, "
        "Infant Prophylaxis and MUAC from the analytical dataset."
    )


# ============================================================
# TAB: GUIDE
# ============================================================

with tab_guide:

    st.header("How to Read This Dashboard")

    st.subheader("Using the dashboard")
    st.markdown(
        """
        - **Filters** in the sidebar (year, age group, gestation stage,
          monitoring category and visit count) apply to every tab.
        - **Year Comparison** ignores the Year filter so two years can be
          compared side by side.
        - **Individual Monitoring Calculator** is in the sidebar. Press
          the button and read the result in the ANC Monitoring tab.
        - **Downloads** are in the Data and Downloads tab, and in the
          Elevated BP and Data Quality tabs.
        """
    )

    st.subheader("What each tab shows")
    st.markdown(
        """
        | Tab | What it shows |
        |---|---|
        | Overview | Age, gestation, visits, weight and blood pressure |
        | Year Comparison | Two years side by side, with the change |
        | Indicators and Trends | Indicators against targets and monthly records |
        | ANC Monitoring | Monitoring categories and the individual calculator |
        | Elevated BP | Records with elevated blood pressure |
        | Data Quality | Missing values, duplicates and implausible values |
        | Age Regression | Exploratory statistical model of maternal age |
        | Data and Downloads | CSV and report downloads, summary statistics |
        """
    )

    st.subheader("Monitoring points")
    st.markdown(
        """
        | Characteristic | Points |
        |---|---|
        | Age below 20, or 35 and above | 1 |
        | Systolic BP 140 or more | 2 |
        | Diastolic BP 90 or more | 2 |
        | Parity 4 or more | 1 |
        | One or no recorded ANC visits | 1 |

        A score of 4 or more is **high** concern, 2 or 3 is **moderate**
        concern and below 2 is **lower** concern. Weight does not add points.
        Gestation above 42 weeks is flagged as implausible but adds no points.
        """
    )

    st.subheader("Terms")
    st.markdown(
        """
        - **ANC:** antenatal care, the care a woman receives during pregnancy.
        - **Parity:** the number of previous births.
        - **Gravidity:** the number of pregnancies, including the current one.
        - **Gestation weeks:** weeks of pregnancy at the visit.
        - **Trimester:** first is up to 13 weeks, second is 14 to 27 weeks and
          third is 28 weeks onward.
        - **Elevated BP:** systolic 140 or more, or diastolic 90 or more.
        - **MAE:** mean absolute error, the average gap between actual and
          predicted age, in years.
        - **RMSE:** root mean squared error, like MAE but it penalises large
          gaps more.
        - **R²:** the share of variation in age explained by the model.
        - **MAPE:** mean absolute percentage error.
        - **Within ± years:** the share of records predicted within that many
          years of the recorded age.
        """
    )

    st.info(
        "This is an exploratory academic and research tool. It is not a "
        "clinical diagnostic system."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Wananchi Hospital ANC Dashboard | Exploratory academic/research tool | "
    "Not a clinical diagnostic system"
)
