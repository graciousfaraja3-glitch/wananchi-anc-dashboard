import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Wananchi ANC Monitoring",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Wananchi Hospital ANC Monitoring Dashboard")

st.markdown(
    "This dashboard provides an exploratory summary of routinely recorded "
    "antenatal care (ANC) data from Wananchi Hospital for January–August "
    "2021 and January–August 2026."
)

st.info(
    "⚠️ The ANC monitoring framework is exploratory and is not a validated "
    "clinical diagnostic or adverse-outcome prediction tool."
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv("wananchi_anc_data.csv")

    numeric_columns = [
        "Year",
        "Age",
        "Parity_clean",
        "Gravidity",
        "Gestation Weeks",
        "Visit Count",
        "Weight (kg)",
        "Systolic BP",
        "Diastolic BP",
        "IPT Doses",
        "Monitoring_Score"
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Reconstruct Parity_clean if only the original Parity column exists
    if "Parity_clean" not in df.columns and "Parity" in df.columns:
        parity_original = df["Parity"].astype(str).str.strip()

        df["Parity_clean"] = pd.to_numeric(
            parity_original.str.extract(
                r"^(\d+)",
                expand=False
            ),
            errors="coerce"
        )

    # Reconstruct Parity_Group if needed
    if "Parity_Group" not in df.columns and "Parity_clean" in df.columns:

        def create_parity_group(x):
            if pd.isna(x):
                return np.nan
            elif x == 0:
                return "0"
            elif x == 1:
                return "1"
            elif x == 2:
                return "2"
            else:
                return "3+"

        df["Parity_Group"] = df["Parity_clean"].apply(
            create_parity_group
        )

    # Reconstruct age groups if needed
    if "Age_Group" not in df.columns and "Age" in df.columns:

        bins = [0, 19, 24, 29, 34, 39, 100]
        labels = [
            "<20",
            "20-24",
            "25-29",
            "30-34",
            "35-39",
            "40+"
        ]

        df["Age_Group"] = pd.cut(
            df["Age"],
            bins=bins,
            labels=labels,
            right=True
        )

    return df


@st.cache_resource
def load_model():
    try:
        return joblib.load(
            "wananchi_age_regression_model.pkl"
        )
    except Exception:
        return None


df = load_data()
age_model = load_model()


# ============================================================
# MONITORING FRAMEWORK
# ============================================================

def calculate_monitoring(row):

    score = 0
    flags = []

    age = row.get("Age")
    parity = row.get("Parity_clean")
    gestation = row.get("Gestation Weeks")
    visits = row.get("Visit Count")
    systolic = row.get("Systolic BP")
    diastolic = row.get("Diastolic BP")

    if pd.notna(age):

        if age < 20:
            score += 1
            flags.append("Age below 20")

        elif age >= 35:
            score += 1
            flags.append("Age 35 or above")

    if pd.notna(systolic) and systolic >= 140:
        score += 2
        flags.append("Elevated systolic BP")

    if pd.notna(diastolic) and diastolic >= 90:
        score += 2
        flags.append("Elevated diastolic BP")

    if pd.notna(parity) and parity >= 4:
        score += 1
        flags.append("High parity")

    if pd.notna(visits) and visits <= 1:
        score += 1
        flags.append("Low recorded ANC visits")

    if pd.notna(gestation):

        if gestation > 42:
            flags.append("Implausible gestational age")

        elif gestation < 0:
            flags.append("Invalid gestational age")

    if score >= 4:
        category = "High monitoring concern"

    elif score >= 2:
        category = "Moderate monitoring concern"

    else:
        category = "Lower monitoring concern"

    return pd.Series({
        "Monitoring_Score": score,
        "Monitoring_Category": category,
        "Monitoring_Flags": "; ".join(flags)
    })


monitoring_results = df.apply(
    calculate_monitoring,
    axis=1
)

df = pd.concat(
    [df, monitoring_results],
    axis=1
)

filtered_df = df.copy()


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("Dashboard Filters")

if "Year" in df.columns:

    years = sorted(
        df["Year"].dropna().unique().tolist()
    )

    selected_years = st.sidebar.multiselect(
        "Select year",
        years,
        default=years
    )

    filtered_df = df[
        df["Year"].isin(selected_years)
    ].copy()

else:

    selected_years = []

    st.sidebar.warning(
        "Year column not found."
    )


# ============================================================
# KPI CARDS
# ============================================================

total_records = len(filtered_df)

if (
    "Age" in filtered_df.columns
    and filtered_df["Age"].notna().any()
):
    mean_age = filtered_df["Age"].mean()
else:
    mean_age = np.nan


if (
    "Visit Count" in filtered_df.columns
    and filtered_df["Visit Count"].notna().any()
):
    mean_visits = filtered_df["Visit Count"].mean()
else:
    mean_visits = np.nan


elevated_bp = 0

if "Systolic BP" in filtered_df.columns:
    elevated_bp += int(
        (filtered_df["Systolic BP"] >= 140).sum()
    )

if "Diastolic BP" in filtered_df.columns:
    elevated_bp += int(
        (filtered_df["Diastolic BP"] >= 90).sum()
    )


k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "ANC Records",
    f"{total_records:,}"
)

k2.metric(
    "Mean Age",
    f"{mean_age:.1f} years"
    if pd.notna(mean_age)
    else "N/A"
)

k3.metric(
    "Mean ANC Visits",
    f"{mean_visits:.1f}"
    if pd.notna(mean_visits)
    else "N/A"
)

k4.metric(
    "Elevated BP Records",
    f"{elevated_bp:,}"
)


# ============================================================
# TABS
# ============================================================

tabs = st.tabs([
    "📊 ANC Dashboard",
    "🚩 ANC Monitoring",
    "📋 Data",
    "🔎 Data Quality",
    "📈 Age Regression"
])


# ============================================================
# TAB 1 — ANC DASHBOARD
# ============================================================

with tabs[0]:

    st.header("ANC Dashboard")

    if filtered_df.empty:

        st.warning(
            "No records match the selected filters."
        )

    else:

        c1, c2 = st.columns(2)

        with c1:

            if "Age" in filtered_df.columns:

                fig = px.histogram(
                    filtered_df,
                    x="Age",
                    nbins=20,
                    title="Age Distribution"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        with c2:

            if "Gestation Weeks" in filtered_df.columns:

                fig = px.histogram(
                    filtered_df,
                    x="Gestation Weeks",
                    nbins=20,
                    title="Gestational Age Distribution"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )


        c3, c4 = st.columns(2)

        with c3:

            if "Visit Count" in filtered_df.columns:

                visit_counts = (
                    filtered_df["Visit Count"]
                    .value_counts(dropna=False)
                    .sort_index()
                    .reset_index()
                )

                visit_counts.columns = [
                    "Visit Count",
                    "Records"
                ]

                fig = px.bar(
                    visit_counts,
                    x="Visit Count",
                    y="Records",
                    title="Recorded ANC Visits"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        with c4:

            if "Weight (kg)" in filtered_df.columns:

                fig = px.box(
                    filtered_df,
                    x=(
                        "Year"
                        if "Year" in filtered_df.columns
                        else None
                    ),
                    y="Weight (kg)",
                    title="Maternal Weight by Year"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )


        c5, c6 = st.columns(2)

        with c5:

            if "Systolic BP" in filtered_df.columns:

                fig = px.box(
                    filtered_df,
                    x=(
                        "Year"
                        if "Year" in filtered_df.columns
                        else None
                    ),
                    y="Systolic BP",
                    title="Systolic Blood Pressure by Year"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        with c6:

            if "Diastolic BP" in filtered_df.columns:

                fig = px.box(
                    filtered_df,
                    x=(
                        "Year"
                        if "Year" in filtered_df.columns
                        else None
                    ),
                    y="Diastolic BP",
                    title="Diastolic Blood Pressure by Year"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )


# ============================================================
# TAB 2 — ANC MONITORING
# ============================================================

with tabs[1]:

    st.header("🚩 ANC Monitoring Framework")

    st.markdown(
        "This section applies a simple, transparent set of monitoring "
        "flags to routinely recorded ANC characteristics. The thresholds "
        "and weights are illustrative and have not been clinically validated."
    )

    if filtered_df.empty:

        st.warning(
            "No records match the selected filters."
        )

    else:

        category_counts = (
            filtered_df["Monitoring_Category"]
            .value_counts()
            .reset_index()
        )

        category_counts.columns = [
            "Category",
            "Records"
        ]

        fig = px.bar(
            category_counts,
            x="Category",
            y="Records",
            title="Monitoring Concern Categories"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


        if "Year" in filtered_df.columns:

            year_summary = (
                filtered_df
                .groupby(
                    [
                        "Year",
                        "Monitoring_Category"
                    ]
                )
                .size()
                .reset_index(
                    name="Records"
                )
            )

            st.subheader(
                "Monitoring Categories by Year"
            )

            st.dataframe(
                year_summary,
                use_container_width=True,
                hide_index=True
            )


        st.subheader(
            "Individual Monitoring Calculator"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            calc_age = st.number_input(
                "Age",
                min_value=10,
                max_value=60,
                value=28
            )

            calc_parity = st.number_input(
                "Parity",
                min_value=0,
                max_value=15,
                value=1
            )

            calc_gestation = st.number_input(
                "Gestation weeks",
                min_value=0,
                max_value=60,
                value=24
            )

        with c2:

            calc_visits = st.number_input(
                "Recorded ANC visits",
                min_value=0,
                max_value=50,
                value=2
            )

            calc_weight = st.number_input(
                "Weight (kg)",
                min_value=20.0,
                max_value=250.0,
                value=65.0
            )

        with c3:

            calc_sbp = st.number_input(
                "Systolic BP",
                min_value=50,
                max_value=250,
                value=120
            )

            calc_dbp = st.number_input(
                "Diastolic BP",
                min_value=30,
                max_value=180,
                value=80
            )


        if st.button(
            "Calculate monitoring flags"
        ):

            test_row = pd.Series({
                "Age": calc_age,
                "Parity_clean": calc_parity,
                "Gestation Weeks": calc_gestation,
                "Visit Count": calc_visits,
                "Weight (kg)": calc_weight,
                "Systolic BP": calc_sbp,
                "Diastolic BP": calc_dbp
            })

            result = calculate_monitoring(
                test_row
            )

            st.metric(
                "Monitoring score",
                int(result["Monitoring_Score"])
            )

            st.write(
                "**Category:**",
                result["Monitoring_Category"]
            )

            if result["Monitoring_Flags"]:

                st.write(
                    "**Flags:**",
                    result["Monitoring_Flags"]
                )

            else:

                st.write(
                    "No monitoring flags identified."
                )


# ============================================================
# TAB 3 — DATA
# ============================================================

with tabs[2]:

    st.header("📋 Data")

    st.caption(
        "The table below shows the records available in the dashboard "
        "dataset after the selected year filters."
    )

    st.write(
        f"Showing **{len(filtered_df):,}** records and "
        f"**{len(filtered_df.columns):,}** columns."
    )

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TAB 4 — DATA QUALITY
# ============================================================

with tabs[3]:

    st.header("🔎 Data Quality")

    st.markdown(
        "Data-quality checks help identify missing, duplicate, or "
        "potentially implausible values before interpreting trends."
    )


    dq1, dq2, dq3 = st.columns(3)

    missing_cells = int(
        filtered_df.isna().sum().sum()
    )

    duplicate_rows = int(
        filtered_df.duplicated().sum()
    )

    dq1.metric(
        "Missing cells",
        f"{missing_cells:,}"
    )

    dq2.metric(
        "Duplicate rows",
        f"{duplicate_rows:,}"
    )

    dq3.metric(
        "Columns",
        f"{len(filtered_df.columns):,}"
    )


    st.subheader(
        "Missingness by Variable"
    )

    missing_table = (
        filtered_df.isna()
        .sum()
        .reset_index()
    )

    missing_table.columns = [
        "Variable",
        "Missing"
    ]

    missing_table["Percent Missing"] = (
        missing_table["Missing"]
        / max(len(filtered_df), 1)
        * 100
    ).round(1)

    st.dataframe(
        missing_table.sort_values(
            "Percent Missing",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )


    st.subheader(
        "Potential Validity Checks"
    )

    validity_rows = []


    if "Gestation Weeks" in filtered_df.columns:

        invalid_gestation = int(
            (
                (filtered_df["Gestation Weeks"] < 0)
                |
                (filtered_df["Gestation Weeks"] > 42)
            ).sum()
        )

        validity_rows.append({
            "Variable": "Gestation Weeks",
            "Potentially implausible records":
                invalid_gestation,
            "Rule": "<0 or >42 weeks"
        })


    if "Systolic BP" in filtered_df.columns:

        invalid_sbp = int(
            (
                (filtered_df["Systolic BP"] < 50)
                |
                (filtered_df["Systolic BP"] > 250)
            ).sum()
        )

        validity_rows.append({
            "Variable": "Systolic BP",
            "Potentially implausible records":
                invalid_sbp,
            "Rule": "<50 or >250 mmHg"
        })


    if "Diastolic BP" in filtered_df.columns:

        invalid_dbp = int(
            (
                (filtered_df["Diastolic BP"] < 30)
                |
                (filtered_df["Diastolic BP"] > 180)
            ).sum()
        )

        validity_rows.append({
            "Variable": "Diastolic BP",
            "Potentially implausible records":
                invalid_dbp,
            "Rule": "<30 or >180 mmHg"
        })


    if "Age" in filtered_df.columns:

        invalid_age = int(
            (
                (filtered_df["Age"] < 10)
                |
                (filtered_df["Age"] > 60)
            ).sum()
        )

        validity_rows.append({
            "Variable": "Age",
            "Potentially implausible records":
                invalid_age,
            "Rule": "<10 or >60 years"
        })


    if validity_rows:

        st.dataframe(
            pd.DataFrame(validity_rows),
            use_container_width=True,
            hide_index=True
        )


    st.warning(
        "Potentially implausible values are flagged for review; they are "
        "not automatically deleted because unusual values may reflect "
        "legitimate clinical records or data-entry issues."
    )


# ============================================================
# TAB 5 — AGE REGRESSION
# ============================================================

with tabs[4]:

    st.header("📈 Age Regression")

    st.markdown(
        "This exploratory regression examines how selected routinely "
        "recorded ANC characteristics relate statistically to recorded "
        "maternal age. Age is used here as the dependent variable because "
        "the ANC register does not contain a reliably linked adverse "
        "maternal or neonatal outcome."
    )


    required = [
        "Age",
        "Parity_clean",
        "Gravidity",
        "Gestation Weeks",
        "Visit Count",
        "Weight (kg)",
        "Systolic BP",
        "Diastolic BP",
        "Year"
    ]

    missing_required = [
        c for c in required
        if c not in df.columns
    ]


    if missing_required:

        st.warning(
            "The regression section cannot run because these required "
            "variables are missing from the dashboard dataset: "
            + ", ".join(missing_required)
        )

        if (
            "Parity_clean" not in df.columns
            and "Parity" not in df.columns
        ):

            st.info(
                "The current CSV does not contain either 'Parity_clean' "
                "or 'Parity'. The other dashboard tabs can still work."
            )


    else:

        reg_df = df[required].copy()

        reg_df["Year_2026"] = (
            pd.to_numeric(
                reg_df["Year"],
                errors="coerce"
            ) == 2026
        ).astype(int)

        reg_df = reg_df.dropna()


        if len(reg_df) < 30:

            st.warning(
                "There are fewer than 30 complete records available "
                "for the exploratory regression."
            )


        else:

            train_df, test_df = train_test_split(
                reg_df,
                test_size=0.20,
                random_state=42
            )


            formula = """
                Age ~ Parity_clean
                + Gravidity
                + Q('Gestation Weeks')
                + Q('Visit Count')
                + Q('Weight (kg)')
                + Q('Systolic BP')
                + Q('Diastolic BP')
                + Year_2026
            """


            try:

                dashboard_model = smf.ols(
                    formula,
                    data=train_df
                ).fit()


                predictions = dashboard_model.predict(
                    test_df
                )

                actual = test_df["Age"]


                mae = mean_absolute_error(
                    actual,
                    predictions
                )

                rmse = np.sqrt(
                    mean_squared_error(
                        actual,
                        predictions
                    )
                )

                r2 = r2_score(
                    actual,
                    predictions
                )


                within_2 = (
                    np.abs(
                        actual - predictions
                    ) <= 2
                ).mean() * 100


                within_5 = (
                    np.abs(
                        actual - predictions
                    ) <= 5
                ).mean() * 100


                r1, r2c, r3, r4 = st.columns(4)


                r1.metric(
                    "MAE",
                    f"{mae:.2f} years"
                )

                r2c.metric(
                    "RMSE",
                    f"{rmse:.2f} years"
                )

                r3.metric(
                    "R²",
                    f"{r2:.3f}"
                )

                r4.metric(
                    "Within ±5 years",
                    f"{within_5:.1f}%"
                )


                st.caption(
                    f"Test-set results from {len(test_df):,} records. "
                    f"Predictions within ±2 years: {within_2:.1f}%."
                )


                st.subheader(
                    "Actual vs Predicted Age"
                )


                prediction_df = pd.DataFrame({
                    "Actual Age": actual,
                    "Predicted Age": predictions
                })


                fig = px.scatter(
                    prediction_df,
                    x="Actual Age",
                    y="Predicted Age",
                    title="Actual vs Predicted Age"
                )


                min_age = min(
                    prediction_df["Actual Age"].min(),
                    prediction_df["Predicted Age"].min()
                )

                max_age = max(
                    prediction_df["Actual Age"].max(),
                    prediction_df["Predicted Age"].max()
                )


                fig.add_shape(
                    type="line",
                    x0=min_age,
                    y0=min_age,
                    x1=max_age,
                    y1=max_age
                )


                st.plotly_chart(
                    fig,
                    use_container_width=True
                )


                st.subheader(
                    "Residuals"
                )


                residual_df = pd.DataFrame({
                    "Predicted Age": predictions,
                    "Residual": actual - predictions
                })


                fig = px.scatter(
                    residual_df,
                    x="Predicted Age",
                    y="Residual",
                    title="Residuals vs Predicted Age"
                )

                fig.add_hline(y=0)

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )


                st.subheader(
                    "Regression Coefficients"
                )


                coefficients = pd.DataFrame({
                    "Variable":
                        dashboard_model.params.index,

                    "Coefficient":
                        dashboard_model.params.values,

                    "P-value":
                        dashboard_model.pvalues.values
                })


                st.dataframe(
                    coefficients,
                    use_container_width=True,
                    hide_index=True
                )


                with st.expander(
                    "Model summary"
                ):

                    st.text(
                        dashboard_model.summary()
                    )


                st.warning(
                    "Interpretation: the regression is exploratory. "
                    "The R² value indicates the proportion of variation "
                    "in recorded age explained by the included variables "
                    "in this sample; it does not establish clinical risk "
                    "or causation. A validated adverse maternal or neonatal "
                    "outcome was not available for reliable linkage in "
                    "the ANC register."
                )


            except Exception as e:

                st.error(
                    "The regression could not be fitted with the available "
                    f"data. Technical detail: {e}"
                )
