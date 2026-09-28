
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Wananchi ANC Monitoring",
    page_icon="🏥",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("wananchi_anc_data.csv")


@st.cache_resource
def load_model():
    try:
        return joblib.load("wananchi_age_regression_model.pkl")
    except Exception:
        return None


df = load_data()
age_model = load_model()


# ============================================================
# TITLE
# ============================================================

st.title("🏥 Wananchi Hospital ANC Monitoring Dashboard")

st.markdown(
    """
    This dashboard provides an exploratory summary of routinely recorded
    antenatal care (ANC) data from Wananchi Hospital for January–August
    2021 and January–August 2026.
    """
)

st.info(
    "⚠️ The ANC monitoring framework is exploratory and is not a validated "
    "clinical diagnostic or adverse-outcome prediction tool."
)


# ============================================================
# SIDEBAR FILTER
# ============================================================

st.sidebar.header("Dashboard Filters")

years = sorted(df["Year"].dropna().unique())

selected_years = st.sidebar.multiselect(
    "Select year",
    years,
    default=years
)

filtered_df = df[df["Year"].isin(selected_years)].copy()


# ============================================================
# KPI CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "ANC Records",
        len(filtered_df)
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
    high_bp = (
        (filtered_df["Systolic BP"] >= 140) |
        (filtered_df["Diastolic BP"] >= 90)
    ).sum()

    st.metric(
        "Elevated BP Records",
        int(high_bp)
    )


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 ANC Dashboard",
    "🚩 ANC Monitoring",
    "📋 Data",
    "🔎 Data Quality",
    "📈 Age Regression"
])


# ============================================================
# TAB 1 — ANC DASHBOARD
# ============================================================

with tab1:

    st.header("ANC Dashboard")

    # --------------------------------------------------------
    # AGE
    # --------------------------------------------------------

    st.subheader("Age Distribution")

    fig_age = px.histogram(
        filtered_df,
        x="Age",
        color="Year",
        nbins=20,
        barmode="overlay",
        title="Age Distribution by Year"
    )

    st.plotly_chart(
        fig_age,
        use_container_width=True
    )


    # --------------------------------------------------------
    # GESTATIONAL AGE
    # --------------------------------------------------------

    st.subheader("Gestational Age")

    fig_gestation = px.histogram(
        filtered_df,
        x="Gestation Weeks",
        color="Year",
        nbins=20,
        barmode="overlay",
        title="Gestational Age Distribution"
    )

    st.plotly_chart(
        fig_gestation,
        use_container_width=True
    )


    # --------------------------------------------------------
    # VISITS AND WEIGHT
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("ANC Visit Count")

        visit_counts = (
            filtered_df
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
            title="ANC Visits by Year"
        )

        st.plotly_chart(
            fig_visits,
            use_container_width=True
        )


    with col2:

        st.subheader("Weight Distribution")

        fig_weight = px.box(
            filtered_df,
            x="Year",
            y="Weight (kg)",
            points="outliers",
            title="Weight Distribution by Year"
        )

        st.plotly_chart(
            fig_weight,
            use_container_width=True
        )


    # --------------------------------------------------------
    # BLOOD PRESSURE
    # --------------------------------------------------------

    st.subheader("Blood Pressure")

    bp_df = filtered_df[
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
        title="Blood Pressure Distribution by Year"
    )

    st.plotly_chart(
        fig_bp,
        use_container_width=True
    )


# ============================================================
# TAB 2 — ANC MONITORING
# ============================================================

with tab2:

    st.header("🚩 Exploratory ANC Monitoring Framework")

    st.markdown(
        """
        The framework assigns monitoring points to selected characteristics
        recorded in the ANC register. It is intended for exploratory
        monitoring only and does not replace clinical assessment.
        """
    )

    # --------------------------------------------------------
    # CATEGORY DISTRIBUTION
    # --------------------------------------------------------

    category_counts = (
        filtered_df["Monitoring_Category"]
        .value_counts()
        .reset_index()
    )

    category_counts.columns = [
        "Monitoring Category",
        "Records"
    ]

    fig_monitoring = px.bar(
        category_counts,
        x="Monitoring Category",
        y="Records",
        title="Monitoring Categories"
    )

    st.plotly_chart(
        fig_monitoring,
        use_container_width=True
    )


    # --------------------------------------------------------
    # MONITORING BY YEAR
    # --------------------------------------------------------

    st.subheader("Monitoring Summary by Year")

    monitoring_summary = (
        filtered_df[
            ["Year", "Monitoring_Category"]
        ]
        .groupby(
            ["Year", "Monitoring_Category"]
        )
        .size()
        .reset_index(name="Records")
    )

    st.dataframe(
        monitoring_summary,
        use_container_width=True
    )


    # --------------------------------------------------------
    # INDIVIDUAL CALCULATOR
    # --------------------------------------------------------

    st.subheader("Individual Monitoring Calculator")

    st.write(
        "Enter example ANC characteristics to demonstrate how the "
        "exploratory monitoring framework operates."
    )

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=10,
            max_value=60,
            value=25
        )

        parity = st.number_input(
            "Parity",
            min_value=0,
            max_value=15,
            value=1
        )

        gestation = st.number_input(
            "Gestation Weeks",
            min_value=0,
            max_value=60,
            value=24
        )

        visits = st.number_input(
            "ANC Visit Count",
            min_value=0,
            max_value=50,
            value=2
        )

    with col2:

        weight = st.number_input(
            "Weight (kg)",
            min_value=20.0,
            max_value=250.0,
            value=65.0
        )

        systolic = st.number_input(
            "Systolic BP",
            min_value=50,
            max_value=250,
            value=120
        )

        diastolic = st.number_input(
            "Diastolic BP",
            min_value=30,
            max_value=150,
            value=80
        )

    if st.button("Assess Monitoring Level"):

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

        st.metric(
            "Monitoring Score",
            score
        )

        st.write(
            "**Monitoring Category:**",
            category
        )

        if flags:

            st.write("**Monitoring Flags:**")

            for flag in flags:
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
# TAB 3 — DATA
# ============================================================

with tab3:

    st.header("📋 Cleaned ANC Data")

    st.dataframe(
        filtered_df,
        use_container_width=True
    )

    st.success(
        "✅ Dashboard data loaded successfully."
    )

    st.caption(
        "The displayed dataset excludes Serial Number, HIV Status, "
        "TB Screening, Infant Prophylaxis and MUAC from the analytical dataset."
    )


# ============================================================
# TAB 4 — DATA QUALITY
# ============================================================

with tab4:

    st.header("🔎 Data Quality Assessment")

    st.markdown(
        """
        This section describes completeness, duplication and selected
        validity checks in the analytical dataset. Flags are intended for
        data-quality investigation and do not automatically imply that
        individual records are incorrect.
        """
    )


    # --------------------------------------------------------
    # BASIC DATA QUALITY KPIs
    # --------------------------------------------------------

    total_records = len(filtered_df)

    duplicate_records = filtered_df.duplicated().sum()

    total_missing = filtered_df.isna().sum().sum()

    total_cells = filtered_df.shape[0] * filtered_df.shape[1]

    completeness = (
        100 * (1 - total_missing / total_cells)
        if total_cells > 0
        else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Records",
            total_records
        )

    with col2:
        st.metric(
            "Duplicate Rows",
            int(duplicate_records)
        )

    with col3:
        st.metric(
            "Missing Cells",
            int(total_missing)
        )

    with col4:
        st.metric(
            "Overall Completeness",
            f"{completeness:.1f}%"
        )


    # --------------------------------------------------------
    # MISSINGNESS
    # --------------------------------------------------------

    st.subheader("Missing Values by Variable")

    missing_table = pd.DataFrame({
        "Variable": filtered_df.columns,
        "Missing Values": filtered_df.isna().sum().values
    })

    missing_table["Missing (%)"] = (
        missing_table["Missing Values"] /
        len(filtered_df) * 100
        if len(filtered_df) > 0
        else 0
    )

    missing_table = missing_table.sort_values(
        "Missing (%)",
        ascending=False
    )

    fig_missing = px.bar(
        missing_table,
        x="Variable",
        y="Missing (%)",
        title="Percentage of Missing Values"
    )

    fig_missing.update_layout(
        xaxis_tickangle=-45
    )

    st.plotly_chart(
        fig_missing,
        use_container_width=True
    )

    st.dataframe(
        missing_table,
        use_container_width=True
    )


    # --------------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------------

    st.subheader("Duplicate Records")

    if duplicate_records > 0:

        st.warning(
            f"{duplicate_records} duplicate row(s) detected."
        )

    else:

        st.success(
            "No exact duplicate rows detected in the selected data."
        )


    # --------------------------------------------------------
    # IMPLAUSIBLE VALUES
    # --------------------------------------------------------

    st.subheader("Selected Validity Checks")

    quality_checks = []

    if "Age" in filtered_df.columns:

        quality_checks.append({
            "Check": "Age below 15 or above 49",
            "Records Flagged": int(
                ((filtered_df["Age"] < 15) |
                 (filtered_df["Age"] > 49)).sum()
            )
        })

    if "Gestation Weeks" in filtered_df.columns:

        quality_checks.append({
            "Check": "Gestation above 42 weeks",
            "Records Flagged": int(
                (filtered_df["Gestation Weeks"] > 42).sum()
            )
        })

    if "Weight (kg)" in filtered_df.columns:

        quality_checks.append({
            "Check": "Weight below 30 kg or above 150 kg",
            "Records Flagged": int(
                ((filtered_df["Weight (kg)"] < 30) |
                 (filtered_df["Weight (kg)"] > 150)).sum()
            )
        })

    if "Systolic BP" in filtered_df.columns:

        quality_checks.append({
            "Check": "Systolic BP below 70 or above 200",
            "Records Flagged": int(
                ((filtered_df["Systolic BP"] < 70) |
                 (filtered_df["Systolic BP"] > 200)).sum()
            )
        })

    if "Diastolic BP" in filtered_df.columns:

        quality_checks.append({
            "Check": "Diastolic BP below 40 or above 120",
            "Records Flagged": int(
                ((filtered_df["Diastolic BP"] < 40) |
                 (filtered_df["Diastolic BP"] > 120)).sum()
            )
        })

    quality_df = pd.DataFrame(
        quality_checks
    )

    st.dataframe(
        quality_df,
        use_container_width=True
    )

    st.info(
        "Validity thresholds shown here are screening thresholds for "
        "data-quality review. They should not be interpreted as clinical "
        "diagnostic thresholds."
    )


# ============================================================
# TAB 5 — AGE REGRESSION
# ============================================================

with tab5:

    st.header("📈 Exploratory Multiple Linear Regression")

    st.markdown(
        """
        This analysis models recorded maternal age using selected ANC
        characteristics. It is an exploratory statistical analysis and
        should not be interpreted as a clinical risk-prediction model.
        """
    )


    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------

    if age_model is None:

        st.error(
            "The regression model file was not found. "
            "Make sure 'wananchi_age_regression_model.pkl' "
            "is in the same folder as app.py."
        )

    else:

        # ----------------------------------------------------
        # CREATE REGRESSION DATA
        # ----------------------------------------------------

        regression_variables = [
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

        available_variables = [
            col for col in regression_variables
            if col in df.columns
        ]

        regression_df = (
            df[available_variables]
            .dropna()
            .copy()
        )

        regression_df["Year_2026"] = (
            regression_df["Year"] == 2026
        ).astype(int)


        # ----------------------------------------------------
        # DISPLAY MODEL METRICS
        # ----------------------------------------------------

        # Use stored model if available
        try:

            formula_test = """
            Age ~ Parity_clean
                 + Gravidity
                 + Q('Gestation Weeks')
                 + Q('Visit Count')
                 + Q('Weight (kg)')
                 + Q('Systolic BP')
                 + Q('Diastolic BP')
                 + Year_2026
            """

            # Refit only to obtain consistent dashboard evaluation
            # on the complete available analytical dataset.
            dashboard_model = smf.ols(
                formula_test,
                data=regression_df
            ).fit()

            predictions = dashboard_model.predict(
                regression_df
            )

            actual = regression_df["Age"]

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
                np.abs(actual - predictions) <= 2
            ).mean() * 100

            within_3 = (
                np.abs(actual - predictions) <= 3
            ).mean() * 100

            within_5 = (
                np.abs(actual - predictions) <= 5
            ).mean() * 100

            within_10 = (
                np.abs(actual - predictions) <= 10
            ).mean() * 100

            mape = (
                np.mean(
                    np.abs(
                        (actual - predictions) /
                        actual
                    )
                ) * 100
            )


            # ------------------------------------------------
            # METRIC CARDS
            # ------------------------------------------------

            st.subheader("Regression Performance")

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "MAE",
                    f"{mae:.2f} years"
                )

            with col2:

                st.metric(
                    "RMSE",
                    f"{rmse:.2f} years"
                )

            with col3:

                st.metric(
                    "R²",
                    f"{r2:.3f}"
                )


            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "Within ±2 years",
                    f"{within_2:.1f}%"
                )

            with col2:

                st.metric(
                    "Within ±3 years",
                    f"{within_3:.1f}%"
                )

            with col3:

                st.metric(
                    "Within ±5 years",
                    f"{within_5:.1f}%"
                )

            with col4:

                st.metric(
                    "Within ±10 years",
                    f"{within_10:.1f}%"
                )


            st.metric(
                "MAPE",
                f"{mape:.2f}%"
            )


            # ------------------------------------------------
            # INTERPRETATION
            # ------------------------------------------------

            st.info(
                f"""
                **Interpretation:** The exploratory model has an R² of
                {r2:.3f}, meaning that approximately {r2 * 100:.1f}% of
                the variation in recorded age is explained by the included
                predictors in this fitted dataset. The MAE is approximately
                {mae:.2f} years.

                The percentages reported above describe the proportion of
                observations whose predicted age falls within the stated
                error range. They should not be interpreted as clinical
                prediction accuracy.
                """
            )


            # ------------------------------------------------
            # ACTUAL VS PREDICTED
            # ------------------------------------------------

            st.subheader("Actual vs Predicted Age")

            prediction_df = pd.DataFrame({
                "Actual Age": actual,
                "Predicted Age": predictions
            })

            fig_prediction = px.scatter(
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

            fig_prediction.add_shape(
                type="line",
                x0=min_age,
                y0=min_age,
                x1=max_age,
                y1=max_age
            )

            st.plotly_chart(
                fig_prediction,
                use_container_width=True
            )


            # ------------------------------------------------
            # RESIDUALS
            # ------------------------------------------------

            st.subheader("Residual Analysis")

            residual_df = pd.DataFrame({
                "Fitted Age": predictions,
                "Residual": actual - predictions
            })

            fig_residual = px.scatter(
                residual_df,
                x="Fitted Age",
                y="Residual",
                title="Residuals vs Fitted Values"
            )

            fig_residual.add_hline(
                y=0
            )

            st.plotly_chart(
                fig_residual,
                use_container_width=True
            )


            # ------------------------------------------------
            # Q-Q PLOT
            # ------------------------------------------------

            st.subheader("Normal Q-Q Plot")

            qq = sm.qqplot(
                dashboard_model.resid,
                line="45",
                fit=True
            )

            st.pyplot(
                qq.figure,
                clear_figure=True
            )


            # ------------------------------------------------
            # MODEL COEFFICIENTS
            # ------------------------------------------------

            st.subheader("Regression Coefficients")

            coefficient_table = pd.DataFrame({
                "Variable": dashboard_model.params.index,
                "Coefficient": dashboard_model.params.values,
                "P-value": dashboard_model.pvalues.values
            })

            coefficient_table["Coefficient"] = (
                coefficient_table["Coefficient"]
                .round(4)
            )

            coefficient_table["P-value"] = (
                coefficient_table["P-value"]
                .round(4)
            )

            st.dataframe(
                coefficient_table,
                use_container_width=True
            )


            # ------------------------------------------------
            # MODEL SUMMARY
            # ------------------------------------------------

            with st.expander("Show full regression model summary"):

                st.text(
                    dashboard_model.summary().as_text()
                )


            st.warning(
                """
                This regression uses Age as the dependent variable because
                the available ANC register does not contain a reliably linked
                validated maternal or neonatal adverse-outcome variable.
                Therefore, the model should be presented as exploratory
                statistical analysis rather than clinical risk prediction.
                """
            )


        except Exception as e:

            st.error(
                f"Regression analysis could not be displayed: {e}"
            )



# FOOTER


st.markdown("---")

st.caption(
    "Wananchi Hospital ANC Dashboard | Exploratory academic/research tool | "
    "Not a clinical diagnostic system"
)
