# Wananchi Hospital ANC Monitoring Dashboard

Exploratory Streamlit dashboard for routinely recorded ANC data from Wananchi Hospital.

## Dashboard Sections

- ANC Dashboard
- Exploratory ANC Monitoring
- Cleaned Data
- Data Quality
- Exploratory Age Regression

## Important Limitations

The monitoring framework is exploratory and is not a validated clinical diagnostic or adverse-outcome prediction tool.

The regression model uses recorded maternal age as the dependent variable because the available ANC register does not contain a reliably linked validated maternal or neonatal adverse-outcome variable. Regression results are therefore exploratory statistical analysis.

## Data Privacy

The original patient-level ANC workbook should not be uploaded to a public GitHub repository.

The deployment dataset should remain sanitized and should not contain direct identifiers or sensitive variables excluded from the analytical dataset.

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
