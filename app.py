import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings

# Streamlit app
st.set_page_config(layout="wide", page_title="Material Production and CO2 Emissions Forecast")

st.title("Material Production and CO2 Emissions Forecast")
st.image("a.png", width=150, caption="Powered by EarthlyAI")

# Instructions for uploading files
st.sidebar.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #f8f9fa; color: #333;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    ### Instructions:
    1. **Upload the Materials DB Excel File first**. This file should contain sheets named **'Solids'** and **'Concentrates'**.
    2. **Upload the Production Data Excel File**. This file should contain a sheet named **'Production'**.
    """,
    unsafe_allow_html=True
)

# Upload Excel files
materials_db_file = st.sidebar.file_uploader("Upload Materials DB Excel File", type="xlsx")
data_file = st.sidebar.file_uploader("Upload Production Data Excel File", type="xlsx")

if materials_db_file and data_file:
    materials_db = pd.ExcelFile(materials_db_file)
    data = pd.ExcelFile(data_file)

    # Verify sheet names
    materials_sheet_names = materials_db.sheet_names

    if 'Solids' in materials_sheet_names and 'Concentrates' in materials_sheet_names:
        materials_solids = materials_db.parse('Solids')
        materials_concentrates = materials_db.parse('Concentrates')
    else:
        st.error("Expected sheets 'Solids' and 'Concentrates' not found in the Materials DB file. Please check the file.")
        st.stop()

    production_sheet_names = data.sheet_names

    if 'Production' in production_sheet_names:
        production_data = data.parse('Production')
    else:
        st.error("Expected sheet 'Production' not found in the Production Data file. Please check the file.")
        st.stop()

    # Analyze material diversity
    solids_materials = materials_solids.iloc[3:, [1]].dropna().rename(columns={"Unnamed: 1": "Material"})
    concentrates_materials = materials_concentrates.iloc[2:, [2]].dropna().rename(columns={"Unnamed: 2": "Material"})

    solids_count = solids_materials["Material"].nunique()
    concentrates_count = concentrates_materials["Material"].nunique()

    all_materials = pd.concat([solids_materials, concentrates_materials], ignore_index=True)
    all_materials_unique = all_materials["Material"].nunique()

    overlap_count = len(set(solids_materials["Material"]).intersection(set(concentrates_materials["Material"])))
    unique_solids = solids_count - overlap_count
    unique_concentrates = concentrates_count - overlap_count

    material_diversity_summary = {
        "Unique Materials (Solids)": solids_count,
        "Unique Materials (Concentrates)": concentrates_count,
        "Total Unique Materials (Combined)": all_materials_unique,
        "Overlapping Materials": overlap_count,
        "Unique to Solids": unique_solids,
        "Unique to Concentrates": unique_concentrates,
    }

    # Layout for displaying results
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Material Diversity Summary")
        st.write(material_diversity_summary)

        # Visualize material diversity
        categories = ['Unique to Solids', 'Unique to Concentrates', 'Overlapping']
        values = [unique_solids, unique_concentrates, overlap_count]

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(values, labels=categories, autopct='%1.1f%%', startangle=140, explode=(0.1, 0.1, 0), shadow=True)
        ax.set_title("Material Diversity Across Solids and Concentrates", fontsize=12)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(categories, values, color=['blue', 'green', 'gray'], alpha=0.7)
        ax.set_title("Material Diversity Breakdown")
        ax.set_ylabel("Values", fontsize=8, labelpad=10)
        ax.set_xlabel("Material Categories", fontsize=8, labelpad=10)
        ax.tick_params(axis='x', labelrotation=45, labelsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)

    # Prepare production and CO2 data
    production_data['Date'] = pd.to_datetime(production_data['Date'])
    production_data.set_index('Date', inplace=True)
    production = production_data['TPM'].replace([np.inf, -np.inf], np.nan).dropna()
    co2_emissions = production_data['MyBC growth CO2 Produced Tons (1,6 kg per ton)'].replace([np.inf, -np.inf], np.nan).dropna()
    plastics_co2_emissions = production_data['Plastic encineration(2.9kg CO2 per kg)'].replace([np.inf, -np.inf], np.nan).dropna()

    # Ensure all series have the same length
    min_length = min(len(production), len(co2_emissions), len(plastics_co2_emissions))
    production = production.iloc[:min_length]
    co2_emissions = co2_emissions.iloc[:min_length]
    plastics_co2_emissions = plastics_co2_emissions.iloc[:min_length]

    # Fit the data with Exponential Smoothing
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    try:
        production_model = ExponentialSmoothing(production, trend='add', seasonal=None).fit()
        co2_model = ExponentialSmoothing(co2_emissions, trend='add', seasonal=None).fit()
        plastics_co2_model = ExponentialSmoothing(plastics_co2_emissions, trend='add', seasonal=None).fit()
    except ValueError as ve:
        st.error("Error in fitting the data with Exponential Smoothing: " + str(ve))
        st.stop()
    except ConvergenceWarning:
        st.warning("Convergence issue detected during model fitting. Predictions might be suboptimal.")

    # Generate future predictions
    future_steps = 24
    future_dates = pd.date_range(start=production.index[-1], periods=future_steps + 1, freq='M')[1:]

    future_production = production_model.forecast(future_steps)
    future_co2 = co2_model.forecast(future_steps)
    future_plastics_co2 = plastics_co2_model.forecast(future_steps)

    # Create DataFrame for future predictions
    predictions = pd.DataFrame({
        'Date': future_dates,
        'Predicted Production (TPM)': future_production.values,
        'Predicted CO2 Emissions (Material)': future_co2.values,
        'Predicted CO2 Emissions (Plastics)': future_plastics_co2.values
    })

    predictions['CO2 per Ton (Material)'] = predictions['Predicted CO2 Emissions (Material)'] / predictions['Predicted Production (TPM)']
    predictions['CO2 per Ton (Plastics)'] = predictions['Predicted CO2 Emissions (Plastics)'] / predictions['Predicted Production (TPM)']
    predictions['Cumulative CO2 Reduction'] = (
        predictions['Predicted CO2 Emissions (Plastics)'] - predictions['Predicted CO2 Emissions (Material)']
    ).cumsum()

    industries_count = 100
    predictions['Total CO2 Reduction (100 Industries)'] = predictions['Cumulative CO2 Reduction'] * industries_count
    predictions['Plastic Avoided (100 Industries)'] = predictions['Predicted Production (TPM)'] * industries_count

    st.subheader("Future Predictions")
    st.write(predictions)

    # Plot forecasts
    st.subheader("Production and CO2 Emissions Forecast")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(production.index, production, label="Historical Production")
    ax.plot(production.index, co2_emissions, label="Historical CO2 Emissions (Material)")
    ax.plot(production.index, plastics_co2_emissions, label="Historical CO2 Emissions (Plastics)")
    ax.plot(predictions['Date'], predictions['Predicted Production (TPM)'], label="Predicted Production", linestyle='--')
    ax.plot(predictions['Date'], predictions['Predicted CO2 Emissions (Material)'], label="Predicted CO2 Emissions (Material)", linestyle='--')
    ax.plot(predictions['Date'], predictions['Predicted CO2 Emissions (Plastics)'], label="Predicted CO2 Emissions (Plastics)", linestyle='--')
    ax.set_xlabel("Date", fontsize=10, labelpad=10)
    ax.set_ylabel("Values", fontsize=10, labelpad=20)
    ax.set_title("Production and CO2 Emissions Forecast")
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    st.subheader("CO2 Emissions per Ton of Production")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(predictions['Date'], predictions['CO2 per Ton (Material)'], label="CO2 per Ton (Material)", linestyle='--')
    ax.plot(predictions['Date'], predictions['CO2 per Ton (Plastics)'], label="CO2 per Ton (Plastics)", linestyle='--')
    ax.set_xlabel("Date")
    ax.set_ylabel("CO2 Emissions per Ton", fontsize=10, labelpad=10)
    ax.set_title("CO2 Emissions per Ton of Production", fontsize=12)
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    st.subheader("CO2 Reduction Potential")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(predictions['Date'], predictions['Cumulative CO2 Reduction'], label="Cumulative CO2 Reduction Potential", linestyle='--', color='green')
    ax.plot(predictions['Date'], predictions['Total CO2 Reduction (100 Industries)'], label="Total CO2 Reduction (100 Industries)", linestyle='--', color='red')
    ax.set_xlabel("Date")
    ax.set_ylabel("CO2 Reduction (Tons)", fontsize=10, labelpad=10)
    ax.set_title("CO2 Reduction Potential by Scaling Material Production", fontsize=12)
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    st.subheader("Plastic Avoidance by Scaling Material Production")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(predictions['Date'], predictions['Plastic Avoided (100 Industries)'], label="Plastic Avoided (100 Industries)", linestyle='--', color='blue')
    ax.set_xlabel("Date")
    ax.set_ylabel("Plastic Avoided (Tons)", fontsize=10, labelpad=10)
    ax.set_title("Plastic Avoidance by Scaling Material Production", fontsize=12)
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    # Display recommendations
    st.subheader("Recommendations")
    recommendations = '''
    1. Invest in technologies to reduce CO2 emissions relative to production.
    2. Shift focus towards scaling material production while decoupling CO2 emissions using advanced engineering solutions.
    3. Highlight sustainability metrics to attract environmentally conscious investors.
    4. Demonstrate how CO2 emissions per ton for new materials are significantly lower compared to plastics (orders of magnitude difference).
    5. Show cumulative CO2 reduction potential as a key metric for scalability.
    6. Continue monitoring and refining the production-to-emission ratio to achieve long-term goals.
    '''
    st.markdown(recommendations)
else:
    st.warning("Please upload both the Materials DB and Production Data Excel files to proceed.")
