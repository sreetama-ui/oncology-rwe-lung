import streamlit as st
import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

# Title
st.title("Lung Cancer Survival Dashboard – Real-World Evidence")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/lung_survival.csv")
    return df

df = load_data()

# Sidebar Filters
st.sidebar.header("Filters")
stages = df["stage"].unique().tolist()
selected_stage = st.sidebar.multiselect("Select Stage(s)", stages, default=stages)

filtered_data = df[df["stage"].isin(selected_stage)]

# KM Survival Plot
st.subheader("Kaplan-Meier Survival Curve")

if len(filtered_data) > 0:
    kmf = KaplanMeierFitter()

    fig, ax = plt.subplots()
    for stage in selected_stage:
        stage_data = filtered_data[filtered_data["stage"] == stage]
        kmf.fit(stage_data["time"], event_observed=stage_data["status"], label=f"Stage {stage}")
        kmf.plot_survival_function(ax=ax)

    plt.xlabel("Time (days)")
    plt.ylabel("Survival Probability")
    plt.title("Overall Survival by Cancer Stage")

    st.pyplot(fig)
else:
    st.warning("Please select at least one stage.")

# Data table toggle
if st.checkbox("Show underlying dataset"):
    st.write(filtered_data)
