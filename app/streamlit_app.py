# app/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Lung Cancer RWE Dashboard", layout="wide")

# --- Helpers ---
@st.cache_data
def load_data():
    df = pd.read_csv("data/lung_survival.csv")
    # Ensure correct dtypes
    if "time" in df.columns:
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
    if "status" in df.columns:
        df["status"] = pd.to_numeric(df["status"], errors="coerce")
    # Simple cleaning
    df = df.dropna(subset=["time", "status"])
    return df

def compute_median_survival(df, group_col="treatment"):
    medians = {}
    kmf = KaplanMeierFitter()
    for g in df[group_col].unique():
        sub = df[df[group_col] == g]
        try:
            kmf.fit(sub["time"], event_observed=sub["status"])
            medians[g] = kmf.median_survival_time_
        except Exception:
            medians[g] = np.nan
    return medians

def run_cox(filtered_df, time_col="time", event_col="status"):
    # Require enough events and rows
    if filtered_df.shape[0] < 20:
        raise ValueError("Not enough observations to fit a stable Cox model (need >=20 rows).")
    if filtered_df[event_col].sum() < 5:
        raise ValueError("Not enough events to fit Cox model (need >=5 events).")

    df = filtered_df.copy()
    # Select covariates: treatment, age, sex, smoking
    covs = []
    if "treatment" in df.columns:
        covs.append("treatment")
    if "age" in df.columns:
        covs.append("age")
    if "sex" in df.columns:
        covs.append("sex")
    if "smoking" in df.columns:
        covs.append("smoking")

    if len(covs) == 0:
        raise ValueError("No covariates available for Cox model.")

    # One-hot encode categorical vars
    df_enc = pd.get_dummies(df[[time_col, event_col] + covs], drop_first=True)
    df_enc = df_enc.rename(columns={time_col: "duration", event_col: "event"})
    # Fit Cox
    cph = CoxPHFitter()
    cph.fit(df_enc, duration_col="duration", event_col="event", step_size=0.1)
    summary = cph.summary.reset_index().rename(columns={"index": "covariate"})
    # Add HR and formatted CI
    summary["HR"] = np.exp(summary["coef"])
    summary["CI_lower"] = np.exp(summary["coef lower 95%"])
    summary["CI_upper"] = np.exp(summary["coef upper 95%"])
    summary = summary[["covariate", "coef", "HR", "CI_lower", "CI_upper", "p"]]
    return cph, summary

# --- Main ---
df = load_data()

st.title("🫁 Lung Cancer Survival Dashboard — Real-World Evidence")
st.markdown("Interactive KM plots, Cox PH model, and downloadable results. Built for RWE / Biostatistics portfolio.")

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    stages = sorted(df["stage"].unique().astype(str).tolist())
    selected_stage = st.multiselect("Select Stage(s)", stages, default=stages)
    # Treatment filter optional
    if "treatment" in df.columns:
        trts = sorted(df["treatment"].unique().astype(str).tolist())
        selected_treatment = st.multiselect("Select Treatment(s) (optional)", trts, default=trts)
    else:
        selected_treatment = None
    st.write("---")
    st.caption("Tip: Use stage filter to subset populations for subgroup analyses.")

# Apply filters
filtered = df[df["stage"].astype(str).isin(selected_stage)]
if selected_treatment is not None:
    filtered = filtered[filtered["treatment"].astype(str).isin(selected_treatment)]

# Layout - three tabs
tab1, tab2, tab3 = st.tabs(["KM Plot", "Cox Model", "Data / Download"])

# --- Tab 1: KM Plot ---
with tab1:
    st.subheader("Kaplan–Meier Survival Curve")
    if filtered.empty:
        st.warning("No data for selected filters. Please choose different Stage/Treatment.")
    else:
        # KM by treatment (if present), otherwise overall by stage
        group_col = "treatment" if "treatment" in filtered.columns else "stage"
        kmf = KaplanMeierFitter()

        fig, ax = plt.subplots(figsize=(8,5))
        for grp in filtered[group_col].unique():
            grp_df = filtered[filtered[group_col] == grp]
            try:
                kmf.fit(grp_df["time"], event_observed=grp_df["status"], label=str(grp))
                kmf.plot_survival_function(ax=ax)
            except Exception as e:
                st.warning(f"Could not fit KM for group {grp}: {e}")

        ax.set_xlabel("Time (months)")
        ax.set_ylabel("Survival Probability")
        ax.set_title(f"Kaplan–Meier by {group_col.capitalize()}")
        st.pyplot(fig, clear_figure=True)

        # Median survival metrics
        medians = compute_median_survival(filtered, group_col=group_col)
        cols = st.columns(len(medians))
        for i,(k,v) in enumerate(medians.items()):
            val = f"{v:.1f} mo" if not pd.isna(v) else "NA"
            cols[i].metric(label=f"Median survival: {k}", value=val)

        # Interpretation text automatically generated
        try:
            groups = list(medians.keys())
            med_vals = [medians[g] for g in groups]
            best_idx = np.nanargmax(med_vals)
            worst_idx = np.nanargmin(med_vals)
            st.markdown(f"*Quick insight:* Among the shown groups, {groups[best_idx]} has the highest median survival ({med_vals[best_idx]:.1f} months) and {groups[worst_idx]} the lowest ({med_vals[worst_idx]:.1f} months).")
        except Exception:
            pass

# --- Tab 2: Cox Model ---
with tab2:
    st.subheader("Cox Proportional Hazards Model (multivariable)")

    show_cox = st.button("Run Cox PH model on current filters")
    if show_cox:
        try:
            cph, summary = run_cox(filtered)
            st.success("Cox model fitted successfully.")
            st.dataframe(summary.style.format({"coef":"{:.3f}", "HR":"{:.3f}", "CI_lower":"{:.3f}", "CI_upper":"{:.3f}", "p":"{:.4f}"}))

            # Download Cox results
            csv = summary.to_csv(index=False).encode("utf-8")
            st.download_button("Download Cox results (CSV)", data=csv, file_name="cox_summary.csv", mime="text/csv")

            # Simple proportional hazards check: display concordance
            try:
                concord = cph.concordance_index_
                st.write(f"Model concordance (C-index): *{concord:.3f}*")
            except Exception:
                pass

            # Provide short interpretative sentences for top covariates
            st.markdown("### Interpretation (auto-generated)")
            for _, row in summary.iterrows():
                cov = row["covariate"]
                hr = row["HR"]
                pval = row["p"]
                if pval < 0.05:
                    sig = "statistically significant"
                else:
                    sig = "not statistically significant"
                st.write(f"- *{cov}*: HR = {hr:.2f} (95% CI: {row['CI_lower']:.2f}–{row['CI_upper']:.2f}), p = {pval:.3f} → {sig}.")

        except Exception as e:
            st.error(f"Could not fit Cox model: {e}")

# --- Tab 3: Data / Download ---
with tab3:
    st.subheader("Filtered dataset")
    st.write(f"Rows: {filtered.shape[0]} | Events (deaths): {int(filtered['status'].sum()) if 'status' in filtered.columns else 'NA'}")

    st.dataframe(filtered.reset_index(drop=True))

    # Download filtered data
    towrite = io.StringIO()
    filtered.to_csv(towrite, index=False)
    st.download_button("Download filtered data (CSV)", data=towrite.getvalue().encode('utf-8'), file_name="lung_filtered.csv", mime="text/csv")

    # Repo / contact footer
    st.markdown("---")
    st.caption("Project: Oncology RWE — Lung Cancer Survival | Built for portfolio (RWE / Biostatistics).")
