"""
Deeper survival analysis on the IBM Telco Customer Churn dataset.

This script extends the basic survival analysis by incorporating a richer
set of covariates and utilising the baseline cumulative hazard from
statsmodels' PHReg results to compute adjusted survival curves.  It
follows similar steps to ``telco_survival_analysis.py`` but adds
additional demographic and service variables to the Cox model.

Key features:

* Load and clean the data, converting relevant categorical variables into
  appropriate numeric or dummy-encoded formats.
* Summarise continuous and categorical variables to provide context for
  subsequent modelling.
* Estimate Kaplan–Meier survival curves overall and stratified by
  contract type.  Compute median survival times and 95 % confidence
  intervals where possible.
* Perform a log–rank test comparing survival across contract types.
* Fit a Cox proportional hazards model with multiple covariates:
  MonthlyCharges (continuous), Contract, InternetService, SeniorCitizen,
  PaymentMethod, PaperlessBilling, OnlineSecurity, and TechSupport.
  Categorical predictors are dummy encoded with a reference level.
* Extract hazard ratios, 95 % confidence intervals, and p‑values for
  interpretation.
* Assess proportional hazards using Spearman correlations between
  Schoenfeld residuals and event times.
* Generate adjusted survival curves by contract type using the
  baseline cumulative hazard from the fitted model.  Other covariates
  are held at baseline levels (median MonthlyCharges, reference
  categories for categorical predictors, and zero for SeniorCitizen).

The analysis is designed to be run as a standalone script.  Plots are
saved into a ``survival_plots_deep`` subdirectory relative to this
script.

Requirements: pandas, numpy, matplotlib, statsmodels, scipy
"""

import os
from typing import Dict, Tuple, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2, spearmanr
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import SurvfuncRight

# Path to the dataset (adjust as needed)
DATA_PATH = os.path.join(os.path.dirname(__file__), "Telco-Customer-Churn.csv")


def load_and_clean(path: str) -> pd.DataFrame:
    """Load the CSV file and perform extended cleaning.

    This function converts ``TotalCharges`` to numeric and drops rows
    with missing or invalid total charges.  It adds an ``event`` column
    indicating churn (1) or censoring (0) and a ``time`` column
    representing tenure in months.  Additional categorical variables
    (PaymentMethod, OnlineSecurity, TechSupport, PaperlessBilling) are
    retained for modelling.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        Cleaned dataset.
    """
    df = pd.read_csv(path)
    # Convert TotalCharges to numeric and drop invalid entries
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).copy()

    # Ensure MonthlyCharges is numeric
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")

    # Event indicator (1 for churned)
    df["event"] = (df["Churn"].str.strip().str.lower() == "yes").astype(int)
    # Time variable is tenure in months
    df["time"] = pd.to_numeric(df["tenure"], errors="coerce")

    # Drop rows with missing time or MonthlyCharges
    df = df.dropna(subset=["time", "MonthlyCharges"]).copy()

    # Convert SeniorCitizen to integer (already int in dataset)
    df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").astype(int)

    # Standardise Yes/No responses: remove whitespace and capitalise
    bool_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in bool_cols:
        df[col] = df[col].str.strip().str.title()

    # Clean categorical variables of interest
    cat_cols = ["Contract", "InternetService", "PaymentMethod",
                "OnlineSecurity", "TechSupport"]
    for col in cat_cols:
        df[col] = df[col].str.strip().str.title()
    # Recode "No Internet Service" into "No" for OnlineSecurity and TechSupport
    for col in ["OnlineSecurity", "TechSupport"]:
        df[col] = df[col].replace({"No Internet Service": "No"})

    return df


def summarise_data(df: pd.DataFrame) -> None:
    """Print descriptive statistics for numerical and categorical variables.

    Summarises ``time`` and ``MonthlyCharges`` using descriptive
    statistics and provides frequency counts for key categorical
    variables including churn, contract type, internet service,
    SeniorCitizen, PaperlessBilling, PaymentMethod, OnlineSecurity, and
    TechSupport.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned dataset.
    """
    # Numerical summary
    num_summary = df[["time", "MonthlyCharges"]].describe().T
    num_summary["median"] = df[["time", "MonthlyCharges"]].median()
    print("Summary of numerical variables (time and MonthlyCharges):")
    print(num_summary)
    print()
    # Categorical summaries
    cat_vars = ["Churn", "Contract", "InternetService", "SeniorCitizen",
                "PaperlessBilling", "PaymentMethod", "OnlineSecurity",
                "TechSupport"]
    for col in cat_vars:
        print(f"{col} counts:")
        print(df[col].value_counts())
        print()


def kaplan_meier(df: pd.DataFrame) -> Tuple[SurvfuncRight, Dict[str, SurvfuncRight]]:
    """Compute Kaplan–Meier survival functions for overall and by contract.

    Returns the overall survival function and a dictionary of survival
    functions keyed by contract type.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned dataset.

    Returns
    -------
    SurvfuncRight
        Overall Kaplan–Meier survival function.
    dict
        Mapping from contract names to Kaplan–Meier survival functions.
    """
    overall = SurvfuncRight(df["time"], df["event"])
    by_contract: Dict[str, SurvfuncRight] = {}
    for contract, group in df.groupby("Contract"):
        by_contract[str(contract)] = SurvfuncRight(group["time"], group["event"])
    return overall, by_contract


def logrank_test(times: np.ndarray, events: np.ndarray, groups: np.ndarray) -> Tuple[float, float]:
    """Perform a log–rank test for comparing survival across multiple groups.

    The implementation follows the approach described in ``telco_survival_analysis.py`` and
    supports two or more groups.  It returns the chi‑squared statistic and
    the associated p‑value.

    Parameters
    ----------
    times : ndarray
        Array of survival times.
    events : ndarray
        Array of event indicators (1 = event occurred, 0 = censored).
    groups : ndarray
        Array of group labels (categorical codes).

    Returns
    -------
    float
        Chi‑squared statistic.
    float
        p‑value for the test.
    """
    unique_times = np.sort(np.unique(times[events == 1]))
    group_vals = np.unique(groups)
    g = len(group_vals)
    observed = np.zeros(g)
    expected = np.zeros(g)
    var = np.zeros((g, g))
    for t in unique_times:
        at_risk = times >= t
        d_total = np.sum((times == t) & (events == 1))
        if d_total == 0:
            continue
        n_total = np.sum(at_risk)
        d_j = np.array([np.sum((times == t) & (events == 1) & (groups == gval)) for gval in group_vals])
        n_j = np.array([np.sum(at_risk & (groups == gval)) for gval in group_vals])
        observed += d_j
        expected += d_total * (n_j / n_total)
        if n_total > 1:
            frac = n_j / n_total
            v_factor = d_total * (n_total - d_total) / (n_total - 1)
            for i in range(g):
                for j in range(g):
                    if i == j:
                        var[i, j] += v_factor * frac[i] * (1 - frac[i])
                    else:
                        var[i, j] += -v_factor * frac[i] * frac[j]
    oe = observed - expected
    inv_var = np.linalg.pinv(var)
    chi_sq = float(oe.T @ inv_var @ oe)
    df_stat = g - 1
    p_val = float(1.0 - chi2.cdf(chi_sq, df_stat))
    return chi_sq, p_val


def fit_cox_model(df: pd.DataFrame, covariate_cols: List[str]) -> Tuple[Any, List[str]]:
    """Fit a Cox proportional hazards model using PHReg.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned dataset.
    covariate_cols : list of str
        List of covariate column names to include in the model.  These
        columns will be dummy encoded if categorical.

    Returns
    -------
    PHRegResults
        Fitted Cox model results.
    list of str
        Column names of the design matrix.
    """
    # Extract covariates
    X = df[covariate_cols]
    # Dummy encode categorical variables (strings or object dtype)
    X = pd.get_dummies(X, columns=[c for c in covariate_cols if df[c].dtype == object], drop_first=True)
    # Ensure numeric dtype
    X = X.astype(float)
    # Store column names for later use
    design_columns = list(X.columns)
    # Response variables
    time_var = df["time"].astype(float)
    event_var = df["event"].astype(int)
    # Fit Cox model (no intercept)
    model = PHReg(time_var, X, status=event_var, ties="breslow")
    result = model.fit()
    return result, design_columns


def schoenfeld_test(res: Any, df: pd.DataFrame, design_columns: List[str]) -> pd.DataFrame:
    """Assess proportional hazards using Schoenfeld residual correlations.

    Computes Spearman correlations between Schoenfeld residuals and event
    times.  A significant correlation indicates potential violation of
    the proportional hazards assumption.

    Parameters
    ----------
    res : PHRegResults
        Fitted Cox model results.
    df : pandas.DataFrame
        Cleaned dataset.
    design_columns : list of str
        Column names of the design matrix.

    Returns
    -------
    pandas.DataFrame
        DataFrame with covariate names, correlation coefficients and
        p‑values.
    """
    resid = res.schoenfeld_residuals
    mask = ~np.isnan(resid).any(axis=1)
    time_events = df.loc[mask, "time"]
    # Use provided column names
    if hasattr(res.params, "index"):
        param_names = list(res.params.index)
    else:
        param_names = design_columns
    res_df = pd.DataFrame(resid[mask], columns=param_names)
    records = []
    for col in res_df.columns:
        rho, p = spearmanr(time_events, res_df[col], nan_policy="omit")
        records.append({"covariate": col, "spearman_rho": rho, "p_value": p})
    return pd.DataFrame(records)


def compute_baseline_survival(res: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Extract baseline survival from a PHReg fitted model.

    Uses the baseline cumulative hazard function to compute baseline
    survival probabilities at the unique event times.

    Parameters
    ----------
    res : PHRegResults
        Fitted Cox model results.

    Returns
    -------
    times : ndarray
        Times at which the baseline survival is evaluated.
    baseline_surv : ndarray
        Baseline survival probabilities.
    """
    # Access baseline cumulative hazard function (dictionary keyed by stratum)
    # For PHReg without strata there is a single entry 0.
    ch_list = res.baseline_cumulative_hazard
    # ch_list is a list of lists: [times, cumulative hazard, survival] ; but note in
    # statsmodels <0.14 baseline_cumulative_hazard is list containing [times, cumhaz, surv].
    try:
        times = ch_list[0][0]
        cumhaz = ch_list[0][1]
    except Exception:
        # Fallback: use interpolation function if available
        baseline_func = res.baseline_cumulative_hazard_function[0]
        # Evaluate on sorted unique event times from the data used in the fit
        times = np.sort(np.unique(res.model.endog))
        cumhaz = baseline_func(times)
    baseline_surv = np.exp(-cumhaz)
    return times, baseline_surv


def plot_survival(overall: SurvfuncRight, by_contract: Dict[str, SurvfuncRight],
                  res: Any, df: pd.DataFrame, outdir: str, design_columns: List[str]) -> None:
    """Create and save survival plots.

    Produces three figures:
      1. Overall Kaplan–Meier survival curve.
      2. Kaplan–Meier curves by contract type.
      3. Adjusted survival curves by contract type using the fitted Cox
         model.  The baseline survival function from the model is
         combined with exponentiated linear predictors for each
         contract scenario.

    Parameters
    ----------
    overall : SurvfuncRight
        Kaplan–Meier survival function for the entire sample.
    by_contract : dict
        Mapping from contract type to Kaplan–Meier survival functions.
    res : PHRegResults
        Fitted Cox model results.
    df : pandas.DataFrame
        Cleaned dataset.
    outdir : str
        Directory where plots will be saved.
    covariate_cols : list of str
        Names of covariates included in the model (used to build
        scenario vectors).
    """
    os.makedirs(outdir, exist_ok=True)
    # 1. Overall Kaplan–Meier curve
    plt.figure()
    plt.step(overall.surv_times, overall.surv_prob, where="post")
    plt.xlabel("Time (months)")
    plt.ylabel("Survival Probability")
    plt.title("Overall Customer Retention (Kaplan–Meier)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.savefig(os.path.join(outdir, "overall_survival.png"))
    plt.close()
    # 2. Kaplan–Meier curves by contract
    plt.figure()
    for contract, sf in by_contract.items():
        plt.step(sf.surv_times, sf.surv_prob, where="post", label=contract)
    plt.xlabel("Time (months)")
    plt.ylabel("Survival Probability")
    plt.title("Customer Retention by Contract Type (Kaplan–Meier)")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.savefig(os.path.join(outdir, "survival_by_contract.png"))
    plt.close()
    # 3. Adjusted survival curves using baseline cumulative hazard
    try:
        times, baseline_surv = compute_baseline_survival(res)
        # Build scenario vectors: each scenario is a row vector aligned to design matrix
        # Use the median MonthlyCharges and set baseline levels for categorical variables
        median_charge = df["MonthlyCharges"].median()
        # Get column names from the fitted model
        design_cols = design_columns
        scenarios: Dict[str, pd.Series] = {}
        # Base scenario: month-to-month contract, baseline categories for other variables
        base_vec = pd.Series(0.0, index=design_cols)
        if "MonthlyCharges" in base_vec.index:
            base_vec["MonthlyCharges"] = median_charge
        # For SeniorCitizen, baseline is 0 (already zero by default)
        # Create scenarios per contract type
        for contract in df["Contract"].unique():
            vec = base_vec.copy()
            # Set dummy for contract
            dummy_col = f"Contract_{contract}"
            # If dummy exists, set to 1; else this is baseline (Month-to-month)
            if dummy_col in vec.index:
                vec[dummy_col] = 1.0
            scenarios[contract] = vec
        # Plot adjusted survival curves
        plt.figure()
        for name, vec in scenarios.items():
            # Compute linear predictor (beta * x)
            xb = float(np.dot(res.params, np.array(vec.values)))
            # Predicted survival = baseline survival ^ exp(xb)
            surv = baseline_surv ** np.exp(xb)
            plt.step(times, surv, where="post", label=name)
        plt.xlabel("Time (months)")
        plt.ylabel("Predicted Survival Probability")
        plt.title("Adjusted Survival Curves by Contract (Cox Model)")
        plt.legend()
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.savefig(os.path.join(outdir, "adjusted_survival_by_contract.png"))
        plt.close()
        print("Adjusted survival curves generated using baseline hazard")
    except Exception as e:
        print(f"Warning: Could not generate adjusted survival curves: {e}")
        print("Skipping adjusted survival plot generation.")


def main() -> None:
    """Execute the survival analysis workflow."""
    # Load and clean data
    df = load_and_clean(DATA_PATH)
    # Summarise data
    summarise_data(df)
    # Kaplan–Meier estimation
    overall_sf, contract_sf = kaplan_meier(df)
    # Compute median survival times and CIs
    try:
        median_time = overall_sf.quantile(0.5)
        median_ci = overall_sf.quantile_ci(0.5, alpha=0.05)
        print(f"Overall median survival time: {median_time} months (95% CI: {median_ci})")
    except Exception:
        print("Median survival time not reached within the observed period.")
    for name, sf in contract_sf.items():
        try:
            m = sf.quantile(0.5)
            ci = sf.quantile_ci(0.5, alpha=0.05)
            print(f"{name} median survival: {m} months (95% CI: {ci})")
        except Exception:
            print(f"{name} median survival time not reached within the observed period.")
    # Log‑rank test comparing contract types
    group_labels = df["Contract"].astype("category").cat.codes.values
    chi_sq, p_val = logrank_test(np.array(df["time"].values), np.array(df["event"].values), np.array(group_labels))
    print()
    print(f"Log‑rank test statistic: {chi_sq:.4f}, p‑value: {p_val:.4g}")
    # Select covariates for the Cox model
    covariate_cols = ["MonthlyCharges", "Contract", "InternetService",
                      "SeniorCitizen", "PaymentMethod", "PaperlessBilling",
                      "OnlineSecurity", "TechSupport"]
    # Fit Cox model
    cox_res, design_cols = fit_cox_model(df, covariate_cols)
    print()
    print("Cox proportional hazards model summary:")
    print(cox_res.summary())
    # Compute hazard ratios and confidence intervals
    coef = cox_res.params
    se = cox_res.bse
    hr = np.exp(coef)
    ci_lower = np.exp(coef - 1.96 * se)
    ci_upper = np.exp(coef + 1.96 * se)
    hr_table = pd.DataFrame({
        "Coefficient": coef,
        "HR": hr,
        "CI_lower": ci_lower,
        "CI_upper": ci_upper,
        "p_value": cox_res.pvalues
    })
    # Sort table by variable name for readability
    hr_table = hr_table.sort_index()
    print("\nHazard ratios and 95% confidence intervals:")
    print(hr_table)
    # Schoenfeld residual tests
    print()
    print("Schoenfeld residual correlation tests (Spearman):")
    residuals_df = schoenfeld_test(cox_res, df, design_cols)
    print(residuals_df)
    # Generate plots
    output_dir = os.path.join(os.path.dirname(__file__), "survival_plots_deep")
    plot_survival(overall_sf, contract_sf, cox_res, df, output_dir, design_cols)
    print(f"\nPlots saved in: {output_dir}")


if __name__ == "__main__":
    main()