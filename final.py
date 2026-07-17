import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import math

# ------------------------------------------------------------
# 1. Load raw data from your Excel file
# ------------------------------------------------------------

def load_raw_data(xlsx_path: str):
    """
    Load option data, zero rates and S&P 500 underlying from the Excel file.
    Expects the following sheets (as in your file):
        - 'SuP500Options2008'
        - 'Zero_Rates 2008'
        - 'SuP500Underlying2008'
    """
    options = pd.read_excel(
        xlsx_path,
        sheet_name="SuP500Options2008",
        parse_dates=["Date_Observation", "Maturity_Date"],
    )

    zero_rates = pd.read_excel(
        xlsx_path,
        sheet_name="Zero_Rates 2008",
        parse_dates=["Date"],
    )

    underlying = pd.read_excel(
        xlsx_path,
        sheet_name="SuP500Underlying2008",
        parse_dates=["Date"],
    )

    return options, zero_rates, underlying


# ------------------------------------------------------------
# 2. Pre-processing: τ, mid-prices, underlying merge, OTM flag
# ------------------------------------------------------------

def preprocess_options(options: pd.DataFrame,
                       underlying: pd.DataFrame) -> pd.DataFrame:
    """
    - Compute time to maturity τ in years.
    - Compute mid quote (Bid+Ask)/2.
    - Merge S&P 500 underlying price S_t.
    - Create an OTM indicator:
        * Call is OTM if K > S_t
        * Put  is OTM if K < S_t
    """
    options = options.copy()
    options["days_to_maturity"] = (
        options["Maturity_Date"] - options["Date_Observation"]
    ).dt.days
    options["tau"] = options["days_to_maturity"] / 365.0

    options["mid_price"] = 0.5 * (options["Bid"] + options["Ask"])

    und = underlying.rename(columns={"Date": "Date_Observation",
                                     "Price_S&P": "S0"})
    options = options.merge(und, on="Date_Observation", how="left")

    def is_otm(row):
        if pd.isna(row["S0"]) or pd.isna(row["Strike"]):
            return False
        if row["Call_Put"] == "C":
            return row["Strike"] > row["S0"]
        elif row["Call_Put"] == "P":
            return row["Strike"] < row["S0"]
        return False

    options["is_otm"] = options.apply(is_otm, axis=1)

    return options


# ------------------------------------------------------------
# 3. Attach risk-free rate r_f (nearest maturity on same date)
# ------------------------------------------------------------

def attach_risk_free(options: pd.DataFrame,
                     zero_rates: pd.DataFrame) -> pd.DataFrame:
    """
    For each option (Date_Observation, days_to_maturity) pick the zero rate r_f
    from 'Zero_Rates 2008' with the same Date and the closest Days_to_Maturity.
    """
    options = options.copy()
    groups = {d: g.reset_index(drop=True)
              for d, g in zero_rates.groupby("Date")}

    def find_rf(row):
        g = groups.get(row["Date_Observation"])
        if g is None:
            return np.nan
        diffs = np.abs(g["Days_to_Maturity"].values - row["days_to_maturity"])
        idx = diffs.argmin()
        return float(g.loc[idx, "r_f"])

    options["r_f"] = options.apply(find_rf, axis=1)
    return options


# ------------------------------------------------------------
# 4. Compute implied forwards & dividend yields from ATM pairs
# ------------------------------------------------------------

def compute_forwards_from_atm_pairs(options: pd.DataFrame) -> pd.DataFrame:
    """
    For each (Date_Observation, Maturity_Date):
        - Find all strikes where both a call and a put exist.
        - Choose the strike K* closest to spot S_t (ATM).
        - Using mid prices C and P at K*, compute the implied forward F via

              C + K e^{-r τ} = P + F e^{-r τ}

          => F = K + (C - P) e^{r τ}.

        - Then use spot-forward parity

              F = S e^{(r - δ) τ}

          => δ = r - (1/τ) ln(F / S).
    """
    rows = []
    df = options.dropna(subset=["mid_price", "tau", "r_f", "S0"]).copy()

    for (date, mat), group in df.groupby(["Date_Observation", "Maturity_Date"]):
        S0 = group["S0"].iloc[0]

        calls = group[group["Call_Put"] == "C"][["Strike", "mid_price", "r_f", "tau"]]
        puts  = group[group["Call_Put"] == "P"][["Strike", "mid_price"]]

        if calls.empty or puts.empty:
            continue

        pairs = calls.merge(puts, on="Strike", suffixes=("_call", "_put"))
        if pairs.empty:
            continue

        pairs["dist_to_spot"] = (pairs["Strike"] - S0).abs()
        atm = pairs.loc[pairs["dist_to_spot"].idxmin()]

        K   = atm["Strike"]
        C   = atm["mid_price_call"]
        P   = atm["mid_price_put"]
        r   = atm["r_f"]
        tau = atm["tau"]

        if tau <= 0 or any(pd.isna(x) for x in [K, C, P, r, S0]):
            continue

        F = K + (C - P) * math.exp(r * tau)
        if F <= 0:
            continue

        delta = r - (1.0 / tau) * math.log(F / S0)

        rows.append(
            {
                "Date_Observation": date,
                "Maturity_Date": mat,
                "S0": S0,
                "r_f": r,
                "tau": tau,
                "K_ATM": K,
                "C_ATM": C,
                "P_ATM": P,
                "F_implied": F,
                "delta_implied": delta,
            }
        )

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# 4b. Estimate r and d from put-call parity regression
# ------------------------------------------------------------

def estimate_r_d_put_call_parity(options: pd.DataFrame,
                                 min_pairs: int = 3) -> pd.DataFrame:
    """
    For each (Date_Observation, Maturity_Date), regress Y = C - P on [1, K]:
        Y = alpha + beta*K + eps
    and infer r, d from alpha, beta.
    """
    rows = []
    df = options.copy()
    df = df.dropna(subset=["Bid", "Ask", "Strike", "tau", "S0"])
    df = df[(df["Bid"] > 0) & (df["Ask"] > 0) & (df["Bid"] <= df["Ask"])]
    df["mid_price"] = 0.5 * (df["Bid"] + df["Ask"])
    df = df[df["mid_price"] > 0]
    df = df[df["mid_price"] >= 0.50]

    for (date, mat), group in df.groupby(["Date_Observation", "Maturity_Date"]):
        tau = float(group["tau"].median())
        S0 = float(group["S0"].median())
        if tau <= 0 or S0 <= 0:
            continue

        calls = group[group["Call_Put"] == "C"][["Strike", "mid_price"]]
        puts = group[group["Call_Put"] == "P"][["Strike", "mid_price"]]
        pairs = calls.merge(puts, on="Strike", suffixes=("_call", "_put"))
        if len(pairs) < min_pairs:
            rows.append(
                {
                    "Date_Observation": date,
                    "Maturity_Date": mat,
                    "tau": tau,
                    "alpha_hat": np.nan,
                    "beta_hat": np.nan,
                    "r_hat": np.nan,
                    "d_hat": np.nan,
                    "n_pairs": len(pairs),
                    "r2": np.nan,
                    "valid_flag": False,
                }
            )
            continue

        K = pairs["Strike"].to_numpy(dtype=float)
        Y = (pairs["mid_price_call"] - pairs["mid_price_put"]).to_numpy(dtype=float)
        X = np.column_stack([np.ones_like(K), K])
        coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
        alpha_hat, beta_hat = coef
        resid = Y - (alpha_hat + beta_hat * K)

        def fit_and_validate(a_hat, b_hat, r2_val):
            if a_hat <= 0 or b_hat >= 0:
                return np.nan, np.nan, False
            d_hat = -(1.0 / tau) * math.log(a_hat / S0)
            r_hat = -(1.0 / tau) * math.log(-b_hat)
            return r_hat, d_hat, True

        ss_res = np.sum(resid ** 2)
        ss_tot = np.sum((Y - Y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        r_hat, d_hat, valid = fit_and_validate(alpha_hat, beta_hat, r2)

        if not valid and len(pairs) >= 5:
            # Drop top/bottom 5% residuals and refit
            keep = (resid >= np.percentile(resid, 5)) & (resid <= np.percentile(resid, 95))
            K2 = K[keep]
            Y2 = Y[keep]
            if len(K2) >= min_pairs:
                X2 = np.column_stack([np.ones_like(K2), K2])
                coef2, *_ = np.linalg.lstsq(X2, Y2, rcond=None)
                alpha_hat, beta_hat = coef2
                resid2 = Y2 - (alpha_hat + beta_hat * K2)
                ss_res = np.sum(resid2 ** 2)
                ss_tot = np.sum((Y2 - Y2.mean()) ** 2)
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
                r_hat, d_hat, valid = fit_and_validate(alpha_hat, beta_hat, r2)

        rows.append(
            {
                "Date_Observation": date,
                "Maturity_Date": mat,
                "tau": tau,
                "alpha_hat": alpha_hat,
                "beta_hat": beta_hat,
                "r_hat": r_hat,
                "d_hat": d_hat,
                "n_pairs": len(pairs),
                "r2": r2,
                "valid_flag": valid,
            }
        )

    return pd.DataFrame(rows)


def build_forwards_from_parity(options: pd.DataFrame,
                               parity_df: pd.DataFrame,
                               atm_forwards: pd.DataFrame) -> pd.DataFrame:
    """
    Build implied forwards using parity r_hat/d_hat when valid,
    fallback to ATM-pair implied forward otherwise.
    """
    base = parity_df.copy()
    # attach spot and tau from options
    spot_tau = options.groupby(["Date_Observation", "Maturity_Date"]).agg(
        S0=("S0", "median"), tau=("tau", "median")
    ).reset_index()
    base = base.merge(spot_tau, on=["Date_Observation", "Maturity_Date"], how="left")
    if "tau_x" in base.columns and "tau_y" in base.columns:
        base = base.rename(columns={"tau_x": "tau"}).drop(columns=["tau_y"])
    base["F_parity"] = base["S0"] * np.exp((base["r_hat"] - base["d_hat"]) * base["tau"])

    merged = base.merge(
        atm_forwards[
            ["Date_Observation", "Maturity_Date", "F_implied", "r_f", "delta_implied"]
        ],
        on=["Date_Observation", "Maturity_Date"],
        how="left",
    )

    merged["r_use"] = np.where(merged["valid_flag"], merged["r_hat"], merged["r_f"])
    merged["d_use"] = np.where(merged["valid_flag"], merged["d_hat"], merged["delta_implied"])
    merged["F_use"] = np.where(merged["valid_flag"], merged["F_parity"], merged["F_implied"])

    return merged[
        [
            "Date_Observation",
            "Maturity_Date",
            "tau",
            "S0",
            "r_use",
            "d_use",
            "F_use",
            "F_implied",
            "r_f",
            "delta_implied",
            "alpha_hat",
            "beta_hat",
            "r_hat",
            "d_hat",
            "n_pairs",
            "r2",
            "valid_flag",
        ]
    ]

# ------------------------------------------------------------
# 5. Convert puts into synthetic calls using the implied forward
# ------------------------------------------------------------

def convert_puts_to_calls(options: pd.DataFrame,
                          forwards: pd.DataFrame) -> pd.DataFrame:
    """
    Attach implied forward F and:
        - Calls: call_equiv_price = mid_price
        - Puts:  C = P + (F - K) e^{-r τ}
    Drop mid_price < 0.5 and negative synthetic calls.
    """
    df = options.copy()

    df = df.merge(
        forwards[["Date_Observation", "Maturity_Date", "F_use", "r_use", "d_use"]],
        on=["Date_Observation", "Maturity_Date"],
        how="left",
    )

    df = df[df["mid_price"] >= 0.50].copy()

    if "r_use" not in df.columns:
        df["r_use"] = np.nan
    if "d_use" not in df.columns:
        df["d_use"] = np.nan
    if "F_use" not in df.columns:
        df["F_use"] = np.nan

    df["r_use"] = df["r_use"].fillna(df["r_f"])
    if "delta_implied" in df.columns:
        df["d_use"] = df["d_use"].fillna(df["delta_implied"])
    else:
        df["d_use"] = df["d_use"].fillna(0.0)
    if "F_implied" in df.columns:
        df["F_use"] = df["F_use"].fillna(df["F_implied"])

    disc = np.exp(-df["r_use"] * df["tau"])
    df["call_equiv_price"] = np.where(
        df["Call_Put"] == "C",
        df["mid_price"],
        df["mid_price"] + (df["F_use"] - df["Strike"]) * disc,
    )

    df = df.dropna(subset=["call_equiv_price"])
    df = df[df["call_equiv_price"] > 0].copy()

    return df


# ------------------------------------------------------------
# 6. High-level wrapper: build Ludwig-style cleaned dataset
# ------------------------------------------------------------

def build_ludwig_dataset(xlsx_path: str):
    options, zero_rates, underlying = load_raw_data(xlsx_path)
    options = preprocess_options(options, underlying)
    options = attach_risk_free(options, zero_rates)
    atm_forwards = compute_forwards_from_atm_pairs(options)
    parity_df = estimate_r_d_put_call_parity(options)
    forwards_df = build_forwards_from_parity(options, parity_df, atm_forwards)
    options_clean = convert_puts_to_calls(options, forwards_df)
    return options_clean, forwards_df


# ------------------------------------------------------------
# 7. Black–Scholes on forward + implied vol solver
# ------------------------------------------------------------

def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_call_on_forward(F, K, r, delta, sigma, tau):
    """BS call price with dividend yield delta."""
    if sigma <= 0 or tau <= 0:
        intrinsic = F * math.exp(-delta * tau) - K * math.exp(-r * tau)
        return max(0.0, intrinsic)
    sqrt_tau = math.sqrt(tau)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    return (F * math.exp(-delta * tau) * norm_cdf(d1)
            - K * math.exp(-r * tau) * norm_cdf(d2))

def implied_vol_from_forward(price, F, K, r, delta, tau,
                             tol=1e-6, max_iter=100):
    """
    Bisection on σ for a call with dividend yield:
        C(σ) = F e^{-δ τ} N(d1) - K e^{-r τ} N(d2)
    """
    if tau <= 0 or F <= 0 or K <= 0:
        return np.nan

    # Intrinsic lower bound
    intrinsic = max(F * math.exp(-delta * tau) - K * math.exp(-r * tau), 0.0)
    if price < intrinsic - 1e-8:
        return np.nan

    low, high = 1e-4, 5.0
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        c_mid = bs_call_on_forward(F, K, r, delta, mid, tau)
        if abs(c_mid - price) < tol:
            return mid
        if c_mid > price:
            high = mid
        else:
            low = mid
    return mid


# ------------------------------------------------------------
# 7b. No-arbitrage filters from equations (3) and (4)
# ------------------------------------------------------------

def filter_no_arb_constraints(df: pd.DataFrame,
                              price_col: str = "call_equiv_price",
                              tol: float = 1e-8) -> pd.DataFrame:
    """
    Apply Eq. (3) price bounds and Eq. (4) strike/convexity constraints.
    Drops rows that violate bounds within each (Date_Observation, Maturity_Date).
    """
    df = df.copy()
    required = ["S0", "Strike", "r_use", "d_use", "tau", "F_use", price_col]
    required = [c for c in required if c in df.columns]
    df = df.dropna(subset=required)

    # Eq. (3):  S e^{-δτ} >= C >= max(0, S e^{-δτ} - K e^{-rτ})
    delta = df["d_use"]
    disc_r = np.exp(-df["r_use"] * df["tau"])
    disc_q = np.exp(-delta * df["tau"])
    upper = df["S0"] * disc_q
    lower = np.maximum(0.0, df["S0"] * disc_q - df["Strike"] * disc_r)
    price = df[price_col]
    ok_eq3 = (price <= upper + tol) & (price >= lower - tol)

    df = df[ok_eq3].copy()

    # Eq. (4): -e^{-rτ} <= dC/dK <= 0 and d2C/dK2 >= 0
    keep = pd.Series(False, index=df.index)
    for _, grp in df.groupby(["Date_Observation", "Maturity_Date"]):
        grp = grp.sort_values("Strike").copy()
        if len(grp) < 2:
            keep.loc[grp.index] = True
            continue

        K = grp["Strike"].to_numpy()
        C = grp[price_col].to_numpy()
        r = float(grp["r_use"].median())
        tau = float(grp["tau"].median())
        dC_dK = np.gradient(C, K)
        d2C_dK2 = np.gradient(dC_dK, K)

        lower_d1 = -np.exp(-r * tau) - tol
        upper_d1 = 0.0 + tol
        ok_d1 = (dC_dK >= lower_d1) & (dC_dK <= upper_d1)
        if len(grp) < 3:
            ok_d2 = np.ones(len(grp), dtype=bool)
        else:
            ok_d2 = d2C_dK2 >= -tol

        keep.loc[grp.index] = ok_d1 & ok_d2

    return df.loc[keep].copy()


# ------------------------------------------------------------
# 8. Build dataset, filter to OTM on 15/09/2008, compute IV & plot
# ------------------------------------------------------------

# 1) Run the full pipeline on your Excel file
xlsx_path = "/Users/davidemariatritto/Desktop/SuP500Options/SuP500Options2008.xlsx"
options_clean, forwards_df = build_ludwig_dataset(xlsx_path)
options_raw, zero_rates, underlying = load_raw_data(xlsx_path)
options_parity = preprocess_options(options_raw, underlying)
options_parity = attach_risk_free(options_parity, zero_rates)
parity_df = estimate_r_d_put_call_parity(options_parity)

# 2) Keep only OTM options
otm_only = options_clean[options_clean["is_otm"]].copy()

# 3) Filter for the trading day 15/09/2008
date_1509 = pd.Timestamp("2008-09-15")
otm_1509 = otm_only[otm_only["Date_Observation"] == date_1509].copy()

# Print OLS parity results for the selected date
parity_day = parity_df[parity_df["Date_Observation"] == date_1509].copy()
if not parity_day.empty:
    print("\nPut-call parity OLS results (2008-09-15):")
    print(
        parity_day[
            ["Maturity_Date", "tau", "alpha_hat", "beta_hat",
             "r_hat", "d_hat", "n_pairs", "r2", "valid_flag"]
        ].sort_values("Maturity_Date")
    )

    # Plot fitted line Y = C - P vs K for each maturity
    for mat in parity_day["Maturity_Date"].unique():
        sub = options_parity[
            (options_parity["Date_Observation"] == date_1509)
            & (options_parity["Maturity_Date"] == mat)
        ]
        calls = sub[sub["Call_Put"] == "C"][["Strike", "mid_price"]]
        puts = sub[sub["Call_Put"] == "P"][["Strike", "mid_price"]]
        pairs = calls.merge(puts, on="Strike", suffixes=("_call", "_put"))
        if len(pairs) < 3:
            continue
        K = pairs["Strike"].to_numpy(dtype=float)
        Y = (pairs["mid_price_call"] - pairs["mid_price_put"]).to_numpy(dtype=float)
        p_row = parity_day[parity_day["Maturity_Date"] == mat].iloc[0]
        alpha_hat = p_row["alpha_hat"]
        beta_hat = p_row["beta_hat"]
        if pd.isna(alpha_hat) or pd.isna(beta_hat):
            continue
        order = np.argsort(K)
        K_sorted = K[order]
        Y_fit = alpha_hat + beta_hat * K_sorted

        plt.figure(figsize=(7, 4))
        plt.scatter(K, Y, color="red", s=18, alpha=0.75, label="C - P")
        plt.plot(K_sorted, Y_fit, color="black", linewidth=1.5, label="OLS fit")
        plt.title(f"Put-call parity fit: {mat.date()}")
        plt.xlabel("Strike K")
        plt.ylabel("C - P")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Plot implied risk-free rate by maturity (days)
    parity_valid = parity_day[parity_day["valid_flag"]].copy()
    if not parity_valid.empty:
        plt.figure(figsize=(7, 4))
        tau_days = parity_valid["tau"] * 365.0
        plt.plot(tau_days, parity_valid["r_hat"], marker="s",
                 color="red", label="Implied r")
        plt.xlabel("Time to maturity (days)")
        plt.ylabel("Implied interest rate")
        plt.title("Implied interest rate (2008-09-15)")
        plt.legend()
        plt.tight_layout()
        plt.show()

# 4) Compute moneyness K/F and implied vol for each option
otm_1509 = otm_1509.dropna(subset=["F_use", "call_equiv_price", "r_use", "d_use", "tau"])

# moneyness
otm_1509["moneyness"] = otm_1509["Strike"] / otm_1509["F_use"]

# implied volatility (σ) from call_equiv_price
def row_iv(row):
    if row["S0"] <= 0 or row["F_use"] <= 0 or row["tau"] <= 0:
        return np.nan
    return implied_vol_from_forward(
        price=row["call_equiv_price"],
        F=row["F_use"],
        K=row["Strike"],
        r=row["r_use"],
        delta=row["d_use"],
        tau=row["tau"],
    )

otm_1509["implied_vol"] = otm_1509.apply(row_iv, axis=1)
otm_1509 = otm_1509.dropna(subset=["implied_vol"])
otm_1509["sqrt_tau"] = np.sqrt(otm_1509["tau"])


# Define a simple neural network structure
def neural_network(weights, inputs, num_neurons):
    """
    2 input features -> num_neurons hidden sigmoid -> 1 linear output
    weights layout:
      w1: (input_dim * num_neurons)
      b1: (num_neurons,)
      w2: (num_neurons,)
      b2: (1,)
    """
    input_dim = inputs.shape[1]
    w1_size = input_dim * num_neurons
    w1 = weights[:w1_size].reshape(input_dim, num_neurons)
    b1 = weights[w1_size : w1_size + num_neurons]
    w2 = weights[w1_size + num_neurons : w1_size + 2 * num_neurons]
    b2 = weights[-1]

    z = inputs @ w1 + b1
    hidden = 1.0 / (1.0 + np.exp(-z))
    out = hidden @ w2 + b2
    return out

# Model function to compute residuals
def model(weights, inputs, targets, num_neurons):
    predictions = neural_network(weights, inputs, num_neurons)
    return predictions - targets

def nguyen_widrow_init(num_neurons, input_dim):
    w1 = np.random.uniform(-0.5, 0.5, size=(input_dim, num_neurons))
    b1 = np.random.uniform(-0.5, 0.5, size=num_neurons)
    beta_nw = 0.7 * num_neurons ** (1.0 / input_dim)
    for j in range(num_neurons):
        norm = np.linalg.norm(w1[:, j])
        if norm > 0:
            w1[:, j] = beta_nw * w1[:, j] / norm
    b1 = np.random.uniform(-beta_nw, beta_nw, size=num_neurons)
    w2 = np.random.uniform(-0.5, 0.5, size=num_neurons)
    b2 = np.random.uniform(-0.5, 0.5)
    return w1, b1, w2, b2

def pack_weights(w1, b1, w2, b2):
    return np.concatenate([w1.ravel(), b1, w2, np.array([b2])])

def evidence_regularized_train_gnbr(
    inputs, targets, initial_weights, num_neurons,
    alpha0=1.0, beta0=1.0,
    max_iter=10, tol=1e-3,
    lm_max_nfev=None,
):
    alpha = float(alpha0)
    beta = float(beta0)
    w = initial_weights.copy()
    n = targets.size
    p = w.size
    log_evidence = np.nan
    eps = 1e-30

    for it in range(1, max_iter + 1):
        # GNBR iteration: LM step
        def reg_residuals(w_):
            pred = neural_network(w_, inputs, num_neurons)
            e = pred - targets
            data_res = np.sqrt(beta) * e
            prior_res = np.sqrt(alpha) * w_
            return np.concatenate([data_res, prior_res])

        res = least_squares(reg_residuals, w, method="lm", max_nfev=lm_max_nfev)
        w_map = res.x

        pred = neural_network(w_map, inputs, num_neurons)
        e = pred - targets
        ED = 0.5 * np.sum(e ** 2)
        EW = 0.5 * np.sum(w_map ** 2)

        # Gauss–Newton Hessian from LM Jacobian
        J_stack = res.jac
        J_data = J_stack[:n, :]
        JTJ = J_data.T @ J_data
        evals = np.linalg.eigvalsh(JTJ)
        evals = np.maximum(evals, 0.0)

        gamma = float(np.sum(evals / (alpha + evals + eps)))
        alpha_new = gamma / (2.0 * EW) if EW > 0 else alpha
        beta_new = (n - gamma) / (2.0 * ED) if ED > 0 else beta

        log_det = np.sum(np.log(alpha + evals + eps))
        log_evidence = (
            0.5 * (n * np.log(beta + eps) + p * np.log(alpha + eps))
            - (alpha * EW + beta * ED)
            - 0.5 * log_det
            - 0.5 * n * np.log(2.0 * np.pi)
        )

        print(
            f"GNBR iter {it:02d} | ED={ED:.6g} EW={EW:.6g} "
            f"alpha={alpha:.6g} beta={beta:.6g} gamma={gamma:.6g}"
        )

        if (abs(alpha_new - alpha) / max(alpha, 1e-15) < tol and
                abs(beta_new - beta) / max(beta, 1e-15) < tol):
            w, alpha, beta = w_map, float(alpha_new), float(beta_new)
            break

        w, alpha, beta = w_map, float(alpha_new), float(beta_new)

    return w, alpha, beta, log_evidence

# Prepare training data (features: moneyness, sqrt_tau; target: implied_vol)
x_data = otm_1509[["moneyness", "sqrt_tau"]].to_numpy(dtype=float)
y_data = otm_1509["implied_vol"].to_numpy(dtype=float)

# Normalize each feature
means = x_data.mean(axis=0)
stds = x_data.std(axis=0)
stds[stds == 0] = 1.0
inputs = (x_data - means) / stds

# Initial random weights sized for 2 inputs
num_neurons = 20
input_dim = inputs.shape[1]
w1_size = input_dim * num_neurons

# Enforce Eq. (3) and (4) constraints; drop violating observations
otm_1509 = filter_no_arb_constraints(otm_1509, price_col="call_equiv_price")
x_data = otm_1509[["moneyness", "sqrt_tau"]].to_numpy(dtype=float)
y_data = otm_1509["implied_vol"].to_numpy(dtype=float)
means = x_data.mean(axis=0)
stds = x_data.std(axis=0)
stds[stds == 0] = 1.0
inputs = (x_data - means) / stds

# Initial LM fit (pre-GNBR)
initial_weights = np.random.randn(w1_size + num_neurons + num_neurons + 1) * 0.1
lm_result = least_squares(
    lambda w: model(w, inputs, y_data, num_neurons),
    initial_weights,
    method="lm",
)
lm_weights = lm_result.x

# GNBR ensemble (B=2)
ensemble_results = []
num_starts = 2
for b in range(1, num_starts + 1):
    if b == 1:
        initial_weights = lm_weights
    else:
        w1, b1, w2, b2 = nguyen_widrow_init(num_neurons, input_dim)
        initial_weights = pack_weights(w1, b1, w2, b2)
    print(f"Network {b}/{num_starts} initialization done.")
    weights, alpha, beta, logev = evidence_regularized_train_gnbr(
        inputs, y_data, initial_weights, num_neurons, alpha0=0.0, beta0=1.0, max_iter=10
    )
    preds = neural_network(weights, inputs, num_neurons)
    cost = np.sum((preds - y_data) ** 2)
    ensemble_results.append(
        {
            "weights": weights,
            "alpha": alpha,
            "beta": beta,
            "log_evidence": logev,
            "cost": cost,
        }
    )

ensemble_results.sort(key=lambda r: r["cost"])
optimized_weights = ensemble_results[0]["weights"]
predictions = neural_network(optimized_weights, inputs, num_neurons)
otm_1509["pred_iv"] = predictions

# Plot results: observed red, NN colored per tau
plt.figure(figsize=(10, 5))
unique_tau = np.sort(otm_1509["tau"].unique())
cmap = plt.get_cmap("plasma")
colors = cmap(np.linspace(0, 1, len(unique_tau)))
obs_labeled = False
for color, tau_val in zip(colors, unique_tau):
    mask = otm_1509["tau"] == tau_val
    m_slice = otm_1509.loc[mask, "moneyness"].to_numpy()
    y_slice = (y_data * 100)[mask]
    pred_slice = (predictions * 100)[mask]
    order = np.argsort(m_slice)
    plt.scatter(
        m_slice,
        y_slice,
        facecolors="white",
        edgecolors="red",
        marker="o",
        s=30,
        alpha=0.85,
        label="Observed IV" if not obs_labeled else None,
    )
    obs_labeled = True
    plt.plot(
        m_slice[order],
        pred_slice[order],
        linestyle="solid",
        color="black",
        alpha=0.85,
        label=f"tau={tau_val:.3f} (NN)",
    )

plt.title('LM NN on 15/09/2008 OTM options (features: moneyness, sqrt(tau))')
plt.xlabel('K / F (moneyness)')
plt.ylabel('Implied Volatility (%)')
plt.legend(title="Lines by tau (years)")
plt.tight_layout()
plt.show()

# 3D implied volatility surface (curves per maturity)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")
obs_labeled = False
for tau_val in unique_tau:
    mask = otm_1509["tau"] == tau_val
    m_slice = otm_1509.loc[mask, "moneyness"].to_numpy()
    y_obs = y_data[mask] * 100.0
    y_pred = predictions[mask] * 100.0
    ttm_days = np.full_like(m_slice, tau_val * 365.0, dtype=float)
    order = np.argsort(m_slice)
    ax.plot(
        m_slice[order],
        ttm_days[order],
        y_pred[order],
        color="black",
        linewidth=0.8,
        alpha=0.9,
    )
    ax.scatter(
        m_slice,
        ttm_days,
        y_obs,
        color="red",
        s=10,
        alpha=0.8,
        label="Observed IV" if not obs_labeled else None,
    )
    obs_labeled = True

ax.set_title("Implied volatility surface")
ax.set_xlabel("K / F (moneyness)")
ax.set_ylabel("Time to maturity (days)")
ax.set_zlabel("Implied Volatility (%)")
ax.invert_yaxis()

if obs_labeled:
    ax.legend(loc="upper right")
plt.tight_layout()
plt.show()

# 3D surface: evaluate NN on a grid in (moneyness, tau)
m_min, m_max = otm_1509["moneyness"].min(), otm_1509["moneyness"].max()
t_min, t_max = otm_1509["tau"].min(), otm_1509["tau"].max()
m_grid = np.linspace(m_min, m_max, 60)
t_grid = np.linspace(t_min, t_max, 40)
M, T = np.meshgrid(m_grid, t_grid)
sqrt_T = np.sqrt(T)
grid_inputs = np.column_stack([M.ravel(), sqrt_T.ravel()])
grid_inputs = (grid_inputs - means) / stds
grid_pred = neural_network(optimized_weights, grid_inputs, num_neurons).reshape(M.shape) * 100.0

fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(111, projection="3d")
T_days = T * 365.0 
surf = ax.plot_surface(
    M, T_days, grid_pred,
    cmap="viridis",
    linewidth=0,
    antialiased=True,
    alpha=0.9,
)
ax.set_title("Implied Volatility Surface (NN fit)")
ax.set_xlabel("K / F (moneyness)")
ax.set_ylabel("Time to maturity (days)")
ax.set_zlabel("Implied Volatility (%)")
ax.invert_yaxis()
fig.colorbar(surf, ax=ax, pad=0.1, label="IV (%)")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Retrieve SPD via numerical differentiation of model call prices
# ------------------------------------------------------------
def compute_spd(df_slice: pd.DataFrame):
    df_slice = df_slice.sort_values("Strike")
    K = df_slice["Strike"].to_numpy()
    F = df_slice["F_use"].to_numpy()
    r = df_slice["r_use"].to_numpy()
    tau_val = float(df_slice["tau"].iloc[0])
    sigma = df_slice["pred_iv"].to_numpy()
    delta = df_slice["d_use"].to_numpy()
    call_pred = np.array(
        [bs_call_on_forward(F[i], K[i], r[i], delta[i], sigma[i], tau_val) for i in range(len(K))]
    )
    dC_dK = np.gradient(call_pred, K)
    d2C_dK2 = np.gradient(dC_dK, K)
    spd = np.exp(r * tau_val) * d2C_dK2
    mny = K / F
    return mny, spd


plt.figure(figsize=(10, 5))
for tau_val, grp in otm_1509.groupby("tau"):
    if len(grp) < 3:
        continue
    mny, spd = compute_spd(grp)
    plt.plot(mny, spd, label=f"tau={tau_val:.3f}")

plt.title("State price density from LM NN (finite differences)")
plt.xlabel("K / F (moneyness)")
plt.ylabel("SPD")
plt.legend(title="tau (years)")
plt.tight_layout()
plt.show()
 