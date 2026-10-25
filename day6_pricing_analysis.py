#!/usr/bin/env python3
"""
Day 6 - Pricing analytics: KPIs, price elasticity, demand prediction,
constrained price optimisation.

Usage:   python day6_pricing_analysis.py [--data DIR] [--out DIR]
Inputs:  day6_daily_pricing_sales.csv, day6_pricing_constraints.csv
Outputs: day6_price_recommendations.csv (+ supporting tables in <out>/day6_supporting/)

Pipeline
  Task 1  clean data, KPIs, reconciliation
  Task 2  elasticity: naive OLS -> controlled OLS -> censored (Tobit) log-normal demand
          model, category and SKU level, empirical-Bayes shrinkage of SKU elasticities
  Task 3  chronological split, baselines vs Tobit vs gradient boosting (MAE / RMSE)
  Task 4  constrained price grid search, revenue-max vs profit-max, Monte-Carlo uncertainty

Key modelling decision: units_sold on stockout days equals the opening inventory, so
it is a LOWER BOUND on true demand (right-censored). Treating it as demand biases
elasticities towards zero and under-predicts demand. We model log-demand as
normal and use the censored likelihood (Tobit): uncensored days contribute the
density, stockout days contribute P(demand >= inventory).
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)
RNG = np.random.default_rng(42)

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/mnt/user-data/uploads")
ap.add_argument("--out", default="/mnt/user-data/outputs")
args, _ = ap.parse_known_args()
SUP = os.path.join(args.out, "day6_supporting")
os.makedirs(SUP, exist_ok=True)


def banner(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


# ----------------------------------------------------------------------------------
# TASK 1 - DATA QUALITY & KPIs
# ----------------------------------------------------------------------------------
banner("TASK 1 - DATA QUALITY")
raw = pd.read_csv(os.path.join(args.data, "day6_daily_pricing_sales.csv"), parse_dates=["date"])
cons = pd.read_csv(os.path.join(args.data, "day6_pricing_constraints.csv"))
print(f"raw rows={len(raw)}  SKUs={raw.sku.nunique()}  days={raw.date.nunique()}  "
      f"-> expected {raw.sku.nunique() * raw.date.nunique()} SKU-days")

df = raw.copy()
issues = []

# 1) duplicates
n_dup = int(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
issues.append(("exact duplicate row", n_dup, "dropped"))

# 2) revenue / profit reconciliation against price x units
df["rev_check"] = df.selling_price_eur * df.units_sold
df["profit_check"] = (df.selling_price_eur - df.unit_cost_eur) * df.units_sold
bad_rev = (df.gross_revenue_eur - df.rev_check).abs() > 0.5
bad_prof = (df.gross_profit_eur - df.profit_check).abs() > 0.5
raw_rev_total = df.gross_revenue_eur.sum()
issues.append(("revenue != price x units (e.g. EUR 99,999 placeholder)", int(bad_rev.sum()),
               "recomputed as price x units"))
issues.append(("profit != (price-cost) x units", int(bad_prof.sum()), "none found"))
df["gross_revenue_eur"] = df.rev_check.round(2)
df["gross_profit_eur"] = df.profit_check.round(2)

# 3) impossible sessions (negative)
bad_sess = df.website_sessions <= 0
med_sess = df[~bad_sess].groupby(["sku", "weekday"]).website_sessions.median()
df["sessions"] = df.website_sessions.astype(float)
df.loc[bad_sess, "sessions"] = [med_sess[(s, w)] for s, w in zip(df.loc[bad_sess, "sku"], df.loc[bad_sess, "weekday"])]
issues.append(("sessions <= 0", int(bad_sess.sum()), "imputed: SKU x weekday median"))

# 4) impossible discount (>100 %)
bad_disc = df.discount_pct > 100
df.loc[bad_disc, "discount_pct"] = df.loc[df.promo_flag == 1, "discount_pct"].clip(upper=100).median()
issues.append(("discount_pct > 100", int(bad_disc.sum()), "set to median promo discount (not a model input)"))

# 5) missing competitor price
n_miss = int(df.competitor_price_eur.isna().sum())
df["comp"] = df.sort_values("date").groupby("sku").competitor_price_eur.transform(lambda s: s.ffill().bfill())
issues.append(("missing competitor price", n_miss, "forward/back-fill within SKU"))

# 6) stockout flag consistency: every flagged day sold exactly the opening stock;
#    some un-flagged days also sold exactly the opening stock -> also censored
df["censored"] = (df.units_sold >= df.inventory_available_start).astype(int)
n_incons = int(((df.censored == 1) & (df.stockout_flag == 0)).sum())
over = int((df.units_sold > df.inventory_available_start).sum())
issues.append(("units_sold == opening stock but stockout_flag=0", n_incons,
               "treated as censored (conservative); sensitivity run below"))
issues.append(("units_sold > opening stock", over, "none found"))
issues.append(("selling price < unit cost", int((df.selling_price_eur < df.unit_cost_eur).sum()), "none found"))

# 7) constraints file consistency with transaction file
chk = df.groupby("sku").agg(lp=("list_price_eur", "first"), uc=("unit_cost_eur", "first")).join(cons.set_index("sku"))
issues.append(("constraint price/cost differs from transactions",
               int(((chk.lp - chk.current_price_eur).abs() > 0.01).sum() + ((chk.uc - chk.unit_cost_eur).abs() > 0.01).sum()),
               "none found: current price == list price"))
issues_df = pd.DataFrame(issues, columns=["issue", "rows_affected", "treatment"])
print(issues_df.to_string(index=False))
print(f"clean rows={len(df)}  (SKU-days complete: {len(df) == 24 * 120})")

df["weekday"] = pd.Categorical(df.weekday, ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
df["cost_total"] = df.unit_cost_eur * df.units_sold


def kpis(g):
    rev, prof, u = g.gross_revenue_eur.sum(), g.gross_profit_eur.sum(), g.units_sold.sum()
    return pd.Series({
        "revenue_eur": rev, "gross_profit_eur": prof, "gross_margin_pct": 100 * prof / rev,
        "units_sold": u, "avg_selling_price_eur": rev / u,
        "stockout_rate_reported_pct": 100 * g.stockout_flag.mean(),
        "stockout_rate_corrected_pct": 100 * g.censored.mean()})


banner("TASK 1 - KPIs")
total = kpis(df)
print("TOTAL\n", total.round(2).to_string())
by_cat = df.groupby("category").apply(kpis)
by_mkt = df.groupby("market").apply(kpis)
by_sku = df.groupby(["sku", "category", "market"]).apply(kpis).reset_index()
print("\nBY CATEGORY\n", by_cat.round(2).to_string())
print("\nBY MARKET\n", by_mkt.round(2).to_string())
print("\nBY SKU (top/bottom by margin & stockouts)\n", by_sku.round(2).to_string(index=False))

banner("TASK 1 - RECONCILIATION")
rec = pd.DataFrame({
    "raw_file_revenue (as delivered, incl. duplicate)": [raw.gross_revenue_eur.sum()],
    "clean_revenue": [df.gross_revenue_eur.sum()],
    "sum(price x units)": [(df.selling_price_eur * df.units_sold).sum()],
    "cost_of_goods": [df.cost_total.sum()],
    "revenue - COGS": [df.gross_revenue_eur.sum() - df.cost_total.sum()],
    "clean_gross_profit": [df.gross_profit_eur.sum()]}).T
rec.columns = ["EUR"]
print(rec.round(2))
print(f"raw file revenue overstated by EUR {raw.gross_revenue_eur.sum() - (raw.selling_price_eur * raw.units_sold).sum():,.2f} "
      "vs price x units (the EUR 99,999 placeholder); the duplicate row adds a further double-count of one SKU-day")
print("max |profit_reported - (price-cost) x units| =", round((raw.gross_profit_eur - (raw.selling_price_eur - raw.unit_cost_eur) * raw.units_sold).abs().max(), 4))
by_cat.to_csv(f"{SUP}/kpi_by_category.csv"); by_mkt.to_csv(f"{SUP}/kpi_by_market.csv")
by_sku.to_csv(f"{SUP}/kpi_by_sku.csv", index=False); issues_df.to_csv(f"{SUP}/data_quality_log.csv", index=False)

# ----------------------------------------------------------------------------------
# TASK 2 - ELASTICITY
# ----------------------------------------------------------------------------------
SKUS = sorted(df.sku.unique())
CATS = sorted(df.category.unique())
WD = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]  # Sunday = base
SKU_CAT = df.drop_duplicates("sku").set_index("sku").category.to_dict()


def build_X(d, mode, beta_offset=None):
    """Design matrix. mode: 'pooled' | 'category' | 'sku' | 'offset' (price term fixed)."""
    cols = {}
    for s in SKUS:
        cols[f"sku_{s}"] = (d.sku == s).astype(float).values
    for w in WD:
        cols[f"wd_{w}"] = (d.weekday.astype(str) == w).astype(float).values
    cols["log_sessions"] = np.log(d.sessions.values)
    cols["promo"] = d.promo_flag.values.astype(float)
    cols["log_comp_price"] = np.log(d.comp.values)
    lp = np.log(d.selling_price_eur.values)
    offset = np.zeros(len(d))
    if mode == "pooled":
        cols["elast_all"] = lp
    elif mode == "category":
        for c in CATS:
            cols[f"elast_{c}"] = lp * (d.category == c).values
    elif mode == "sku":
        for s in SKUS:
            cols[f"elast_{s}"] = lp * (d.sku == s).values
    elif mode == "offset":
        offset = lp * d.sku.map(beta_offset).values
    X = pd.DataFrame(cols, index=d.index)
    return X, offset


class Tobit:
    """log(D) ~ N(X b + offset, s^2); right-censored at log(inventory) when stockout."""

    def fit(self, d, mode, beta_offset=None):
        self.mode, self.beta_offset = mode, beta_offset
        X, off = build_X(d, mode, beta_offset)
        self.names = list(X.columns)
        A = X.values
        ly = np.log(d.units_sold.values)
        lc = np.log(d.inventory_available_start.values)
        cen = d.censored.values.astype(bool)
        self._data = (A, off, ly, lc, cen)
        k = A.shape[1]
        b0 = np.linalg.lstsq(A, ly - off, rcond=None)[0]
        r = minimize(self._nll, np.r_[b0, np.log(0.15)], jac=True, method="L-BFGS-B",
                     options={"maxiter": 5000, "maxfun": 50000})
        self.p = r.x
        self.b, self.sigma = r.x[:k], float(np.exp(r.x[k]))
        self.loglik = -r.fun
        self.n_cens = int(cen.sum())
        self._cov(d)
        return self

    def _scores(self, p):
        A, off, ly, lc, cen = self._data
        k = A.shape[1]
        b, s = p[:k], np.exp(p[k])
        eta = A @ b + off
        z = (ly - eta) / s
        a = (lc - eta) / s
        lam = np.exp(norm.logpdf(a) - norm.logsf(a))
        ll = np.where(cen, norm.logsf(a), norm.logpdf(z) - np.log(s))
        g_eta = np.where(cen, lam / s, z / s)
        g_ls = np.where(cen, lam * a, z * z - 1.0)
        return ll, g_eta, g_ls

    def _nll(self, p):
        A = self._data[0]
        ll, g_eta, g_ls = self._scores(p)
        grad = np.r_[A.T @ g_eta, g_ls.sum()]
        return -ll.sum(), -grad

    def _cov(self, d):
        """Robust sandwich covariance. Clustered by SKU for pooled/category models; HC for SKU-slope model."""
        A = self._data[0]
        p = self.p
        k1 = len(p)
        H = np.zeros((k1, k1))
        for j in range(k1):
            e = np.zeros(k1); e[j] = 1e-5
            H[:, j] = (self._nll(p + e)[1] - self._nll(p - e)[1]) / 2e-5
        H = (H + H.T) / 2
        _, g_eta, g_ls = self._scores(p)
        S = np.column_stack([A * g_eta[:, None], g_ls])
        if self.mode in ("pooled", "category", "offset"):
            S = pd.DataFrame(S).groupby(d.sku.values).sum().values
        Hi = np.linalg.pinv(H)
        meat = S.T @ S
        self.cov = Hi @ meat @ Hi
        self.cov_model = Hi
        self.se = np.sqrt(np.clip(np.diag(self.cov), 0, None))

    def coef(self, name):
        i = self.names.index(name)
        return self.b[i], self.se[i]

    def eta(self, d, beta_offset=None):
        X, off = build_X(d, self.mode, self.beta_offset if beta_offset is None else beta_offset)
        return X.values @ self.b + off

    def mean_demand(self, d):
        return np.exp(self.eta(d) + 0.5 * self.sigma ** 2)

    def imputed_log_demand(self, d):
        """Observed log units, except on censored rows: E[log D | D >= stock] (truncated normal mean)."""
        eta = self.eta(d)
        ly = np.log(d.units_sold.values)
        lc = np.log(d.inventory_available_start.values)
        a = (lc - eta) / self.sigma
        lam = np.exp(norm.logpdf(a) - norm.logsf(a))
        return np.where(d.censored.values == 1, eta + self.sigma * lam, ly)


def ols_elasticity(d, controls=True):
    """OLS of log(units) on log(price) [+ controls]; returns elasticity."""
    X, _ = build_X(d, "pooled") if controls else (pd.DataFrame({"c": np.ones(len(d))}, index=d.index), 0)
    if controls:
        A = X.values
    else:
        A = np.column_stack([np.ones(len(d)), np.log(d.selling_price_eur.values)])
    y = np.log(d.units_sold.values)
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    return b[-1]


banner("TASK 2 - PRICE ELASTICITY")
print("Price-variation check (selling price / list price):")
print(df.assign(ratio=df.selling_price_eur / df.list_price_eur).groupby(["pricing_test_group", "promo_flag"]).ratio
      .describe()[["count", "mean", "std", "min", "max"]].round(3))

unc = df[df.censored == 0]
naive = ols_elasticity(df, controls=False)
naive_ctrl_all = ols_elasticity(df, True)
naive_ctrl_unc = ols_elasticity(unc, True)
m_pool = Tobit().fit(df, "pooled")
m_cat = Tobit().fit(df, "category")
m_sku = Tobit().fit(df, "sku")

print("\nWhy censoring matters - pooled price elasticity under different treatments:")
cmp = pd.DataFrame({
    "elasticity": [naive, naive_ctrl_all, naive_ctrl_unc, m_pool.coef("elast_all")[0]],
    "std_err": [np.nan, np.nan, np.nan, m_pool.coef("elast_all")[1]]},
    index=["(a) OLS log-log, no controls, all days",
           "(b) OLS + controls, all days (stockout sales treated as demand)",
           "(c) OLS + controls, drop stockout days (selection bias)",
           "(d) Censored Tobit + controls  <-- used"])
print(cmp.round(3).to_string())
print(f"\nTobit controls (pooled model): log_sessions={m_pool.coef('log_sessions')[0]:.3f} (se {m_pool.coef('log_sessions')[1]:.3f}), "
      f"promo={m_pool.coef('promo')[0]:.3f} (se {m_pool.coef('promo')[1]:.3f}), "
      f"log_comp_price={m_pool.coef('log_comp_price')[0]:.3f} (se {m_pool.coef('log_comp_price')[1]:.3f}), sigma={m_pool.sigma:.3f}")
print("weekday effects (vs Sunday): " + ", ".join(f"{w[:3]}={m_pool.coef('wd_' + w)[0]:+.3f}" for w in WD))

# robustness of pooled elasticity
rob = []
for lab, sub in [("all days", df), ("price-test group only", df[df.pricing_test_group == "Price test"]),
                 ("non-promo days only", df[df.promo_flag == 0]),
                 ("stockout flag as reported (not corrected)", df.assign(censored=df.stockout_flag))]:
    m = Tobit().fit(sub, "pooled")
    rob.append((lab, len(sub), m.coef("elast_all")[0], m.coef("elast_all")[1]))
rob = pd.DataFrame(rob, columns=["sample", "n", "elasticity", "se"])
print("\nRobustness (Tobit, pooled elasticity):\n", rob.round(3).to_string(index=False))
rob.to_csv(f"{SUP}/elasticity_robustness.csv", index=False)

# category level
cat_el = pd.DataFrame({
    "category": CATS,
    "elasticity": [m_cat.coef(f"elast_{c}")[0] for c in CATS],
    "std_err": [m_cat.coef(f"elast_{c}")[1] for c in CATS],
    "stockout_share_pct": [100 * df[df.category == c].censored.mean() for c in CATS]})
cat_el["ci90_low"] = cat_el.elasticity - 1.645 * cat_el.std_err
cat_el["ci90_high"] = cat_el.elasticity + 1.645 * cat_el.std_err
print("\nCategory elasticities (Tobit, robust SE clustered by SKU):\n", cat_el.round(3).to_string(index=False))

# SKU level + empirical-Bayes shrinkage towards category
sku_el = pd.DataFrame({
    "sku": SKUS, "category": [SKU_CAT[s] for s in SKUS],
    "beta_sku": [m_sku.coef(f"elast_{s}")[0] for s in SKUS],
    "se_sku": [m_sku.coef(f"elast_{s}")[1] for s in SKUS],
    "censored_share_pct": [100 * df[df.sku == s].censored.mean() for s in SKUS]})
sku_el["beta_cat"] = sku_el.category.map(cat_el.set_index("category").elasticity)
sku_el["se_cat"] = sku_el.category.map(cat_el.set_index("category").std_err)
dev = sku_el.beta_sku - sku_el.beta_cat
tau2 = max(0.0, float(np.mean(dev ** 2 - sku_el.se_sku ** 2)))
sku_el["weight_on_sku"] = tau2 / (tau2 + sku_el.se_sku ** 2)
sku_el["beta_shrunk"] = sku_el.beta_cat + sku_el.weight_on_sku * dev
sku_el["post_sd_dev"] = np.sqrt(sku_el.weight_on_sku) * sku_el.se_sku
sku_el["post_sd"] = np.sqrt(sku_el.post_sd_dev ** 2 + ((1 - sku_el.weight_on_sku) * sku_el.se_cat) ** 2)
sku_el["t_stat_sku"] = sku_el.beta_sku / sku_el.se_sku
sku_el["reliable_sku_estimate"] = (sku_el.t_stat_sku.abs() > 2) & (sku_el.censored_share_pct < 60)
hi, lo = sku_el.beta_shrunk + 1.645 * sku_el.post_sd, sku_el.beta_shrunk - 1.645 * sku_el.post_sd
sku_el["class"] = np.where(hi < -1, "elastic (|e|>1)", np.where(lo > -1, "inelastic (|e|<1)", "unit-elastic / uncertain"))
print(f"\nBetween-SKU variance of elasticity tau^2 = {tau2:.4f} (tau = {np.sqrt(tau2):.3f}) -> shrinkage weight "
      f"{sku_el.weight_on_sku.min():.2f}-{sku_el.weight_on_sku.max():.2f}")
print(sku_el[["sku", "category", "beta_sku", "se_sku", "censored_share_pct", "reliable_sku_estimate", "beta_shrunk", "post_sd", "class"]]
      .round(3).to_string(index=False))
sku_el.to_csv(f"{SUP}/elasticity_by_sku.csv", index=False)
cat_el.to_csv(f"{SUP}/elasticity_by_category.csv", index=False)
BETA = sku_el.set_index("sku").beta_shrunk.to_dict()

# residual diagnostics: drift over time / by group
d_diag = df.copy()
d_diag["resid"] = np.where(d_diag.censored == 0, np.log(d_diag.units_sold) - m_cat.eta(d_diag), np.nan)
print("\nDiagnostics - mean log residual (uncensored days) by month:",
      d_diag.groupby(d_diag.date.dt.to_period("M")).resid.mean().round(3).to_dict())
print("Diagnostics - residual sd by category:", d_diag.groupby("category").resid.std().round(3).to_dict())

# ----------------------------------------------------------------------------------
# TASK 3 - DEMAND PREDICTION
# ----------------------------------------------------------------------------------
banner("TASK 3 - DEMAND PREDICTION")
dates = np.sort(df.date.unique())
split_date = dates[90]
train, test = df[df.date < split_date].copy(), df[df.date >= split_date].copy()
print(f"Chronological split: train {train.date.min().date()} - {train.date.max().date()} ({len(train)} rows, "
      f"{train.censored.mean():.1%} censored) | test {test.date.min().date()} - {test.date.max().date()} "
      f"({len(test)} rows, {test.censored.mean():.1%} censored)")
print("""
Why stockout-day sales are a biased target: on those days units_sold = opening stock, i.e. a LOWER BOUND on demand.
Regressing on them (or dropping them) teaches a model that high-demand days are rarer than they are -> demand and
conversion are under-predicted, and because censoring is more frequent at LOW prices (where demand is highest) the
price slope is flattened (elasticity biased towards zero). Handling: (1) Tobit - stockout days enter as
P(D >= stock); (2) for the ML model, censored targets are replaced by E[log D | D >= stock] from the Tobit.
Never used as predictors: revenue, profit, stockout flag, units sold, opening inventory.
""")


def ml_features(d):
    f = pd.DataFrame(index=d.index)
    f["sku_code"] = d.sku.map({s: i for i, s in enumerate(SKUS)})
    f["market_code"] = d.market.map({"DE": 0, "FR": 1, "NL": 2})
    f["cat_code"] = d.category.map({c: i for i, c in enumerate(CATS)})
    f["log_price"] = np.log(d.selling_price_eur)
    f["log_comp"] = np.log(d.comp)
    f["price_vs_comp"] = d.selling_price_eur / d.comp
    f["price_vs_list"] = d.selling_price_eur / d.list_price_eur
    f["log_sessions"] = np.log(d.sessions)
    f["promo"] = d.promo_flag
    f["weekday"] = d.weekday.cat.codes
    f["is_weekend"] = (d.weekday.cat.codes >= 5).astype(int)
    return f


def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
def mae(a, b): return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


# baselines (trained on train only)
bl1 = train.groupby(["sku", "weekday"], observed=True).units_sold.mean()
bl2 = (train.groupby("sku").units_sold.sum() / train.groupby("sku").sessions.sum())
pred = {}
pred["Baseline 1: SKU x weekday mean units"] = np.array([bl1[(s, w)] for s, w in zip(test.sku, test.weekday)])
pred["Baseline 2: SKU conversion x sessions"] = test.sku.map(bl2).values * test.sessions.values

# Tobit (category slopes) on train
tb = Tobit().fit(train, "category")
pred["Censored Tobit (category elasticity)"] = tb.mean_demand(test)
tbs = Tobit().fit(train, "sku")
pred["Censored Tobit (SKU elasticity, unshrunk)"] = tbs.mean_demand(test)

# Gradient boosting: naive target vs censoring-corrected target
Ftr, Fte = ml_features(train), ml_features(test)
gb_params = dict(n_estimators=400, learning_rate=0.03, max_depth=3, subsample=0.8, min_samples_leaf=15, random_state=1)
y_naive = np.log(train.units_sold / train.sessions)
y_fix = tb.imputed_log_demand(train) - np.log(train.sessions.values)
gb_n = GradientBoostingRegressor(**gb_params).fit(Ftr, y_naive)
gb_c = GradientBoostingRegressor(**gb_params).fit(Ftr, y_fix)
pred["GBM, naive target (censored sales as demand)"] = np.exp(gb_n.predict(Fte)) * test.sessions.values
pred["GBM, censoring-corrected target"] = np.exp(gb_c.predict(Fte) + 0.5 * tb.sigma ** 2) * test.sessions.values

stock, obs = test.inventory_available_start.values, test.units_sold.values
unc_mask = test.censored.values == 0
rows = []
for name, dh in pred.items():
    cap = np.minimum(dh, stock)
    rows.append({
        "model": name,
        "MAE_all_days(sales=min(pred,stock))": mae(obs, cap), "RMSE_all_days": rmse(obs, cap),
        "MAE_non_stockout_days(uncapped)": mae(obs[unc_mask], dh[unc_mask]), "RMSE_non_stockout_days": rmse(obs[unc_mask], dh[unc_mask]),
        "mean_pred_demand_on_stockout_days": dh[~unc_mask].mean(),
        "stockout_hit_rate_pct": 100 * np.mean((dh >= stock) == (~unc_mask)),
        "bias_non_stockout_days(pred-obs)": float(np.mean(dh[unc_mask] - obs[unc_mask]))})
res3 = pd.DataFrame(rows).set_index("model")
print(res3.round(2).to_string())
print(f"(on stockout days mean opening stock = {stock[~unc_mask].mean():.1f}: true demand is >= this)")
res3.to_csv(f"{SUP}/model_comparison.csv")
best = res3["RMSE_all_days"].idxmin()
print(f"\nBest on RMSE (all days): {best}")
# out-of-sample elasticity stability
print(f"Pooled elasticity: train-only Tobit {Tobit().fit(train, 'pooled').coef('elast_all')[0]:.3f} vs full {m_pool.coef('elast_all')[0]:.3f}")

# ----------------------------------------------------------------------------------
# TASK 4 - OPTIMAL PRICE RECOMMENDATION
# ----------------------------------------------------------------------------------
banner("TASK 4 - OPTIMAL PRICES")
# Final demand model: shrunk SKU elasticities enter as a fixed offset, all other terms re-estimated on all data
final = Tobit().fit(df, "offset", beta_offset=BETA)
H = 30
fut_dates = pd.date_range(df.date.max() + pd.Timedelta(days=1), periods=H)
scen = []
for _, r in cons.iterrows():
    for dt in fut_dates:
        scen.append(dict(sku=r.sku, category=r.category, market=r.market, date=dt, weekday=dt.day_name(),
                         sessions=r.expected_daily_sessions, promo_flag=0, comp=r.competitor_reference_price_eur,
                         selling_price_eur=r.current_price_eur))
scen = pd.DataFrame(scen)
scen["weekday"] = pd.Categorical(scen.weekday, ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
scen["d0"] = final.mean_demand(scen)
D0 = scen.groupby("sku").d0.sum()            # 30-day unconstrained demand at CURRENT price
print(f"Scenario: {fut_dates[0].date()} - {fut_dates[-1].date()}, no promo, competitor at reference price, "
      "expected sessions from constraints file")

hist = df.groupby("sku").agg(hist_sessions=("sessions", "mean"), hist_obs_units=("units_sold", "mean"))
c = cons.set_index("sku").join(hist)
c["D0_30d"] = D0
c["beta"] = pd.Series(BETA)
c["beta_sd"] = sku_el.set_index("sku").post_sd
c["P_floor_margin"] = c.unit_cost_eur / (1 - c.minimum_margin_pct / 100)
c["P_lo"] = np.maximum(c.current_price_eur * (1 - c.max_price_change_pct / 100), c.P_floor_margin)
c["P_hi"] = c.current_price_eur * (1 + c.max_price_change_pct / 100)
assert (c.P_lo <= c.P_hi).all(), "infeasible SKU"
obs_rng = (df[df.promo_flag == 0].assign(r=lambda x: x.selling_price_eur / x.list_price_eur).groupby("sku").r.agg(["min", "max"]))
c["tested_max_ratio"] = obs_rng["max"]
cur_margin = 100 * (c.current_price_eur - c.unit_cost_eur) / c.current_price_eur
print(f"Current gross margin range {cur_margin.min():.1f}%-{cur_margin.max():.1f}% (all above floor: "
      f"{(cur_margin >= c.minimum_margin_pct).all()}); margin floor binds the price range for "
      f"{int((c.P_lo > c.current_price_eur * (1 - c.max_price_change_pct / 100)).sum())} SKUs")


def economics(p, r, beta=None):
    """30-day demand, units sold (stock-capped), revenue, profit at price p."""
    b = r.beta if beta is None else beta
    dem = r.D0_30d * (p / r.current_price_eur) ** b
    units = np.minimum(dem, r.available_inventory_next_30d)
    return dem, units, p * units, (p - r.unit_cost_eur) * units


out = []
for sku, r in c.iterrows():
    grid = np.round(np.arange(np.ceil(r.P_lo * 100) / 100, r.P_hi + 1e-9, 0.01), 2)
    dem, units, rev, prof = economics(grid, r)
    i_p, i_r = int(np.argmax(prof)), int(np.argmax(rev))
    dem0, u0, rev0, prof0 = economics(np.array([r.current_price_eur]), r)
    p_star = grid[i_p]
    # unconstrained (no change cap / margin floor / stock cap) theoretical optimum, for reference
    theo = r.unit_cost_eur * r.beta / (1 + r.beta) if r.beta < -1 else np.nan
    bind = []
    if abs(p_star - r.P_hi) < 0.011: bind.append("max price change (+)")
    if abs(p_star - r.P_lo) < 0.011: bind.append("margin floor" if r.P_lo > r.current_price_eur * (1 - r.max_price_change_pct / 100) + 0.011 else "max price change (-)")
    if dem[i_p] >= 0.99 * r.available_inventory_next_30d: bind.append("inventory cap (demand ~ stock)")
    out.append(dict(
        sku=sku, category=r.category, market=r.market, current_price_eur=r.current_price_eur,
        recommended_price_eur=p_star, price_change_pct=100 * (p_star / r.current_price_eur - 1),
        elasticity=r.beta, unit_cost_eur=r.unit_cost_eur,
        estimated_demand_30d=dem[i_p], expected_units_sold_30d=units[i_p],
        baseline_demand_30d=dem0[0], baseline_units_sold_30d=u0[0],
        baseline_revenue_eur=rev0[0], optimized_revenue_eur=rev[i_p],
        baseline_profit_eur=prof0[0], optimized_profit_eur=prof[i_p],
        profit_lift_eur=prof[i_p] - prof0[0], profit_lift_pct=100 * (prof[i_p] / prof0[0] - 1),
        revenue_max_price_eur=grid[i_r], revenue_at_rev_max_eur=rev[i_r], profit_at_rev_max_price_eur=prof[i_r],
        unconstrained_theory_price_eur=theo,
        gross_margin_at_rec_pct=100 * (p_star - r.unit_cost_eur) / p_star,
        min_margin_pct=r.minimum_margin_pct, max_change_pct=r.max_price_change_pct,
        inventory_30d=r.available_inventory_next_30d,
        binding_constraints="; ".join(bind) if bind else "none (interior optimum)",
        beyond_tested_price_range=bool(p_star / r.current_price_eur > r.tested_max_ratio + 0.005),
        price_vs_competitor_pct=100 * (p_star / r.competitor_reference_price_eur - 1)))
rec = pd.DataFrame(out)

# --- Monte-Carlo uncertainty on elasticity -> distribution of profit lift --------------
NS = 4000
cat_se = cat_el.set_index("category").std_err
draws_cat = {cat: RNG.normal(cat_el.set_index("category").elasticity[cat], cat_se[cat], NS) for cat in CATS}
lift_draws = np.zeros((NS, len(rec)))
for j, (_, rr) in enumerate(rec.iterrows()):
    s = sku_el.set_index("sku").loc[rr.sku]
    dev_mean = s.weight_on_sku * (s.beta_sku - s.beta_cat)
    b = draws_cat[s.category] + dev_mean + RNG.normal(0, s.post_sd_dev, NS)
    b = np.minimum(b, -0.2)   # demand must slope downward
    r = c.loc[rr.sku]
    _, _, _, p_new = economics(rr.recommended_price_eur, r, beta=b)
    _, _, _, p_old = economics(r.current_price_eur, r, beta=b)
    lift_draws[:, j] = p_new - p_old
rec["prob_lift_positive"] = (lift_draws > 0).mean(axis=0)
rec["lift_p05_eur"] = np.percentile(lift_draws, 5, axis=0)
rec["lift_p95_eur"] = np.percentile(lift_draws, 95, axis=0)
tot_draw = lift_draws.sum(axis=1)

rec["confidence"] = np.where(rec.prob_lift_positive >= 0.9, "High", np.where(rec.prob_lift_positive >= 0.75, "Medium", "Low"))

# --- Stress tests: parameters other than elasticity are also uncertain ----------------------
def portfolio_lift(beta_mult=1.0, level_mult=1.0, hist_sessions=False):
    tot = 0.0
    for _, rr in rec.iterrows():
        r = c.loc[rr.sku].copy()
        lv = level_mult * (r.hist_sessions / r.expected_daily_sessions if hist_sessions else 1.0)
        r["D0_30d"] = r.D0_30d * lv
        b = r.beta * beta_mult
        _, _, _, pn = economics(rr.recommended_price_eur, r, beta=b)
        _, _, _, po = economics(r.current_price_eur, r, beta=b)
        tot += pn - po
    return tot
stress = pd.DataFrame({"scenario": [
    "Central estimate", "Demand 25% more price-elastic", "Demand 50% more price-elastic",
    "Baseline demand level -15% (traffic/seasonality miss)", "Traffic = historical average instead of forecast",
    "Elasticity +25% AND demand level -15%"],
    "portfolio_profit_lift_eur": [portfolio_lift(), portfolio_lift(beta_mult=1.25), portfolio_lift(beta_mult=1.5),
                                  portfolio_lift(level_mult=0.85), portfolio_lift(hist_sessions=True),
                                  portfolio_lift(beta_mult=1.25, level_mult=0.85)]})
stress.to_csv(f"{SUP}/stress_tests.csv", index=False)
print("\nStress tests (lift of the recommended prices vs current prices):\n", stress.round(0).to_string(index=False))
print("expected vs historical sessions ratio:", (c.expected_daily_sessions / c.hist_sessions).round(2).describe()[["min", "mean", "max"]].round(2).to_dict())

# --- Output --------------------------------------------------------------------------------------
rec["recommended_price_eur"] = rec.recommended_price_eur.round(2)
core = ["sku", "category", "market", "current_price_eur", "recommended_price_eur", "price_change_pct",
        "estimated_demand_30d", "expected_units_sold_30d", "baseline_profit_eur", "optimized_profit_eur",
        "profit_lift_eur", "profit_lift_pct"]
extra = [x for x in rec.columns if x not in core]
final_csv = rec[core + extra].copy()
num = final_csv.select_dtypes("number").columns
final_csv[num] = final_csv[num].round(3)
final_csv.to_csv(os.path.join(args.out, "day6_price_recommendations.csv"), index=False)

pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
print(rec[["sku", "category", "current_price_eur", "recommended_price_eur", "price_change_pct", "elasticity",
           "estimated_demand_30d", "expected_units_sold_30d", "baseline_profit_eur", "optimized_profit_eur",
           "profit_lift_eur", "profit_lift_pct", "revenue_max_price_eur", "binding_constraints",
           "prob_lift_positive"]].to_string(index=False))

T = rec[["baseline_revenue_eur", "optimized_revenue_eur", "baseline_profit_eur", "optimized_profit_eur", "profit_lift_eur",
         "baseline_units_sold_30d", "expected_units_sold_30d"]].sum()
print("\nPORTFOLIO (30 days)")
print(T.to_string())
print(f"profit lift {T.profit_lift_eur:,.0f} EUR = {100 * T.profit_lift_eur / T.baseline_profit_eur:.1f}%   |  "
      f"revenue change {T.optimized_revenue_eur - T.baseline_revenue_eur:,.0f} EUR "
      f"({100 * (T.optimized_revenue_eur / T.baseline_revenue_eur - 1):.1f}%)   |  "
      f"units change {T.expected_units_sold_30d - T.baseline_units_sold_30d:,.0f} "
      f"({100 * (T.expected_units_sold_30d / T.baseline_units_sold_30d - 1):.1f}%)")
print(f"Elasticity uncertainty (Monte Carlo): portfolio lift mean {tot_draw.mean():,.0f}, 90% interval "
      f"[{np.percentile(tot_draw, 5):,.0f}, {np.percentile(tot_draw, 95):,.0f}], P(lift>0) = {(tot_draw > 0).mean():.1%}")
print(f"SKUs with P(lift>0) >= 90%: {(rec.prob_lift_positive >= .9).sum()} of {len(rec)}; "
      f"recommended price above tested range: {int(rec.beyond_tested_price_range.sum())}")
print("Price direction:", (np.sign(rec.price_change_pct.round(2))).value_counts().to_dict())
print("Binding constraints:\n", rec.binding_constraints.value_counts().to_string())

# revenue-max vs profit-max summary
rv = rec[["sku", "current_price_eur", "revenue_max_price_eur", "recommended_price_eur"]].copy()
rv["profit_if_revenue_max"] = rec.profit_at_rev_max_price_eur
rv["profit_if_profit_max"] = rec.optimized_profit_eur
rv["profit_given_up_by_revenue_max"] = rv.profit_if_profit_max - rv.profit_if_revenue_max
print("\nRevenue-max vs profit-max price (portfolio):  revenue-max policy profit "
      f"{rv.profit_if_revenue_max.sum():,.0f} vs profit-max {rv.profit_if_profit_max.sum():,.0f}; "
      f"revenue-max SKUs priced at the floor: {(rv.revenue_max_price_eur < rv.current_price_eur - 0.01).sum()} (lower than current), "
      f"above current: {(rv.revenue_max_price_eur > rv.current_price_eur + 0.01).sum()}")
rv.to_csv(f"{SUP}/revenue_max_vs_profit_max.csv", index=False)
rec.sort_values("profit_lift_eur", ascending=False).head(6)[
    ["sku", "category", "current_price_eur", "recommended_price_eur", "profit_lift_eur", "prob_lift_positive"]]\
    .to_csv(f"{SUP}/top6_priorities.csv", index=False)
cat_sum = rec.groupby("category")[["baseline_profit_eur", "optimized_profit_eur", "profit_lift_eur"]].sum()
cat_sum["lift_pct"] = 100 * cat_sum.profit_lift_eur / cat_sum.baseline_profit_eur
print("\nBy category:\n", cat_sum.to_string())
print("\nTop priorities:\n", rec.sort_values("profit_lift_eur", ascending=False).head(6)[
    ["sku", "category", "market", "current_price_eur", "recommended_price_eur", "price_change_pct", "profit_lift_eur",
     "prob_lift_positive", "binding_constraints"]].to_string(index=False))
print("\nDone. Outputs written to", args.out)
