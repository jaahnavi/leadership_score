"""
Two-Level Regression for Leadership Scorer
-------------------------------------------
Level 1 — Per-dimension calibration:
    For each of the 7 dimensions, fit a regression:
    H_dimension = a * A_dimension + b
    Reports R² to show how well auto tracks human per dimension.

Level 2 — Overall weight learning:
    H_Overall = w1*H_Influence + w2*H_Impact + ... + w7*H_Seniority
    Learns how your team implicitly weights each dimension.
    Outputs optimal weights ready to paste into weights.json.

Usage:
    python calibrate_and_train.py --csv scores_norm.csv
    python calibrate_and_train.py --csv scores_norm.csv --train-size 40
    python calibrate_and_train.py --csv scores_norm.csv --export-weights weights_ml.json

Dependencies:
    pip install pandas scikit-learn numpy
"""

import argparse
import json
import csv
import sys
from pathlib import Path

# ── Column name mapping ────────────────────────────────────────────────────────

AUTO_COLS = {
    "influence":   "A_Influence",
    "impact":      "A_Impact",
    "initiative":  "A_Initiative",
    "mentorship":  "A_Mentorship",
    "scope_scale": "A_Scope_scale",
    "ownership":   "A_Ownership",
    "seniority":   "A_Seniority",
}

HUMAN_COLS = {
    "influence":   "H_Influence",
    "impact":      "H_Impact",
    "initiative":  "H_Initiative",
    "mentorship":  "H_ Mentorship",   # note: space in your CSV header
    "scope_scale": "H_Scope/Scale",
    "ownership":   "H_Ownership",
    "seniority":   "H_Seniority",
}

HUMAN_OVERALL = "H_Overall"
AUTO_OVERALL  = "A_Overall"
DIMS = list(AUTO_COLS.keys())


# ── Data loading ───────────────────────────────────────────────────────────────

def load_csv(path: str) -> tuple[list[dict], list[str]]:
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
        headers = reader.fieldnames
    return rows, headers

def safe_float(val):
    try:
        return float(str(val).strip())
    except (ValueError, TypeError):
        return None

def extract_arrays(rows, col_a, col_b):
    """Extract two parallel float arrays, skipping rows with missing values."""
    a_vals, b_vals = [], []
    for r in rows:
        a = safe_float(r.get(col_a))
        b = safe_float(r.get(col_b))
        if a is not None and b is not None:
            a_vals.append(a)
            b_vals.append(b)
    return a_vals, b_vals


# ── Level 1: Per-dimension calibration ────────────────────────────────────────

def level1_calibration(rows: list[dict]) -> dict:
    """
    For each dimension: fit H = a*A + b
    Returns dict of {dim: {r2, slope, intercept, n}}
    """
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        import numpy as np
    except ImportError:
        print("Missing: pip install scikit-learn numpy")
        sys.exit(1)

    results = {}

    print("\n" + "="*65)
    print("  LEVEL 1 — Per-Dimension Calibration")
    print("  How well does the auto-scorer track human judgement?")
    print("="*65)
    print(f"  {'Dimension':<15} {'R²':>6}  {'Slope':>7}  {'Intercept':>10}  {'Signal'}")
    print("  " + "-"*60)

    for dim in DIMS:
        auto_vals, human_vals = extract_arrays(rows, AUTO_COLS[dim], HUMAN_COLS[dim])
        n = len(auto_vals)
        if n < 3:
            print(f"  {dim:<15}  {'N/A':>6}  (fewer than 3 valid rows)")
            continue

        X = np.array(auto_vals).reshape(-1, 1)
        y = np.array(human_vals)

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r2     = r2_score(y, y_pred)
        slope  = model.coef_[0]
        intercept = model.intercept_

        # Signal strength label
        if r2 >= 0.7:   signal = "STRONG   ✓"
        elif r2 >= 0.4: signal = "MODERATE ~"
        else:           signal = "WEAK     ✗"

        results[dim] = {
            "r2": round(r2, 4),
            "slope": round(float(slope), 4),
            "intercept": round(float(intercept), 4),
            "n": n,
            "signal": signal.strip(),
        }

        print(f"  {dim:<15} {r2:>6.3f}  {slope:>7.3f}  {intercept:>10.3f}  {signal}")

    print()
    print("  R² interpretation:")
    print("    >= 0.7  STRONG   — auto-scorer reliably tracks human judgement")
    print("    >= 0.4  MODERATE — auto-scorer partially tracks human judgement")
    print("    <  0.4  WEAK     — auto-scorer is missing what humans see here")

    return results


# ── Level 2: Overall weight learning ──────────────────────────────────────────

def level2_weights(rows: list[dict], train_size: int = None) -> dict:
    """
    Fit: H_Overall = w1*H_Influence + w2*H_Impact + ... + w7*H_Seniority
    Uses human dimension scores as inputs — learns implicit weighting.
    Then validates against auto scores to show end-to-end performance.
    """
    try:
        from sklearn.linear_model import LinearRegression, Ridge
        from sklearn.metrics import r2_score, mean_absolute_error
        import numpy as np
    except ImportError:
        print("Missing: pip install scikit-learn numpy")
        sys.exit(1)

    # Build feature matrix from human dimension scores
    X_rows, y_rows = [], []
    for r in rows:
        h_dims = [safe_float(r.get(HUMAN_COLS[d])) for d in DIMS]
        h_overall = safe_float(r.get(HUMAN_OVERALL))
        if None not in h_dims and h_overall is not None:
            X_rows.append(h_dims)
            y_rows.append(h_overall)

    X = np.array(X_rows)
    y = np.array(y_rows)
    n = len(y)

    if n < 5:
        print(f"\nNot enough valid rows for Level 2 ({n} found, need >= 5)")
        return {}

    # Train/test split
    if train_size and train_size < n:
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        split_msg = f"Train: {train_size}  Test: {n - train_size}"
    else:
        X_train, X_test = X, X
        y_train, y_test = y, y
        split_msg = f"All {n} resumes (no held-out test set)"

    # Use Ridge regression — more stable than plain linear with small datasets
    # alpha=0.1 is mild regularisation, prevents any one dimension dominating
    model = Ridge(alpha=0.1, fit_intercept=True)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    r2_train  = r2_score(y_train, y_pred_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    r2_test   = r2_score(y_test,  y_pred_test)  if len(y_test) > 1 else None
    mae_test  = mean_absolute_error(y_test, y_pred_test) if len(y_test) > 1 else None

    # Raw coefficients
    raw_coefs = {dim: float(model.coef_[i]) for i, dim in enumerate(DIMS)}

    # Normalise coefficients so they sum to 100 (directly usable as weights)
    coef_sum = sum(abs(c) for c in raw_coefs.values())
    if coef_sum > 0:
        opt_weights = {dim: round(abs(raw_coefs[dim]) / coef_sum * 100, 2)
                       for dim in DIMS}
    else:
        opt_weights = {dim: round(100/len(DIMS), 2) for dim in DIMS}

    print("\n" + "="*65)
    print("  LEVEL 2 — Overall Weight Learning")
    print("  H_Overall = w1*H_Influence + w2*H_Impact + ...")
    print("="*65)
    print(f"  {split_msg}")
    print(f"\n  {'Dimension':<15} {'Raw Coef':>10}  {'Opt Weight':>10}  {'Direction'}")
    print("  " + "-"*55)
    for dim in DIMS:
        coef = raw_coefs[dim]
        wt   = opt_weights[dim]
        direction = "positive ↑" if coef >= 0 else "negative ↓ (review)"
        print(f"  {dim:<15} {coef:>10.4f}  {wt:>9.2f}%  {direction}")

    print(f"\n  Model performance:")
    print(f"    Train R²  : {r2_train:.3f}   MAE: {mae_train:.2f} score points")
    if r2_test is not None:
        print(f"    Test  R²  : {r2_test:.3f}   MAE: {mae_test:.2f} score points")

    print(f"\n  Intercept : {model.intercept_:.3f}")
    print(f"\n  R² interpretation:")
    print(f"    >= 0.8  excellent  — weights explain human scoring well")
    print(f"    >= 0.6  good       — reasonable fit")
    print(f"    <  0.6  review     — dimensions may need rethinking")

    # End-to-end check: auto scores vs human overall
    print("\n" + "="*65)
    print("  END-TO-END CHECK")
    print("  How well do auto dimension scores predict H_Overall?")
    print("="*65)
    X_auto_rows, y_overall_rows = [], []
    for r in rows:
        a_dims = [safe_float(r.get(AUTO_COLS[d])) for d in DIMS]
        h_overall = safe_float(r.get(HUMAN_OVERALL))
        if None not in a_dims and h_overall is not None:
            X_auto_rows.append(a_dims)
            y_overall_rows.append(h_overall)

    if X_auto_rows:
        X_auto = np.array(X_auto_rows)
        y_auto = np.array(y_overall_rows)
        auto_model = Ridge(alpha=0.1)
        auto_model.fit(X_auto, y_auto)
        y_auto_pred = auto_model.predict(X_auto)
        r2_auto  = r2_score(y_auto, y_auto_pred)
        mae_auto = mean_absolute_error(y_auto, y_auto_pred)
        print(f"  Auto → H_Overall   R²: {r2_auto:.3f}   MAE: {mae_auto:.2f} score points")
        if r2_auto >= 0.6:
            print("  Good alignment — auto-scorer is broadly tracking human judgement")
        else:
            print("  Low alignment — consider reviewing weak dimensions from Level 1")

    return opt_weights


# ── Export weights.json ────────────────────────────────────────────────────────

def export_weights(opt_weights: dict, output_path: str):
    config = {
        "_max_scores": {
            "influence": 20, "impact": 25, "initiative": 15,
            "mentorship": 15, "scope_scale": 15, "ownership": 10,
            "seniority": 10,
        },
        "weights": opt_weights,
        "grade_thresholds": {"A": 85, "B": 70, "C": 55, "D": 40},
        "version": "ml-v1",
        "last_updated": str(__import__("datetime").date.today()),
        "source": "two-level-ridge-regression",
    }
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n  Weights saved to: {output_path}")
    print(f"  Copy this file to your project root to replace weights.json")


# ── Prediction helper ──────────────────────────────────────────────────────────

def predict_new(opt_weights: dict, auto_scores: dict) -> float:
    """
    Given auto dimension scores (0-1), predict overall score using learned weights.
    auto_scores: {influence: 0.5, impact: 0.8, ...}
    """
    total = sum(opt_weights[dim] * auto_scores.get(dim, 0) for dim in DIMS)
    return round(total, 1)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Two-level regression for leadership scorer calibration",
        epilog="""
Examples:
  python calibrate_and_train.py --csv scores_norm.csv
  python calibrate_and_train.py --csv scores_norm.csv --train-size 30
  python calibrate_and_train.py --csv scores_norm.csv --export-weights weights_ml.json
  python calibrate_and_train.py --csv scores_norm.csv --level 1
  python calibrate_and_train.py --csv scores_norm.csv --level 2
        """
    )
    parser.add_argument("--csv",            required=True, help="Path to scores CSV")
    parser.add_argument("--train-size",     type=int, default=None,
                        help="Number of rows to use for training (rest = test)")
    parser.add_argument("--export-weights", metavar="JSON",
                        help="Save optimised weights to this JSON file")
    parser.add_argument("--level",          type=int, choices=[1, 2],
                        help="Run only level 1 or level 2 (default: both)")
    args = parser.parse_args()

    rows, _ = load_csv(args.csv)
    print(f"\nLoaded {len(rows)} resumes from '{args.csv}'")

    opt_weights = {}

    if args.level in (None, 1):
        level1_calibration(rows)

    if args.level in (None, 2):
        opt_weights = level2_weights(rows, train_size=args.train_size)

    if args.export_weights and opt_weights:
        export_weights(opt_weights, args.export_weights)
    elif opt_weights:
        print("\n  To save these weights, re-run with --export-weights weights_ml.json")

if __name__ == "__main__":
    main()
