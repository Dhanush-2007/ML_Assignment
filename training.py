import warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_validate

warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

ROLLNO = "BT2024169"

SEED = 42
FOLDS = 5

# Alpha values for Ridge
RIDGE_ALPHAS = [
    0.0001,
    0.001,
    0.005,
    0.01,
    0.05,
    0.1,
    0.25,
    0.5,
    1,
    2,
    5,
    10,
    25,
    50,
    100,
    500,
    1000
]

# Alpha values for Lasso
LASSO_ALPHAS = [
    0.00001,
    0.00005,
    0.0001,
    0.0002,
    0.0005,
    0.001,
    0.002,
    0.005,
    0.007,
    0.01,
    0.02,
    0.05,
    0.1
]

CV = KFold(
    n_splits=FOLDS,
    shuffle=True,
    random_state=SEED
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, X, y):

    result = cross_validate(
        model,
        X,
        y,
        cv=CV,
        scoring={
            "mse": "neg_mean_squared_error",
            "r2": "r2"
        },
        n_jobs=1
    )

    mse = -result["test_mse"].mean()
    r2 = result["test_r2"].mean()

    return mse, r2


# ============================================================
# CHECK ONE DEGREE
# ============================================================

def check_degree(X_poly, y, degree):

    results = []

    # --------------------------------------------------------
    # OLS
    # --------------------------------------------------------

    mse, r2 = evaluate(
        LinearRegression(),
        X_poly,
        y
    )

    results.append({
        "method": "OLS",
        "mse": mse,
        "r2": r2,
        "alpha": None
    })


    # --------------------------------------------------------
    # RIDGE
    # --------------------------------------------------------

    for alpha in RIDGE_ALPHAS:

        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=alpha)
        )

        mse, r2 = evaluate(
            model,
            X_poly,
            y
        )

        results.append({
            "method": "Ridge",
            "mse": mse,
            "r2": r2,
            "alpha": alpha
        })


    # --------------------------------------------------------
    # LASSO
    # --------------------------------------------------------

    for alpha in LASSO_ALPHAS:

        model = make_pipeline(
            StandardScaler(),
            Lasso(
                alpha=alpha,
                max_iter=20000,
                tol=1e-3,
                random_state=SEED
            )
        )

        mse, r2 = evaluate(
            model,
            X_poly,
            y
        )

        results.append({
            "method": "Lasso",
            "mse": mse,
            "r2": r2,
            "alpha": alpha
        })


    # Return the candidate with lowest CV MSE
    return min(results, key=lambda item: item["mse"])


# ============================================================
# RUN COMPLETE PROBLEM
# ============================================================

def solve_problem(problem, maximum_degree):

    print("\n")
    print("=" * 75)
    print(f"PROBLEM {problem}")
    print("=" * 75)

    filename = f"{ROLLNO}_train_var{problem}.csv"

    train = pd.read_csv(filename)

    X = train.drop(columns="y")
    y = train["y"]

    print("Training file :", filename)
    print("Samples       :", len(train))
    print("Input features:", X.shape[1])
    print(f"Degrees       : 1 to {maximum_degree}")
    print(f"CV folds      : {FOLDS}")

    best_overall = None

    print("\n" + "-" * 75)

    # --------------------------------------------------------
    # Degree search
    # --------------------------------------------------------

    for degree in range(1, maximum_degree + 1):

        polynomial = PolynomialFeatures(
            degree=degree,
            include_bias=False
        )

        X_poly = polynomial.fit_transform(X)

        best_degree = check_degree(
            X_poly,
            y,
            degree
        )

        method = best_degree["method"]
        mse = best_degree["mse"]
        r2 = best_degree["r2"]
        alpha = best_degree["alpha"]

        if alpha is None:
            alpha_text = ""
        else:
            alpha_text = f" | alpha = {alpha:g}"

        print(
            f"Degree {degree:2d} | "
            f"Best = {method:<6} | "
            f"MSE = {mse:.8f} | "
            f"R2 = {r2:.8f}"
            f"{alpha_text}"
        )

        # Keep overall best
        if (
            best_overall is None
            or mse < best_overall["mse"]
        ):
            best_overall = {
                "degree": degree,
                "method": method,
                "mse": mse,
                "r2": r2,
                "alpha": alpha
            }


    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("\n" + "=" * 75)
    print(f"BEST MODEL FOR PROBLEM {problem}")
    print("=" * 75)

    print("Method :", best_overall["method"])
    print("Degree :", best_overall["degree"])

    if best_overall["alpha"] is not None:
        print("Alpha  :", best_overall["alpha"])

    print("CV MSE :", best_overall["mse"])
    print("CV R2  :", best_overall["r2"])

    print("=" * 75)

    return best_overall


# ============================================================
# MAIN
# ============================================================

def main():

    # Problem 1:
    # Degree 1 through 10
    result_var1 = solve_problem(
        problem=1,
        maximum_degree=10
    )

    # Problem 2:
    # Degree 1 through 20
    result_var2 = solve_problem(
        problem=2,
        maximum_degree=20
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n\n")
    print("#" * 75)
    print("FINAL SUMMARY")
    print("#" * 75)

    print("\nVAR1")
    print("Method :", result_var1["method"])
    print("Degree :", result_var1["degree"])
    print("Alpha  :", result_var1["alpha"])
    print("MSE    :", result_var1["mse"])
    print("R2     :", result_var1["r2"])

    print("\nVAR2")
    print("Method :", result_var2["method"])
    print("Degree :", result_var2["degree"])
    print("Alpha  :", result_var2["alpha"])
    print("MSE    :", result_var2["mse"])
    print("R2     :", result_var2["r2"])

    print("\n" + "#" * 75)


if __name__ == "__main__":
    main()