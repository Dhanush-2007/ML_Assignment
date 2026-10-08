import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score

ROLLNO = "BT2024169"
DEGREE = 12

train = pd.read_csv(f"{ROLLNO}_train_var2.csv")
test = pd.read_csv(f"{ROLLNO}_test_var2.csv")

X = train.drop(columns="y")
y = train["y"]
X_test = test[X.columns]

folds = KFold(n_splits=5, shuffle=True, random_state=42)

# A finer alpha search
alpha_grid = np.logspace(-5, 4, 150)

model = Pipeline([
    ("polynomial", PolynomialFeatures(degree=DEGREE, include_bias=False)),
    ("scaler", StandardScaler()),
    ("ridge", RidgeCV(
        alphas=alpha_grid,
        cv=folds,
        scoring="neg_mean_squared_error"
    ))
])

model.fit(X, y)

chosen_alpha = model.named_steps["ridge"].alpha_

# Check the selected model with cross-validation
check_model = Pipeline([
    ("polynomial", PolynomialFeatures(degree=DEGREE, include_bias=False)),
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=chosen_alpha))
])

mse = -cross_val_score(
    check_model, X, y,
    cv=folds,
    scoring="neg_mean_squared_error",
    n_jobs=-1
).mean()

r2 = cross_val_score(
    check_model, X, y,
    cv=folds,
    scoring="r2",
    n_jobs=-1
).mean()

# Refit using every training sample
check_model.fit(X, y)
predictions = check_model.predict(X_test)

output = pd.DataFrame({"y": predictions})
output.to_csv(f"{ROLLNO}_pred_var2.csv", index=False)

print("VAR2")
print("Method:", "Ridge")
print("Degree:", DEGREE)
print("Alpha:", chosen_alpha)
print("CV MSE:", mse)
print("CV R2:", r2)
print("Saved:", f"{ROLLNO}_pred_var2.csv")
