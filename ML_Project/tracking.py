# pip install mlflow scikit-learn pandas numpy

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Home_Credit_AMT_CREDIT_Regression")

DATA_PATH = r"C:\Users\User\Downloads\Machine Learning End to End Project\csv\application_train.csv"

df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")

target = "AMT_CREDIT"

feature_cols = [
    "AMT_INCOME_TOTAL",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "CNT_CHILDREN",
    "DAYS_BIRTH",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
]

model_df = df[[target] + feature_cols].copy()

rows_before = len(model_df)
model_df = model_df.dropna()
rows_after = len(model_df)

print(f"Rows before dropna: {rows_before:,}")
print(f"Rows after dropna:  {rows_after:,}")
print(f"Dropped rows:       {rows_before - rows_after:,}")

X = model_df[feature_cols]
y = model_df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(f"Train size: {len(X_train):,}")
print(f"Test size:  {len(X_test):,}")


models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0),
    "Elastic Net": ElasticNet(alpha=1.0, l1_ratio=0.5),
    "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ),
}


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return r2, mae, rmse


mlflow.set_experiment("Home_Credit_AMT_CREDIT_Regression")

registered_model_name = "Home_Credit_AMT_CREDIT_Model"

results = []

for name, model in models.items():

    with mlflow.start_run(run_name=name) as run:

        model.fit(X_train, y_train)

        r2, mae, rmse = evaluate_model(model, X_test, y_test)

        mlflow.log_param("model_name", name)

        if hasattr(model, "alpha"):
            mlflow.log_param("alpha", model.alpha)

        if hasattr(model, "max_depth"):
            mlflow.log_param("max_depth", model.max_depth)

        if hasattr(model, "n_estimators"):
            mlflow.log_param("n_estimators", model.n_estimators)

        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=registered_model_name
        )

        results.append({
            "Model": name,
            "Run ID": run.info.run_id,
            "R²": r2,
            "MAE": mae,
            "RMSE": rmse
        })

        print(f"Logged and registered: {name}")
        print(f"R²: {r2:.4f}, MAE: {mae:,.2f}, RMSE: {rmse:,.2f}")
        print("-" * 60)


comparison_df = pd.DataFrame(results).sort_values("R²", ascending=False).reset_index(drop=True)

print("\nModel Comparison")
print(comparison_df)


# ---------------------------------------------------------
# Save 10 versions of the BEST model in MLflow Model Registry
# ---------------------------------------------------------

best_model_name = comparison_df.loc[0, "Model"]
best_model = models[best_model_name]

print("\nBest Model:", best_model_name)

for version_no in range(1, 11):

    with mlflow.start_run(run_name=f"{best_model_name}_version_{version_no}") as run:

        best_model.fit(X_train, y_train)

        r2, mae, rmse = evaluate_model(best_model, X_test, y_test)

        mlflow.log_param("version_number", version_no)
        mlflow.log_param("model_name", best_model_name)

        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name=registered_model_name
        )

        print(f"Saved MLflow model version {version_no}")


# ---------------------------------------------------------
# Show registered model versions
# ---------------------------------------------------------

client = mlflow.tracking.MlflowClient()

versions = client.search_model_versions(
    f"name='{registered_model_name}'"
)

version_data = []

for v in versions:
    version_data.append({
        "Model Name": v.name,
        "Version": v.version,
        "Stage": v.current_stage,
        "Status": v.status,
        "Run ID": v.run_id
    })

versions_df = pd.DataFrame(version_data).sort_values(
    "Version",
    ascending=True
)

print("\nRegistered Model Versions")
print(versions_df)


# ---------------------------------------------------------
# Load latest registered model and predict
# ---------------------------------------------------------

latest_version = versions_df["Version"].astype(int).max()

model_uri = f"models:/{registered_model_name}/{latest_version}"

loaded_model = mlflow.sklearn.load_model(model_uri)

sample_prediction = loaded_model.predict(X_test.head(5))

print("\nLatest Model URI:", model_uri)
print("Sample Predictions:")
print(sample_prediction)