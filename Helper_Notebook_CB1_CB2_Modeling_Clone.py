# Databricks notebook source
# MAGIC %md
# MAGIC # Assignment 2 Template (Databricks): FNN + CV + Grid Search + Feature Comparisons + MLflow
# MAGIC
# MAGIC This helper notebook is organized as reusable **template cells** for Assignment 2.
# MAGIC
# MAGIC It covers:
# MAGIC 1. Data loading and JSON parsing
# MAGIC 2. Feature matrix construction (`desc`, `maccs`, `morgan`, `all`)
# MAGIC 3. Feedforward Neural Network (FNN) for regression and classification
# MAGIC 4. 5-fold cross-validation
# MAGIC 5. Grid search over FNN hyperparameters
# MAGIC 6. Feature-set comparisons
# MAGIC 7. MLflow logging for params + metrics

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0) Load Delta table

# COMMAND ----------

TABLE = "workspace.default.cb_1_cb_2_features_v_2"
SAMPLE_FRAC = 0.35
RANDOM_SEED = 42

# Use this experiment path in your own Databricks workspace
MLFLOW_EXPERIMENT = "/Users/your.name@school.edu/CB1_CB2_Assignment2"

df_spark = spark.table(TABLE)
cols = [
    "molecule_chembl_id",
    "cb1_p", "cb2_p", "delta_p_cb1_minus_cb2",
    "cb1_active", "cb2_active", "selectivity_direction",
    "morgan_fp_str", "maccs_fp_str", "rdkit_desc50_str"
]

pdf = (
    df_spark.select(*[c for c in cols if c in df_spark.columns])
    .sample(withReplacement=False, fraction=SAMPLE_FRAC, seed=RANDOM_SEED)
    .toPandas()
)

print("Rows in pandas sample:", len(pdf))
pdf.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) Parse JSON arrays and build descriptor matrices

# COMMAND ----------

import json
import random
import numpy as np
import pandas as pd


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass


def parse_json_array(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    try:
        return json.loads(x)
    except Exception:
        return None


pdf["morgan_fp"] = pdf["morgan_fp_str"].apply(parse_json_array)
pdf["maccs_fp"] = pdf["maccs_fp_str"].apply(parse_json_array)
pdf["rdkit_desc50"] = pdf["rdkit_desc50_str"].apply(parse_json_array)

pdf = pdf.dropna(subset=["morgan_fp", "maccs_fp", "rdkit_desc50"]).reset_index(drop=True)

X_morgan = np.vstack(pdf["morgan_fp"].apply(lambda v: np.array(v, dtype=np.float32)))
X_maccs = np.vstack(pdf["maccs_fp"].apply(lambda v: np.array(v, dtype=np.float32)))
X_desc = np.vstack(pdf["rdkit_desc50"].apply(lambda v: np.array(v, dtype=np.float32)))
X_all = np.hstack([X_desc, X_maccs, X_morgan])

FEATURES = {
    "desc": X_desc,
    "maccs": X_maccs,
    "morgan": X_morgan,
    "all": X_all,
}

for name, mat in FEATURES.items():
    print(f"{name:>6}: {mat.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) FNN utility functions (shared by regression + classification)

# COMMAND ----------

import mlflow
import mlflow.tensorflow
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping


mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(MLFLOW_EXPERIMENT)


def make_fnn_regressor(input_dim, hidden_layers=(256, 128), dropout=0.2, learning_rate=1e-3):
    model = Sequential()
    model.add(Dense(hidden_layers[0], activation="relu", input_shape=(input_dim,)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
    model.add(Dense(1, activation="linear"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse"), "mae"],
    )
    return model


def make_fnn_classifier(
    input_dim,
    n_classes,
    hidden_layers=(256, 128),
    dropout=0.2,
    learning_rate=1e-3,
):
    model = Sequential()
    model.add(Dense(hidden_layers[0], activation="relu", input_shape=(input_dim,)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

    if n_classes == 2:
        model.add(Dense(1, activation="sigmoid"))
        loss = "binary_crossentropy"
    else:
        model.add(Dense(n_classes, activation="softmax"))
        loss = "sparse_categorical_crossentropy"

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=["accuracy"],
    )
    return model

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) Regression template cell: CV + grid search + feature comparisons
# MAGIC
# MAGIC Set `REG_TARGET` to one of:
# MAGIC - `cb1_p`
# MAGIC - `cb2_p`
# MAGIC - `delta_p_cb1_minus_cb2`

# COMMAND ----------

REG_TARGET = "cb1_p"

reg_grid = {
    "hidden_layers": [(128, 64), (256, 128)],
    "dropout": [0.1, 0.3],
    "learning_rate": [1e-3],
    "batch_size": [32, 64],
    "epochs": [60],
}

set_seed(RANDOM_SEED)
y_reg = pd.to_numeric(pdf[REG_TARGET], errors="coerce")
mask_reg = y_reg.notna()
y_reg = y_reg.loc[mask_reg].to_numpy(dtype=np.float32)

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

reg_results = []

for feature_name, X_full in FEATURES.items():
    X_reg = X_full[mask_reg.to_numpy()]

    scaler = StandardScaler()
    X_reg = scaler.fit_transform(X_reg)

    best_cv_rmse = np.inf
    best_config = None

    for params in ParameterGrid(reg_grid):
        fold_rmses = []

        for train_idx, valid_idx in cv.split(X_reg, y_reg):
            X_tr, X_va = X_reg[train_idx], X_reg[valid_idx]
            y_tr, y_va = y_reg[train_idx], y_reg[valid_idx]

            model = make_fnn_regressor(
                input_dim=X_reg.shape[1],
                hidden_layers=params["hidden_layers"],
                dropout=params["dropout"],
                learning_rate=params["learning_rate"],
            )

            es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
            model.fit(
                X_tr,
                y_tr,
                validation_data=(X_va, y_va),
                epochs=params["epochs"],
                batch_size=params["batch_size"],
                verbose=0,
                callbacks=[es],
            )

            y_va_pred = model.predict(X_va, verbose=0).reshape(-1)
            rmse = float(np.sqrt(mean_squared_error(y_va, y_va_pred)))
            fold_rmses.append(rmse)

        mean_rmse = float(np.mean(fold_rmses))
        if mean_rmse < best_cv_rmse:
            best_cv_rmse = mean_rmse
            best_config = params

    # Final fit + full-data metrics (for template comparison)
    final_model = make_fnn_regressor(
        input_dim=X_reg.shape[1],
        hidden_layers=best_config["hidden_layers"],
        dropout=best_config["dropout"],
        learning_rate=best_config["learning_rate"],
    )
    final_model.fit(
        X_reg,
        y_reg,
        epochs=best_config["epochs"],
        batch_size=best_config["batch_size"],
        verbose=0,
    )

    y_hat = final_model.predict(X_reg, verbose=0).reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(y_reg, y_hat)))
    mae = float(mean_absolute_error(y_reg, y_hat))
    r2 = float(r2_score(y_reg, y_hat))

    with mlflow.start_run(run_name=f"FNN_REG_{REG_TARGET}_{feature_name}"):
        mlflow.log_param("task", "regression")
        mlflow.log_param("target", REG_TARGET)
        mlflow.log_param("feature_set", feature_name)
        mlflow.log_param("cv_folds", 5)
        for k, v in best_config.items():
            mlflow.log_param(f"best_{k}", str(v))

        mlflow.log_metric("cv_rmse", best_cv_rmse)
        mlflow.log_metric("rmse_full", rmse)
        mlflow.log_metric("mae_full", mae)
        mlflow.log_metric("r2_full", r2)

    reg_results.append(
        {
            "feature_set": feature_name,
            "target": REG_TARGET,
            "cv_rmse": best_cv_rmse,
            "rmse_full": rmse,
            "mae_full": mae,
            "r2_full": r2,
            "best_config": best_config,
        }
    )

pd.DataFrame(reg_results).sort_values("cv_rmse")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4) Classification template cell: CV + grid search + feature comparisons
# MAGIC
# MAGIC Set `CLS_TARGET` to one of:
# MAGIC - `cb1_active` (binary)
# MAGIC - `cb2_active` (binary)
# MAGIC - `selectivity_direction` (multi-class)

# COMMAND ----------

CLS_TARGET = "selectivity_direction"

cls_grid = {
    "hidden_layers": [(128, 64), (256, 128)],
    "dropout": [0.1, 0.3],
    "learning_rate": [1e-3],
    "batch_size": [32, 64],
    "epochs": [60],
}

set_seed(RANDOM_SEED)

if CLS_TARGET in ["cb1_active", "cb2_active"]:
    y_cls_raw = pd.to_numeric(pdf[CLS_TARGET], errors="coerce")
    mask_cls = y_cls_raw.notna()
    y_cls = y_cls_raw.loc[mask_cls].astype(int).to_numpy()
    n_classes = 2
else:
    y_cls_raw = pdf[CLS_TARGET].astype(str).str.strip()
    y_cls_raw = y_cls_raw.replace({"": np.nan, "None": np.nan, "nan": np.nan, "NaN": np.nan})
    mask_cls = y_cls_raw.notna()
    y_clean = y_cls_raw.loc[mask_cls]
    encoder = LabelEncoder()
    y_cls = encoder.fit_transform(y_clean)
    n_classes = len(np.unique(y_cls))

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

cls_results = []

for feature_name, X_full in FEATURES.items():
    X_cls = X_full[mask_cls.to_numpy()]

    scaler = StandardScaler()
    X_cls = scaler.fit_transform(X_cls)

    best_cv_f1 = -np.inf
    best_config = None

    for params in ParameterGrid(cls_grid):
        fold_f1 = []

        for train_idx, valid_idx in cv.split(X_cls, y_cls):
            X_tr, X_va = X_cls[train_idx], X_cls[valid_idx]
            y_tr, y_va = y_cls[train_idx], y_cls[valid_idx]

            model = make_fnn_classifier(
                input_dim=X_cls.shape[1],
                n_classes=n_classes,
                hidden_layers=params["hidden_layers"],
                dropout=params["dropout"],
                learning_rate=params["learning_rate"],
            )

            es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
            model.fit(
                X_tr,
                y_tr,
                validation_data=(X_va, y_va),
                epochs=params["epochs"],
                batch_size=params["batch_size"],
                verbose=0,
                callbacks=[es],
            )

            if n_classes == 2:
                y_va_prob = model.predict(X_va, verbose=0).reshape(-1)
                y_va_pred = (y_va_prob >= 0.5).astype(int)
                f1 = f1_score(y_va, y_va_pred, zero_division=0)
            else:
                y_va_prob = model.predict(X_va, verbose=0)
                y_va_pred = np.argmax(y_va_prob, axis=1)
                f1 = f1_score(y_va, y_va_pred, average="macro", zero_division=0)

            fold_f1.append(float(f1))

        mean_f1 = float(np.mean(fold_f1))
        if mean_f1 > best_cv_f1:
            best_cv_f1 = mean_f1
            best_config = params

    final_model = make_fnn_classifier(
        input_dim=X_cls.shape[1],
        n_classes=n_classes,
        hidden_layers=best_config["hidden_layers"],
        dropout=best_config["dropout"],
        learning_rate=best_config["learning_rate"],
    )
    final_model.fit(
        X_cls,
        y_cls,
        epochs=best_config["epochs"],
        batch_size=best_config["batch_size"],
        verbose=0,
    )

    if n_classes == 2:
        y_prob = final_model.predict(X_cls, verbose=0).reshape(-1)
        y_pred = (y_prob >= 0.5).astype(int)
        auc = float(roc_auc_score(y_cls, y_prob))
        f1_avg = "binary"
    else:
        y_prob = final_model.predict(X_cls, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        auc = np.nan
        f1_avg = "macro"

    acc = float(accuracy_score(y_cls, y_pred))
    prec = float(precision_score(y_cls, y_pred, average=f1_avg, zero_division=0))
    rec = float(recall_score(y_cls, y_pred, average=f1_avg, zero_division=0))
    f1 = float(f1_score(y_cls, y_pred, average=f1_avg, zero_division=0))
    mcc = float(matthews_corrcoef(y_cls, y_pred))

    with mlflow.start_run(run_name=f"FNN_CLS_{CLS_TARGET}_{feature_name}"):
        mlflow.log_param("task", "classification")
        mlflow.log_param("target", CLS_TARGET)
        mlflow.log_param("feature_set", feature_name)
        mlflow.log_param("n_classes", n_classes)
        mlflow.log_param("cv_folds", 5)
        for k, v in best_config.items():
            mlflow.log_param(f"best_{k}", str(v))

        mlflow.log_metric("cv_f1", best_cv_f1)
        mlflow.log_metric("accuracy_full", acc)
        mlflow.log_metric("precision_full", prec)
        mlflow.log_metric("recall_full", rec)
        mlflow.log_metric("f1_full", f1)
        mlflow.log_metric("mcc_full", mcc)
        if not np.isnan(auc):
            mlflow.log_metric("roc_auc_full", auc)

    cls_results.append(
        {
            "feature_set": feature_name,
            "target": CLS_TARGET,
            "cv_f1": best_cv_f1,
            "accuracy_full": acc,
            "precision_full": prec,
            "recall_full": rec,
            "f1_full": f1,
            "mcc_full": mcc,
            "roc_auc_full": auc,
            "best_config": best_config,
        }
    )

pd.DataFrame(cls_results).sort_values("cv_f1", ascending=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5) Notes for assignment write-up
# MAGIC
# MAGIC - Report the **best feature set** by CV metric (RMSE for regression, F1 for classification).
# MAGIC - Include the **best hyperparameters** from grid search.
# MAGIC - Compare at least two targets (e.g., `cb1_p` and `selectivity_direction`).
# MAGIC - Include MLflow screenshots/tables with run IDs, parameters, and metrics.
