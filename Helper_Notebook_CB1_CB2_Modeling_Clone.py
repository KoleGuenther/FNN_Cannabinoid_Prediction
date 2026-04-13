# Databricks notebook source
# MAGIC %md
# MAGIC # Helper Notebook to predict CB1/CB2 Ligand with Machine Learning
# MAGIC Author: Dr. Sylvester Olubou Orimaye, PhD, MPH
# MAGIC Created: 2026-03-01
# MAGIC
# MAGIC This notebook is a **helper code** for building:
# MAGIC - **Regression** models (predicting continuous outcomes like `cb1_p`, `cb2_p`, or `delta_p_cb1_minus_cb2`)
# MAGIC - **Classification** models (predicting labels like `cb1_active`, `cb2_active`, or `selectivity_direction`)
# MAGIC
# MAGIC You will build on the Notebook to complete your subsequent assignments, including adding **additional variables** from the dataset, performing **cross-validation**, **grid search**, and future **deeplearning** tasks.
# MAGIC
# MAGIC Note that I have only used the three popular cheminformatics features encoded as **strings** (JSON arrays):
# MAGIC - `morgan_fp_str` (2048 bits)
# MAGIC - `maccs_fp_str` (166 bits)
# MAGIC - `rdkit_desc50_str` (50 floats)
# MAGIC
# MAGIC However, there are other features, that I have not included in the models. We seem to be doing well with binary classification, just by using these three features, however, **we must improve on the R2 values** for our regression models to acurately predict the **_cb1_p_**, **_cb2_p_**, and the **_delta_p_cb1_minus_cb2_** numerical outcomes.
# MAGIC
# MAGIC In this helper Notebook, I have done the following for you to get started:
# MAGIC 1) Download and load the **cb1_cb2_features_v2.csv** file and convert to a Delta table. Then run the below:
# MAGIC a) Convert JSON strings to Pandas arrays  
# MAGIC b) Build `X` (features) and `y` (target)  
# MAGIC c) Train + evaluate regression and classification models
# MAGIC
# MAGIC You now need to modify the Notebook slightly to fulfil the remaining part of the assignment.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0) Load Data from Delta

# COMMAND ----------

#Change this to the actual cb1_cb2_features_v2 delta table once you create it
TABLE = "workspace.default.cb_1_cb_2_features_v_2"  # <-- remember to change this

df_spark = spark.table(TABLE)
display(df_spark.limit(5))
print("Rows:", df_spark.count())


# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) Select Columns and Convert to pandas

# COMMAND ----------

# For large tables, start with a sample so your driver does not run out of memory.
# You can increase SAMPLE_FRAC gradually, but I fear you will run out your memory. There are about 6000 rows in the table, we are only using 20% of them.
 
SAMPLE_FRAC = 0.35

# Add more features here once you run this helper sucessfully
cols = [
    "molecule_chembl_id",
    "cb1_p","cb2_p","delta_p_cb1_minus_cb2",
    "cb1_active","cb2_active","selectivity_direction",
    "morgan_fp_str","maccs_fp_str","rdkit_desc50_str"
]

pdf = (df_spark.select(*[c for c in cols if c in df_spark.columns])
              .sample(withReplacement=False, fraction=SAMPLE_FRAC, seed=42)
              .toPandas())

#see sample data
print("pandas rows:", len(pdf))
pdf.head()


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) Parse JSON strings into numeric arrays

# COMMAND ----------

import json
import numpy as np
import pandas as pd

def parse_json_array(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    try:
        return json.loads(x)
    except Exception:
        return None

# Parse
pdf["morgan_fp"] = pdf["morgan_fp_str"].apply(parse_json_array)
pdf["maccs_fp"] = pdf["maccs_fp_str"].apply(parse_json_array)
pdf["rdkit_desc50"] = pdf["rdkit_desc50_str"].apply(parse_json_array)

# Drop rows where any feature vector is missing
pdf = pdf.dropna(subset=["morgan_fp","maccs_fp","rdkit_desc50"]).reset_index(drop=True)

# Convert to numpy arrays
X_morgan = np.vstack(pdf["morgan_fp"].apply(lambda v: np.array(v, dtype=np.float32)))
X_maccs  = np.vstack(pdf["maccs_fp"].apply(lambda v: np.array(v, dtype=np.float32)))
X_desc   = np.vstack(pdf["rdkit_desc50"].apply(lambda v: np.array(v, dtype=np.float32)))

print("Shapes:")
print("  X_desc  :", X_desc.shape)   # (n, 50)
print("  X_maccs :", X_maccs.shape)  # (n, 166)
print("  X_morgan:", X_morgan.shape) # (n, 2048)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) Build Feature Matrices

# COMMAND ----------

# DBTITLE 1,I corrected missing values and NaNs.
import numpy as np
import pandas as pd

#We need to correct missing values otherwise, we won't be able to run regression models. Remeber, we cannot plot NaN vs. the Y on the graph.

TARGET = "cb1_p"    #<-- change to cb2_p or delta_p_cb1_minus_cb2



# 1) Force target to numeric; bad values become real NaN
y_raw = pd.to_numeric(pdf[TARGET], errors="coerce")

# 2) Build a mask of valid rows (target not NaN)
mask = y_raw.notna()

print("Target missing count:", (~mask).sum(), "out of", len(mask))

# 3) Filter BOTH X and y using the same mask
pdf2 = pdf.loc[mask].reset_index(drop=True)
y = y_raw.loc[mask].astype(float).reset_index(drop=True)

# 4) Rebuild X from pdf2 (so lengths match)
X_morgan = np.vstack(pdf2["morgan_fp"].apply(lambda v: np.array(v, dtype=np.float32)))
X_maccs  = np.vstack(pdf2["maccs_fp"].apply(lambda v: np.array(v, dtype=np.float32)))
X_desc   = np.vstack(pdf2["rdkit_desc50"].apply(lambda v: np.array(v, dtype=np.float32)))
X_all = np.hstack([X_desc, X_maccs, X_morgan])

# 5) Final sanity check (must be False when there are no more missing or NaN values)
print("y contains NaN?", np.isnan(y.to_numpy()).any())
print("X_all rows:", X_all.shape[0], "y rows:", len(y))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1 - Train/Test Split Strategy
# MAGIC We are going to split the data into training and testing splits 80/20, using seed 42 so our results are reproducible.
# MAGIC - Removed from the cell 13 and moved here to cell 11.
# MAGIC
# MAGIC You will also see 5fold Cross-Validation implemented in order to have a result that is even more trustworthy.
# MAGIC
# MAGIC You will see the grid search in cell 13

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# Random seed can be changed here, used for reproducibility. I always use 42 since its the answer to the ultimate question.
RANDOM_SEED = 42

# ---- Choose feature set
feature_set = "desc"  #<-- change to "desc", "maccs", "morgan", "all"
if feature_set == "desc":
    X = X_desc
elif feature_set == "maccs":
    X = X_maccs
elif feature_set == "morgan":
    X = X_morgan
elif feature_set == "all":
    X = X_all

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_SEED
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train length:", len(y_train))
print("y_test length:", len(y_test))

# Cross Validation

cv = KFold(
    n_splits=5, # <-- Split count, change to 10 for more thorough testing
    shuffle=True,
    random_state=RANDOM_SEED
)

print("Number of CV folds:", cv.get_n_splits())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4) Perform LinearRegression & RandomForest Regression

# COMMAND ----------

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# idk why it does this but this fixes the registry_uri error
mlflow.set_registry_uri("databricks-uc")

# Setup MLFlow experiment
mlflow.set_experiment("/Users/kole.guenther@uhsp.edu/CB1_CB2_Modeling")

descriptor_type = feature_set  #<-- change to "desc", "maccs", "morgan", "all"
target_label = "cb1_p"  #<-- change to "cb1_p" or "cb2_p" or "delta_p_cb1_minus_cb2"

param_grid_lr = {
    "fit_intercept": [True, False]
}

with mlflow.start_run(run_name=f"LinearRegression_{descriptor_type}_{target_label}"):
    # Logging Setup Information
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("descriptor_type", descriptor_type)
    mlflow.log_param("target_label", target_label)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_seed", RANDOM_SEED)
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("grid_search", True)

    lr_grid = GridSearchCV(
        estimator=LinearRegression(),
        param_grid=param_grid_lr,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
        n_jobs=-1
    )
    lr_grid.fit(X_train, y_train)
    best_lr = lr_grid.best_estimator_

    # Predictions
    y_train_pred = best_lr.predict(X_train)
    y_test_pred = best_lr.predict(X_test)

    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    cv_rmse_mean = -lr_grid.best_score_

    # Best Parameters
    mlflow.log_param("best_fit_intercept", lr_grid.best_params_["fit_intercept"])

    # Log Metrics
    mlflow.log_metric("train_rmse", train_rmse)
    mlflow.log_metric("test_rmse", test_rmse)
    mlflow.log_metric("test_mae", test_mae)
    mlflow.log_metric("test_r2", test_r2)
    mlflow.log_metric("cv_rmse_mean", cv_rmse_mean)

    # Log Model
    mlflow.sklearn.log_model(best_lr, artifact_path="linear_regression_model")
    print("\nBest Parameters:", lr_grid.best_params_)
    print("Best CV RMSE:", cv_rmse_mean)
    print("Train RMSE:", train_rmse)
    print("Test RMSE:", test_rmse)
    print("Test MAE :", test_mae)
    print("Test R2  :", test_r2)


# Nonlinear baseline: RandomForest
rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
pred2 = rf.predict(X_test)

rmse2 = mean_squared_error(y_test, pred2, squared=False)
mae2  = mean_absolute_error(y_test, pred2)
r2_2  = r2_score(y_test, pred2)

print("\nRandomForest Regression")
print("  RMSE:", rmse2)
print("  MAE :", mae2)
print("  R2  :", r2_2)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Comparing performance across descriptor types
# MAGIC ![image_1773464257025.png](./image_1773464257025.png "image_1773464257025.png")
# MAGIC ![image_1773464246819.png](./image_1773464246819.png "image_1773464246819.png")
# MAGIC
# MAGIC
# MAGIC From this, we can see that Morgan completely collapses when faced with the CV RMSE score. All also performs pretty poorly, followed by a decent but not great result by the Maccs descriptor type.
# MAGIC
# MAGIC 'Desc' descriptor type has the most stable results with:
# MAGIC - Train RMSE: 0.96
# MAGIC - Test RMSE: 1.06
# MAGIC - CV RMSE: 1.05
# MAGIC
# MAGIC This shows how high dimensional descriptors can absolutely overfit a model, especially with Linear Regression. 
# MAGIC
# MAGIC Desc did a much better job with this as it has a low amount of features at 49, able to be much more stable with training, validation, and with test errors.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Coefficient Interpretation
# MAGIC For 'all' descriptor type:
# MAGIC
# MAGIC Due to 2263 features being used, I expected the magnitude to be small, and that is what we see here. There is such a large amount of features that the predictive weights of them is spread so thin, rather than relying on certain factors that make the biggest difference.
# MAGIC
# MAGIC Most of these coefficients have small magnitudes, meaning that each feature contributes only a small amount to the predicted binding affinity.
# MAGIC
# MAGIC There are both positive and negatives magnitudes here. Positive coefficients indicate that the feature increases predicted binding potency, while negative coefficients show reduced potency.
# MAGIC
# MAGIC I believe that with large, high dimensional features as we have, it is going to be extremely challenging to use all descriptors.

# COMMAND ----------

coef = best_lr.coef_
print("Number of coefficients:", len(coef))
print("First 10 coefficients:", coef[:10])

important_idx = np.argsort(np.abs(coef))[-10:]

print("Top coefficient indices:", important_idx)
print("Top coefficient values:", coef[important_idx])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5) Perform Logistic Regression and RandomForest _Classification_

# COMMAND ----------

# DBTITLE 1,Cleanup and check for NaNs
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

TARGET = "selectivity_direction" # <-- this is the only classification (label) variable. It has three values --> (CB1_selective, CB2_selective, nonselective_or_unknown)

# 1) Clean/select labels (treat empty/None/nan strings as missing)
y_raw = pdf[TARGET].astype(str).str.strip()
y_raw = y_raw.replace({"": np.nan, "None": np.nan, "nan": np.nan, "NaN": np.nan})

mask = y_raw.notna()
pdf2 = pdf.loc[mask].reset_index(drop=True)
y_clean = y_raw.loc[mask].reset_index(drop=True)

# Optional: drop "nonselective_or_unknown" if you only want true selective classes
keep_mask = y_clean.isin(["CB1_selective", "CB2_selective"])
pdf2 = pdf2.loc[keep_mask].reset_index(drop=True)
y_clean = y_clean.loc[keep_mask].reset_index(drop=True)

# 2) Encode labels to ints (recommended)
le = LabelEncoder()
y = le.fit_transform(y_clean)
print("Label mapping:", dict(zip(le.classes_, range(len(le.classes_)))))

# 3) Rebuild X from bracketed comma-separated strings: "[1,0,0,...]"
X_morgan = np.vstack(
    pdf2["morgan_fp_str"].astype(str).str.strip().str[1:-1].str.split(",").apply(
        lambda lst: np.array([x.strip() for x in lst], dtype=np.float32)
    )
)

X_maccs = np.vstack(
    pdf2["maccs_fp_str"].astype(str).str.strip().str[1:-1].str.split(",").apply(
        lambda lst: np.array([x.strip() for x in lst], dtype=np.float32)
    )
)

X_desc = np.vstack(
    pdf2["rdkit_desc50_str"].astype(str).str.strip().str[1:-1].str.split(",").apply(
        lambda lst: np.array([x.strip() for x in lst], dtype=np.float32)
    )
)

X_all = np.hstack([X_desc, X_maccs, X_morgan])

print("Shapes:", X_desc.shape, X_maccs.shape, X_morgan.shape, "=>", X_all.shape)
print("Any NaN in y?", np.isnan(y).any())
print("Any NaN in X_all?", np.isnan(X_all).any())

# COMMAND ----------

# DBTITLE 1,Run the Classification Models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ---- Choose a classification target
#y = pdf["cb1_active"].astype(int)  # <-- change to cb2_active

# Random seed can be changed here, used for reproducibility. I always use 42 since its the answer to the ultimate question.
RANDOM_SEED = 42

# ---- Choose feature set
feature_set = "desc"  #<-- change to "desc", "maccs", "morgan", "all"
if feature_set == "desc":
    X = X_desc
elif feature_set == "maccs":
    X = X_maccs
elif feature_set == "morgan":
    X = X_morgan
elif feature_set == "all":
    X = X_all

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_SEED
)

# Baseline: Logistic Regression
logreg = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler(with_mean=False)),
    ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_SEED))
])

# idk why it does this but this fixes the registry_uri error
mlflow.set_registry_uri("databricks-uc")

# Setup MLFlow experiment
mlflow.set_experiment("/Users/kole.guenther@uhsp.edu/CB1_CB2_Modeling")

with mlflow.start_run(run_name=f"LogisticRegression_{feature_set}_{target_label}"):
    logreg.fit(X_train, y_train)

    pred = logreg.predict(X_test)
    pred_prob = logreg.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred_prob)
    
    # Log Metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("mcc", mcc)
    mlflow.log_metric("roc_auc", roc_auc)
    
    # Log Parameters
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("feature_set", feature_set)
    mlflow.log_param("target_label", target_label)
    mlflow.log_param("random_seed", RANDOM_SEED)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("max_iter", 2000)
    mlflow.log_param("imputer_strategy", "median")
    mlflow.log_param("scaler", "StandardScaler(with_mean=False)")

    # Log Model
    mlflow.sklearn.log_model(logreg, artifact_path="logistic_regression_model")

    # Print Metrics
    print("Logistic Regression")
    print("  Accuracy :", accuracy)
    print("  Precision:", precision)
    print("  Recall   :", recall)
    print("  F1       :", f1)
    print("  MCC      :", mcc)
    print("  ROC-AUC  :", roc_auc)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Regression vs Logistic
# MAGIC ![image_1773465673442.png](./image_1773465673442.png "image_1773465673442.png")
# MAGIC ![image_1773465689276.png](./image_1773465689276.png "image_1773465689276.png")
# MAGIC
# MAGIC Regression is significantly more meaningful, as Logistic regression only shows the difference between Positive and Negative cases, where as Regression actually predicts the values themselves. It keeps the information about how strongly the molecule binds, or I guess the 'magnitude' in a way, whereas the information that comes from a Logistic Regression would just be a "Yes" or "No" type of answer.
# MAGIC
# MAGIC Logistic hides what Regression reveals in that aspect, where Logistic doesn't fully communicate its result. Molecules slightly below the threshold or slightly above look completely different, while molecules within the threshold, even if very different binding, appear the same.
# MAGIC
# MAGIC Classification will appear much stronger than it actually is because it can be very wrong, maybe it predicts a value 50% higher than the binding, but its still considered a positive case, and therefore gets it "correct". This is misleading because its metrics will be artificially boosted due to the introduction of binary classification, rather than checking how close the prediction and actual result are.
# MAGIC
# MAGIC Regardless, we can observe that DESC has the closest ROC-AUC and F1 score, showing how it has the most stable results and least likely to have any type of overfitting.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6) XGBoost Regressor and Classifier

# COMMAND ----------

# Remember to install XGBoost if not available in your environment.
%pip install xgboost
%restart_python

# COMMAND ----------

# DBTITLE 1,XGBoost Regressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb


TARGET = "cb1_p"   # or cb2_p or delta_p_cb1_minus_cb2

# 1) Make y numeric and filter rows with non-missing target
y_raw = pd.to_numeric(pdf[TARGET], errors="coerce")
mask = y_raw.notna()

pdf2 = pdf.loc[mask].reset_index(drop=True)
y = y_raw.loc[mask].astype(np.float32).reset_index(drop=True)

# 2) Parse feature strings from pdf2 (NOT pdf)
X_morgan = np.vstack(
    pdf2["morgan_fp_str"].astype(str).str.strip().str[1:-1].str.split(",").apply(
        lambda lst: np.array([x.strip() for x in lst], dtype=np.float32)
    )
)
X_maccs = np.vstack(
    pdf2["maccs_fp_str"].astype(str).str.strip().str[1:-1].str.split(",").apply(
        lambda lst: np.array([x.strip() for x in lst], dtype=np.float32)
    )
)
X_desc = np.vstack(
    pdf2["rdkit_desc50_str"].astype(str).str.strip().str[1:-1].str.split(",").apply(
        lambda lst: np.array([x.strip() for x in lst], dtype=np.float32)
    )
)

X_all = np.hstack([X_desc, X_maccs, X_morgan])

print("X rows:", X_all.shape[0], "y rows:", len(y))  # must match

# 3) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y.to_numpy(), test_size=0.2, random_state=42
)

# 4) XGBoost regressor
xgb_reg = xgb.XGBRegressor(
    n_estimators=800,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_reg.fit(X_train, y_train)
pred = xgb_reg.predict(X_test)

xgb_rmse = mean_squared_error(y_test, pred, squared=False)
xgb_mae  = mean_absolute_error(y_test, pred)
xgb_r2  = r2_score(y_test, pred)

print("\nXGBoost Regressor")
print("  RMSE:", xgb_rmse)
print("  MAE :", xgb_mae)
print("  R2  :", xgb_r2)

# COMMAND ----------

# DBTITLE 1,XGboost Classification
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np

# ----------- Helper function -----------
def parse_vec_col(series: pd.Series, dtype=np.float32):
    return np.vstack(
        series.astype(str)
              .str.strip()
              .str[1:-1]              # remove brackets
              .str.split(",")
              .apply(lambda lst: np.array([x.strip() for x in lst], dtype=dtype))
    )


TARGET = "selectivity_direction"

# Clean labels
y_raw = pdf[TARGET].astype(str).str.strip()
y_raw = y_raw.replace({"": np.nan, "None": np.nan, "nan": np.nan, "NaN": np.nan})

mask = y_raw.notna()
pdf2 = pdf.loc[mask].reset_index(drop=True)
y_clean = y_raw.loc[mask].reset_index(drop=True)

# Optional: keep only CB1_selective vs CB2_selective (drop nonselective)
keep = y_clean.isin(["CB1_selective", "CB2_selective"])
pdf2 = pdf2.loc[keep].reset_index(drop=True)
y_clean = y_clean.loc[keep].reset_index(drop=True)

# Encode to integers
le = LabelEncoder()
y = le.fit_transform(y_clean)
print("Class mapping:", dict(zip(le.classes_, range(len(le.classes_)))))

# Build X from aligned pdf2
X_morgan = parse_vec_col(pdf2["morgan_fp_str"])
X_maccs  = parse_vec_col(pdf2["maccs_fp_str"])
X_desc   = parse_vec_col(pdf2["rdkit_desc50_str"])
X_all = np.hstack([X_desc, X_maccs, X_morgan])

# ---- Choose feature set
feature_set = "desc"  #<-- change to "desc", "maccs", "morgan", "all"
if feature_set == "desc":
    X = X_desc
elif feature_set == "maccs":
    X = X_maccs
elif feature_set == "morgan":
    X = X_morgan
elif feature_set == "all":
    X = X_all

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2, random_state=42, stratify=y
)

clf = xgb.XGBClassifier(
    n_estimators=900,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=1.0,
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

clf.fit(X_train, y_train)
pred = clf.predict(X_test)

#confusion matrix
print(classification_report(y_test, pred, target_names=le.classes_))

# Weighted results
print("\nXGBoost Classifier")
print("  Accuracy :", accuracy_score(y_test, pred))
print("  Precision:", precision_score(y_test, pred, zero_division=0))
print("  Recall   :", recall_score(y_test, pred, zero_division=0))
print("  F1       :", f1_score(y_test, pred, zero_division=0))
print("  MCC      :", matthews_corrcoef(y_test, pred))

# COMMAND ----------

import mlflow
import mlflow.sklearn
import xgboost as xgb
import pandas as pd

from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# idk why it does this but this fixes the registry_uri error
mlflow.set_registry_uri("databricks-uc")

# Setup MLFlow experiment
mlflow.set_experiment("/Users/kole.guenther@uhsp.edu/CB1_CB2_Modeling")

param_grid = {
    "n_estimators": [300, 500],
    "max_depth": [4, 6],
    "learning_rate": [0.03, 0.05],
    "subsample": [0.8],
    "colsample_bytree": [0.8]
}

xgb_base = xgb.XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

cv = KFold(n_splits=5, shuffle=True, random_state=42)

with mlflow.start_run(run_name=f"XGBoost_{feature_set}_{TARGET}"):

    grid = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)
    best_xgb = grid.best_estimator_

    # Predictions
    train_pred = best_xgb.predict(X_train)
    test_pred = best_xgb.predict(X_test)

    # Train metrics
    train_rmse = mean_squared_error(y_train, train_pred, squared=False)
    train_mae = mean_absolute_error(y_train, train_pred)
    train_r2 = r2_score(y_train, train_pred)

    # Test metrics
    test_rmse = mean_squared_error(y_test, test_pred, squared=False)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)

    # Cross-validation
    cv_scores = cross_val_score(
        best_xgb,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )
    cv_rmse_scores = -cv_scores
    cv_rmse_mean = cv_rmse_scores.mean()
    cv_rmse_std = cv_rmse_scores.std()

    # Overfitting metric
    rmse_gap = test_rmse - train_rmse

    # MLflow logging
    mlflow.log_param("model_type", "XGBoostRegressor")
    mlflow.log_param("target", TARGET)
    mlflow.log_param("feature_dimensionality", int(X_train.shape[1]))
    mlflow.log_param("train_rows", int(X_train.shape[0]))
    mlflow.log_param("test_rows", int(X_test.shape[0]))
    mlflow.log_param("cv_folds", 5)

    mlflow.log_params(grid.best_params_)

    mlflow.log_metric("train_rmse", float(train_rmse))
    mlflow.log_metric("train_mae", float(train_mae))
    mlflow.log_metric("train_r2", float(train_r2))

    mlflow.log_metric("test_rmse", float(test_rmse))
    mlflow.log_metric("test_mae", float(test_mae))
    mlflow.log_metric("test_r2", float(test_r2))

    mlflow.log_metric("cv_rmse_mean", float(cv_rmse_mean))
    mlflow.log_metric("cv_rmse_std", float(cv_rmse_std))
    mlflow.log_metric("rmse_gap_test_minus_train", float(rmse_gap))

    mlflow.sklearn.log_model(best_xgb, "xgboost_regressor_model")

    # Results
    results_df = pd.DataFrame([{
        "target": TARGET,
        "feature_dimensionality": X_train.shape[1],
        "best_params": str(grid.best_params_),
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "cv_rmse_mean": cv_rmse_mean,
        "cv_rmse_std": cv_rmse_std,
        "rmse_gap_test_minus_train": rmse_gap
    }])

    results_df.to_csv("xgboost_part4_results.csv", index=False)
    mlflow.log_artifact("xgboost_part4_results.csv")

    print("Best parameters:", grid.best_params_)
    print("\nXGBoost Regressor Results")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE : {test_rmse:.4f}")
    print(f"Train MAE : {train_mae:.4f}")
    print(f"Test MAE  : {test_mae:.4f}")
    print(f"Train R²  : {train_r2:.4f}")
    print(f"Test R²   : {test_r2:.4f}")
    print(f"CV RMSE Mean: {cv_rmse_mean:.4f}")
    print(f"CV RMSE Std : {cv_rmse_std:.4f}")
    print(f"RMSE Gap (Test - Train): {rmse_gap:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 4/5/6/7
# MAGIC ![image_1773467896360.png](./image_1773467896360.png "image_1773467896360.png")
# MAGIC ### Ensemble vs Linear
# MAGIC From what I can tell, ensemble performs much better than linear regression. This is likely due to the fact that it no longer assumes the relationship between affinity and features is linear. It is almost guaranteed that there is a complex and nonlinear relationship between the molecule descriptors and affinity. Due to this, Linear Regression is definitely limited, so XGBoost can do better here to show a more accurate results when predicting affinity.
# MAGIC
# MAGIC ### Overfitting
# MAGIC Overfitting can be deteced by comparing the training RMSE to the Test RMSE, so here we can just make a statistic that takes the the test rmse and subtracts the training rmse from it. The bigger the gap, the more likely there is some type of overfitting in the model, where it has learned from the training set well, but then struggles to generalize. The gaps we can observe here are between 0.16 and 0.26, which honestly shows that there is likely not much overfitting. This score means that they learned very well from the training set, and also performed pretty well on the testing set. These lower numbers are just much better than what was shown with the Linear Regression, almost by an order of magnitude, making these results very impressive in my opinion.
# MAGIC
# MAGIC ### Sensitivity to high-dimentsional fingerprints
# MAGIC We can actually see a difference where high-dimensional feature sets do better here by capturing more information than the sets with much lower dimensionality. DESC looks like it isnt able to capture as much information from the training set, possibly not being able to understand some of the patterns that the others do. This seems to cause it to overfit, and not being able to generalize as well as the others, hence the higher gap score, and doing better on the train rmse scores too. Ensemble methods like XGBoost also tend to handle high-dimensional spaces better than linear models, which we can see with this experiment too.
# MAGIC
# MAGIC ### Bias-Variance Tradeoff
# MAGIC This XGBoost shows very low bias, as it almost perfectly fits to the training data. This shows that the model is very capable of learning complex relationships, like the ones shown between the molecule features and their affinity. The gap statistic that was made show that there is variance though, where it cannot fully generalize to new molecules it has not seen before. In my opinion, I think this is a lot better than I thought it might do, but this might be caused by my increase in the sampling, doing 0.35 instead of the default 0.2 since I thought my computer could handle it! Overall, i think the ensemble model showcases a low bias, low-moderate variance. I think this increase in variance is well worth switching from linear regression to xgboost in order to take advantage of the much much lower bias.
# MAGIC
# MAGIC ### Part 5
# MAGIC Higher Dimensionality improves RMSE when used with Ensemble models. Although the sweet spot seems to be between MACCS and Morgan (which is a LARGE sweet spot, between 166 and 1024/2048), but both perform equally well.
# MAGIC
# MAGIC Variance does increase across splits, but not very much.
# MAGIC
# MAGIC ![image_1773469146308.png](./image_1773469146308.png "image_1773469146308.png")![image_1773469156348.png](./image_1773469156348.png "image_1773469156348.png")
# MAGIC
# MAGIC Ensemble handles high-dimensional space much better than regression does!
# MAGIC
# MAGIC Overfitting appears in the DESC mostly for the Ensemble method, and with Morgan in Linear Regression.
# MAGIC
# MAGIC ### Part 7
# MAGIC Regression is more informative than classification because it holds much more information in its result. Knowing how far "off" the prediction is from the actual is very valuable. Most error in classification will be seen in the borderline affinity molecules, where in extreme situations, the model could be very wrong, but since its still "active", it is considered correct. Regression measures by how much error there is, so the evaluation isnt inflated or artificially higher like classification is.
# MAGIC
# MAGIC Overfitting mainly appeared in DESC for XGboost, and Morgan in Linear Regression. I talk about it more in their respective sections if you want to see more.
# MAGIC
# MAGIC Descriptor dimensionality was generally "lower = more stable". In the case of ensemble methods, MACCS was more stable than DESC just due to the fact that it can deal much better with higher-dimensionality than Linear Regression can.
# MAGIC
# MAGIC The best model is definitely XGBoost, I think it is able to do the best job recognizing and learning the complex relationships between the features and the molecule affinity, much better than Linear Regression does. Logistic Regression isn't even in the conversation, it is just too easily manipulated by the artificially increased results, and doesn't show how bad it is at its job.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7) Suggested Student Deliverables

# COMMAND ----------

# MAGIC %md
# MAGIC Following the Assignment document on Brightspace for instructions. They include but not limited to the following:
# MAGIC
# MAGIC 1. **Feature comparison**: Which performed best (Descriptors vs MACCS vs Morgan vs Combined) for:
# MAGIC    - Regression
# MAGIC    - Classification  
# MAGIC 2. **Model comparison**: Which model performed best and why?  
# MAGIC 3. Metrics:
# MAGIC    - Regression: RMSE, MAE, R²
# MAGIC    - Classification: Accuracy, Precision, Recall, F1, MCC, AUC  
# MAGIC 4. Your reflection
# MAGIC
