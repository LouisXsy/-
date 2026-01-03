import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
# from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import sys
print(sys.executable)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

EXCEL_PATH = "回归预测.xlsx"

def read_train_test(excel_path: str):
    xl = pd.ExcelFile(excel_path)
    sheets = xl.sheet_names

    if ("训练集" in sheets) and ("测试集" in sheets):
        train = pd.read_excel(excel_path, sheet_name="训练集", header=None)
        test  = pd.read_excel(excel_path, sheet_name="测试集", header=None)
    else:
        train = pd.read_excel(excel_path, sheet_name=0, header=None)
        test  = pd.read_excel(excel_path, sheet_name=1, header=None)
    return train, test

df_train, df_test = read_train_test(EXCEL_PATH)
print("Train shape:", df_train.shape, "Test shape:", df_test.shape)

X_train, y_train = df_train.iloc[:, :-1], df_train.iloc[:, -1]
X_test,  y_test  = df_test.iloc[:, :-1],  df_test.iloc[:, -1]

# 数值标准化 + 类别 OneHot 
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X_train.columns if c not in num_cols]

# OneHotEncoder 参数在不同 sklearn 版本有差异：sparse_output vs sparse
def make_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", make_onehot(), cat_cols),
    ],
    remainder="drop"
)

# 候选集成模型 + CV 调参 
candidates = [
    ("HGBDT", HistGradientBoostingRegressor(random_state=RANDOM_STATE), {
        "model__learning_rate": np.exp(np.linspace(np.log(0.01), np.log(0.2), 50)),
        "model__max_depth": list(range(3, 13)),
        "model__max_leaf_nodes": [31, 63, 127, 255],
        "model__min_samples_leaf": list(range(5, 51)),
        "model__l2_regularization": np.exp(np.linspace(np.log(1e-4), np.log(1.0), 50)),
    }),
    ("GBDT", GradientBoostingRegressor(random_state=RANDOM_STATE), {
        "model__learning_rate": np.exp(np.linspace(np.log(0.01), np.log(0.2), 30)),
        "model__n_estimators":  np.arange(200, 1001, 100),
        "model__max_depth":     [2, 3, 4, 5],
        "model__min_samples_leaf": [5, 10, 20, 30, 40, 50],
        "model__subsample":     [0.7, 0.8, 0.9, 1.0],
        "model__max_features":  [None, "sqrt", "log2"],
    }),
    ("RF", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1), {
        "model__n_estimators":  np.arange(200, 1201, 200),
        "model__max_depth":     [None, 8, 12, 16, 20, 30],
        "model__min_samples_leaf": [1, 2, 5, 10, 20],
        "model__max_features":  ["sqrt", "log2", None],
        "model__bootstrap":     [True, False],
    }),
]
# candidates.append(
#         ("XGBoost",
#          XGBRegressor(
#              objective="reg:squarederror",
#              eval_metric="rmse",
#              random_state=RANDOM_STATE,
#              n_jobs=-1,
#          ),
#          {
#              "model__n_estimators": [300, 600, 900, 1200],
#              "model__max_depth": [2, 3, 4, 5, 6, 8, 10],
#              "model__learning_rate": np.exp(np.linspace(np.log(0.01), np.log(0.3), 40)),
#              "model__subsample": np.linspace(0.6, 1.0, 5),
#              "model__colsample_bytree": np.linspace(0.6, 1.0, 5),
#              "model__min_child_weight": [1, 2, 5, 10],
#              "model__gamma": [0, 0.1, 0.2, 0.5, 1.0],
#              "model__reg_alpha": [0, 1e-4, 1e-3, 1e-2, 0.1, 1.0],
#              "model__reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
#          })
#     )
# candidates.append(
#         ("LightGBM",
#          LGBMRegressor(
#              objective="regression",
#              random_state=RANDOM_STATE,
#              n_jobs=-1,
#              verbosity=-1,
#          ),
#          {
#              "model__n_estimators": [400, 800, 1200, 1600],
#              "model__num_leaves": [15, 31, 63, 127, 255],
#              "model__learning_rate": np.exp(np.linspace(np.log(0.005), np.log(0.2), 40)),
#              "model__subsample": np.linspace(0.6, 1.0, 5),
#              "model__colsample_bytree": np.linspace(0.6, 1.0, 5),
#              "model__min_child_samples": [10, 20, 30, 50, 80],
#              "model__reg_alpha": [0, 1e-4, 1e-3, 1e-2, 0.1, 1.0],
#              "model__reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
#          })
#     )
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

best_name = None
best_search = None
best_cv_rmse = np.inf

for name, model, param_dist in candidates:
    pipe = Pipeline([
        ("prep", preprocess),
        ("model", model),
    ])
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=40,  
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,
    )
    search.fit(X_train, y_train)
    cv_rmse = -search.best_score_
    print(f"\n[{name}] best CV RMSE = {cv_rmse:.6f}")
    print(f"[{name}] best params = {search.best_params_}")

    if cv_rmse < best_cv_rmse:
        best_cv_rmse = cv_rmse
        best_name = name
        best_search = search

print(f"\n==== Selected model by CV: {best_name} (CV RMSE={best_cv_rmse:.6f}) ====")

best_pipe = best_search.best_estimator_

# 测试集评估：SSE + 相对误差 + 误差均值/方差
y_pred = best_pipe.predict(X_test)

residual = (y_test - y_pred).to_numpy()
SSE = float(np.sum(residual ** 2))

# 相对平方误差 RSE：相对于“只预测测试集均值”的基线
y_bar = float(np.mean(y_test))
den = float(np.sum((y_test - y_bar) ** 2))
sqrsum = float(np.sum((y_test ) ** 2))
RSE = float(SSE / (den + 1e-12))   
RSE_sqrsum = float(SSE / (sqrsum + 1e-12)) 
R2_test = 1.0 - RSE

# 平均绝对相对误差
eps = 1e-12
rel_err = np.abs(residual) / (np.abs(y_test.to_numpy()) + eps)
MARE = float(np.mean(rel_err))

MSE = mean_squared_error(y_pred=y_pred,y_true=y_test)

print("\n===== Test Metrics =====")
print(f"SSE = {SSE:.6f}")
print(f"MSE = {MSE:.6f}")
print(f"RSE(相对平方误差，基线=预测均值) = {RSE:.6f}   (对应 R^2_test = {R2_test:.6f})")
print(f"RSE(相对平方误差，基线=预测0) = {RSE_sqrsum:.6f}")
print(f"MARE(平均绝对相对误差) = {MARE:.6f}")

# 报告所有测试样本误差的均值及其方差
err_mean = float(np.mean(residual))
err_var  = float(np.var(residual, ddof=1))  

print("\n===== Error (residual = y - y_hat) summary =====")
print(f"残差均值 mean(e) = {err_mean:.6f}")
print(f"残差方差 var(e)  = {err_var:.6f}")

rel_mean = float(np.mean(rel_err))
rel_var  = float(np.var(rel_err, ddof=1))
print("\n===== Relative error summary (abs(e)/abs(y)) =====")
print(f"相对误差均值 mean(rel) = {rel_mean:.6f}")
print(f"相对误差方差 var(rel)  = {rel_var:.6f}")
