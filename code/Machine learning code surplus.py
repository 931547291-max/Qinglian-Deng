# -*- coding: utf-8 -*-
"""
机器学习 + SHAP 完整流程示例
包含模型：LR / RF / SVM / KNN / XGBoost / CatBoost / LightGBM
评价指标：Accuracy, Kappa
解释模型：以 RF 为例做 SHAP 分析（可修改为其他树模型）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, cohen_kappa_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import shap
import os


# ========================== 一、基础配置（只需要改这里） ==========================

# 1. 数据文件路径（改成你自己的）
data_path = "data.csv"

# 2. 特征列名（12 个驱动因子，改成你自己的真实列名）
# 例如：["DEM", "Slope", "Aspect", "Temp", "Precip", "NDVI", "Soil", "NPP", "GDP", "Pop", "Dist_River", "Dist_Road"]
feature_cols = [
    "X1", "X2", "X3", "X4", "X5", "X6",
    "X7", "X8", "X9", "X10", "X11", "X12"
]

# 3. 目标列名（碳储量类别列，改成你自己的）
# 比如 "carbon_class", "C_class" 等
target_col = "carbon_class"

# 4. 随机种子（保证结果可重复）
RANDOM_STATE = 42


# ========================== 二、读数据 & 预处理 ==========================

print(">>> 读取数据...")
data = pd.read_csv(data_path)

# 只保留需要的列，避免混入其他字段
data = data[feature_cols + [target_col]].dropna()

X = data[feature_cols]
y = data[target_col]

# 若 y 不是数值类别（例如 高/中/低），用 LabelEncoder 编成 0,1,2...
le = LabelEncoder()
y_enc = le.fit_transform(y)

print("数据量：", X.shape[0], "样本；", X.shape[1], "个特征")
print("类别编码映射：", dict(zip(le.classes_, le.transform(le.classes_))))

# 划分训练集 / 测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc,
    test_size=0.3,
    random_state=RANDOM_STATE,
    stratify=y_enc
)

# 标准化（只给 LR / SVM / KNN 用）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ========================== 三、训练 7 种模型并计算 Accuracy & Kappa ==========================

results = []

print("\n>>> 训练各模型并计算 Accuracy / Kappa ...")

# 1) Logistic Regression
print("训练 Logistic Regression...")
lr = LogisticRegression(max_iter=1000, multi_class="auto", random_state=RANDOM_STATE)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
acc_lr = accuracy_score(y_test, y_pred_lr)
kappa_lr = cohen_kappa_score(y_test, y_pred_lr)
results.append(("LR", acc_lr, kappa_lr))

# 2) Random Forest
print("训练 Random Forest...")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf.fit(X_train, y_train)  # 树模型用原始数据即可
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
kappa_rf = cohen_kappa_score(y_test, y_pred_rf)
results.append(("RF", acc_rf, kappa_rf))

# 3) SVM
print("训练 SVM...")
svm = SVC(kernel="rbf", C=1.0, probability=True, random_state=RANDOM_STATE)
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)
acc_svm = accuracy_score(y_test, y_pred_svm)
kappa_svm = cohen_kappa_score(y_test, y_pred_svm)
results.append(("SVM", acc_svm, kappa_svm))

# 4) KNN
print("训练 KNN...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
acc_knn = accuracy_score(y_test, y_pred_knn)
kappa_knn = cohen_kappa_score(y_test, y_pred_knn)
results.append(("KNN", acc_knn, kappa_knn))

# 5) XGBoost
print("训练 XGBoost...")
xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=len(np.unique(y_enc)),
    random_state=RANDOM_STATE,
    n_jobs=-1
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
acc_xgb = accuracy_score(y_test, y_pred_xgb)
kappa_xgb = cohen_kappa_score(y_test, y_pred_xgb)
results.append(("XGBoost", acc_xgb, kappa_xgb))

# 6) CatBoost
print("训练 CatBoost...")
cat = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    loss_function="MultiClass",
    random_seed=RANDOM_STATE,
    verbose=False
)
cat.fit(X_train, y_train)
y_pred_cat = cat.predict(X_test)
y_pred_cat = y_pred_cat.reshape(-1).astype(int)  # CatBoost 输出有时是二维
acc_cat = accuracy_score(y_test, y_pred_cat)
kappa_cat = cohen_kappa_score(y_test, y_pred_cat)
results.append(("CatBoost", acc_cat, kappa_cat))

# 7) LightGBM
print("训练 LightGBM...")
lgb = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=-1,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
lgb.fit(X_train, y_train)
y_pred_lgb = lgb.predict(X_test)
acc_lgb = accuracy_score(y_test, y_pred_lgb)
kappa_lgb = cohen_kappa_score(y_test, y_pred_lgb)
results.append(("LightGBM", acc_lgb, kappa_lgb))

# 整理结果表
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Kappa"]).sort_values(
    by="Kappa", ascending=False
)

print("\n================ 模型表现（按 Kappa 排序） ================")
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# 保存结果为表格（可用于论文的 Accuracy / Kappa 表）
results_df.to_csv("model_accuracy_kappa.csv", index=False, encoding="utf-8-sig")
print("\n>>> 模型 Accuracy & Kappa 已保存为：model_accuracy_kappa.csv")


# ========================== 四、选择模型并做 SHAP 分析（以 RF 为例） ==========================

# 你可以改成 results_df 第一名对应的模型。
# 这里默认用 RF（随机森林），和你论文一致。
best_model_name = "RF"
best_model = rf

print(f"\n>>> 使用模型 {best_model_name} 做 SHAP 分析...")

# 对树模型使用 TreeExplainer
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)  # 多分类：list[num_class]，每个 [n_samples, n_features]

# 为了后面的表格，选一个类别做分析（一般选样本最多的类别）
# 这里简单选类别 0，你可以根据需要改：
class_index = 0

shap_values_class = shap_values[class_index]  # [n_samples, n_features]

# 计算每个特征的平均绝对 SHAP 值（该类别）
mean_abs_shap = np.mean(np.abs(shap_values_class), axis=0)  # [n_features]

shap_importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "MeanAbsSHAP": mean_abs_shap
}).sort_values("MeanAbsSHAP", ascending=False)

print("\n================ 特征重要性（按 MeanAbsSHAP 排序） ================")
print(shap_importance_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

# 保存 SHAP 重要性表（可用于论文中“驱动因子贡献度”表）
shap_importance_df.to_csv("shap_feature_importance.csv", index=False, encoding="utf-8-sig")
print("\n>>> SHAP 特征重要性已保存为：shap_feature_importance.csv")


# ========================== 五、绘图（可直接用于论文） ==========================

# 创建输出文件夹
os.makedirs("figures", exist_ok=True)

# 1. SHAP bar 图（整体重要性）
print("\n>>> 绘制 SHAP bar 图（整体重要性）...")
plt.figure()
shap.summary_plot(
    shap_values_class,           # 某个类别的 shap 值
    X_test,
    feature_names=feature_cols,
    plot_type="bar",
    show=False
)
plt.title(f"Feature importance (Mean |SHAP|) - class {class_index}")
plt.tight_layout()
plt.savefig("figures/shap_bar_importance.png", dpi=300, bbox_inches="tight")
plt.close()
print("已保存：figures/shap_bar_importance.png")

# 2. SHAP dot 图（每个样本的贡献，可展示正负影响）
print(">>> 绘制 SHAP dot 图（散点）...")
plt.figure()
shap.summary_plot(
    shap_values_class,
    X_test,
    feature_names=feature_cols,
    show=False
)
plt.title(f"SHAP summary plot - class {class_index}")
plt.tight_layout()
plt.savefig("figures/shap_dot_summary.png", dpi=300, bbox_inches="tight")
plt.close()
print("已保存：figures/shap_dot_summary.png")

# 3. SHAP dependence plot（以最重要的一个因子为例）
top_feature = shap_importance_df["Feature"].iloc[0]
print(f">>> 绘制 SHAP dependence 图（最重要特征：{top_feature}）...")

plt.figure()
shap.dependence_plot(
    top_feature,
    shap_values_class,
    X_test,
    show=False
)
plt.title(f"SHAP dependence plot - {top_feature}")
plt.tight_layout()
plt.savefig(f"figures/shap_dependence_{top_feature}.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"已保存：figures/shap_dependence_{top_feature}.png")

print("\n>>> 全部分析完成。结果文件：")
print("    - model_accuracy_kappa.csv：7 个模型的 Accuracy & Kappa")
print("    - shap_feature_importance.csv：特征平均 SHAP 值表")
print("    - figures/shap_bar_importance.png：SHAP 条形图")
print("    - figures/shap_dot_summary.png：SHAP 散点图")
print(f"    - figures/shap_dependence_{top_feature}.png：最重要特征的 SHAP 依赖图")
