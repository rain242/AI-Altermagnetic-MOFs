"""
Molecular Classification Model - Reproducible Version
分子分类模型 - 可复现版本

This script reproduces the best KNN model (KNN_n2_uniform, seed=260, SMOTE=0.1)
with complete metrics evaluation and predictions.

本脚本复现最佳 KNN 模型 (KNN_n2_uniform, seed=260, SMOTE=0.1)
包含完整的评估指标和预测结果。
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
import joblib
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Create output directory
# 创建输出目录
OUTPUT_DIR = 'classification_reproducible'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("Molecular Classification - Reproducible Pipeline")
print("分子分类 - 可复现流程")
print("=" * 80)

# =============================================================================
# Step 1: Load Data
# 步骤 1: 加载数据
# =============================================================================
print("\n[Step 1/6] Loading Data...")
print("[步骤 1/6] 加载数据...")

train_df = pd.read_csv('01_train.csv')
test_df = pd.read_csv('02_test.csv')

try:
    pre_df = pd.read_csv('03_pre.csv', encoding='gbk')
except:
    pre_df = pd.read_csv('03_pre.csv', encoding='latin-1')

print(f"Training samples: {len(train_df)}")
print(f"测试集分子数：{len(test_df)}")
print(f"预测集分子数：{len(pre_df)}")
print(f"\nClass distribution in training set:")
print(f"训练集类别分布:\n{train_df['alter'].value_counts()}")

# =============================================================================
# Step 2: Feature Engineering
# 步骤 2: 特征工程
# =============================================================================
print("\n[Step 2/6] Feature Engineering...")
print("[步骤 2/6] 特征工程...")

def generate_features(smiles_list):
    """
    Generate molecular fingerprints and descriptors
    生成分子指纹和描述符
    
    Parameters:
    -----------
    smiles_list : list
        List of SMILES strings
        SMILES 字符串列表
    
    Returns:
    --------
    np.ndarray
        Combined feature matrix
        组合特征矩阵
    """
    morgan = []
    maccs = []
    rdkit = []
    
    # Generate fingerprints
    # 生成指纹
    for smi in tqdm(smiles_list, desc="Generating fingerprints/生成指纹", ncols=100, leave=False):
        try:
            mol = Chem.MolFromSmiles(str(smi).strip())
            if mol:
                # Morgan fingerprint (512 bits, radius=2)
                # Morgan 指纹 (512 位，半径=2)
                morgan.append(np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)))
                # MACCS keys (167 bits)
                # MACCS 指纹 (167 位)
                maccs.append(np.array(MACCSkeys.GenMACCSKeys(mol)))
                # RDKit fingerprint (512 bits)
                # RDKit 指纹 (512 位)
                rdkit.append(np.array(Chem.RDKFingerprint(mol, fpSize=512)))
            else:
                morgan.append(np.zeros(512))
                maccs.append(np.zeros(167))
                rdkit.append(np.zeros(512))
        except:
            morgan.append(np.zeros(512))
            maccs.append(np.zeros(167))
            rdkit.append(np.zeros(512))
    
    # Generate descriptors
    # 生成描述符
    desc_names = [desc[0] for desc in Descriptors._descList]
    descs = []
    for smi in tqdm(smiles_list, desc="Generating descriptors/生成描述符", ncols=100, leave=False):
        try:
            mol = Chem.MolFromSmiles(str(smi).strip())
            if mol:
                desc_values = [desc[1](mol) for desc in Descriptors._descList]
                descs.append(desc_values)
            else:
                descs.append([0] * len(desc_names))
        except:
            descs.append([0] * len(desc_names))
    
    # Combine all features
    # 组合所有特征
    return np.hstack([morgan, maccs, rdkit, descs])

# Generate training features
# 生成训练特征
train_smiles = train_df['smiles'].tolist()
train_y = train_df['alter'].values

X_train_full = generate_features(train_smiles)
print(f"\nFeature dimension: {X_train_full.shape[1]}")
print(f"特征维度：{X_train_full.shape[1]}")

# Handle NaN and infinity
# 处理 NaN 和无穷值
X_train_full = np.nan_to_num(X_train_full, nan=0, posinf=0, neginf=0)

# Impute missing values
# 填补缺失值
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train_full)

# Scale features
# 缩放特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)

# =============================================================================
# Step 3: Generate Test and Prediction Features
# 步骤 3: 生成测试集和预测集特征
# =============================================================================
print("\n[Step 3/6] Generating Test and Prediction Features...")
print("[步骤 3/6] 生成测试集和预测集特征...")

X_test_scaled = scaler.transform(imputer.transform(
    np.nan_to_num(generate_features(test_df['smiles'].tolist()), nan=0, posinf=0, neginf=0)
))

X_pre_scaled = scaler.transform(imputer.transform(
    np.nan_to_num(generate_features(pre_df['smiles'].tolist()), nan=0, posinf=0, neginf=0)
))

# =============================================================================
# Step 4: Apply SMOTE and Train Model
# 步骤 4: 应用 SMOTE 并训练模型
# =============================================================================
print("\n[Step 4/6] Applying SMOTE and Training Model...")
print("[步骤 4/6] 应用 SMOTE 并训练模型...")

# Best model parameters from ml_classification_v7.py
# 来自 ml_classification_v7.py 的最佳模型参数
RANDOM_SEED = 260
SMOTE_RATIO = 0.1
KNN_N_NEIGHBORS = 2
KNN_WEIGHTS = 'uniform'

np.random.seed(RANDOM_SEED)

# Apply SMOTE with ratio 0.1
# 应用 SMOTE，采样率 0.1
smote = SMOTE(random_state=RANDOM_SEED, sampling_strategy=SMOTE_RATIO, k_neighbors=min(5, sum(train_y == 1) - 1))
X_balanced, y_balanced = smote.fit_resample(X_train_scaled, train_y)

print(f"After SMOTE - Total samples: {len(y_balanced)}")
print(f"SMOTE 后 - 总样本数：{len(y_balanced)}")
print(f"Class distribution: {np.bincount(y_balanced)}")
print(f"类别分布：{np.bincount(y_balanced)}")

# Split dataset (80/20)
# 划分数据集 (80/20)
X_train, X_val, y_train, y_val = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=RANDOM_SEED, stratify=y_balanced
)

print(f"\nTraining set size: {len(X_train)}")
print(f"训练集大小：{len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"验证集大小：{len(X_val)}")

# Train KNN model with best parameters
# 使用最佳参数训练 KNN 模型
print("\nTraining KNN model...")
print("训练 KNN 模型...")

model = KNeighborsClassifier(
    n_neighbors=KNN_N_NEIGHBORS,
    weights=KNN_WEIGHTS,
    metric='euclidean',
    n_jobs=-1
)

model.fit(X_train, y_train)
print("Model training completed!")
print("模型训练完成!")

# =============================================================================
# Step 5: Evaluate Model
# 步骤 5: 评估模型
# =============================================================================
print("\n[Step 5/6] Evaluating Model...")
print("[步骤 5/6] 评估模型...")

# Predict on validation set
# 在验证集上预测
y_val_pred = model.predict(X_val)
y_val_prob = model.predict_proba(X_val)[:, 1]

# Calculate comprehensive metrics
# 计算综合指标
acc = accuracy_score(y_val, y_val_pred)
prec = precision_score(y_val, y_val_pred, average='weighted')
rec = recall_score(y_val, y_val_pred, average='weighted')
f1 = f1_score(y_val, y_val_pred, average='weighted')

# Confusion matrix
# 混淆矩阵
cm = confusion_matrix(y_val, y_val_pred)

print(f"\nValidation Set Metrics:")
print(f"验证集指标:")
print(f"  Accuracy: {acc:.4f}")
print(f"  Precision (weighted): {prec:.4f}")
print(f"  Recall (weighted): {rec:.4f}")
print(f"  F1 Score (weighted): {f1:.4f}")

print(f"\nConfusion Matrix:")
print(f"混淆矩阵:")
print(cm)

# Create metrics CSV
# 创建指标 CSV
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision (weighted)', 'Recall (weighted)', 'F1 Score (weighted)'],
    'Value': [acc, prec, rec, f1],
    'Threshold': ['>=0.70', '>=0.70', '>=0.70', '>=0.70'],
    'Pass': ['✓' if acc >= 0.70 else '✗', 
             '✓' if prec >= 0.70 else '✗',
             '✓' if rec >= 0.70 else '✗',
             '✓' if f1 >= 0.70 else '✗']
})

metrics_df.to_csv(f'{OUTPUT_DIR}/model_metrics.csv', index=False, encoding='utf-8-sig')
print(f"\nMetrics saved to: {OUTPUT_DIR}/model_metrics.csv")
print(f"指标已保存至：{OUTPUT_DIR}/model_metrics.csv")

# Save detailed classification report
# 保存详细分类报告
class_report = classification_report(y_val, y_val_pred, output_dict=True)
class_report_df = pd.DataFrame(class_report).transpose()
class_report_df.to_csv(f'{OUTPUT_DIR}/classification_report.csv', encoding='utf-8-sig')

# =============================================================================
# Step 6: Make Predictions
# 步骤 6: 进行预测
# =============================================================================
print("\n[Step 6/6] Making Predictions...")
print("[步骤 6/6] 进行预测...")

# Predict test set (02_test.csv)
# 预测测试集 (02_test.csv)
print("\nPredicting test set (02_test.csv)...")
print("预测测试集 (02_test.csv)...")

y_test_pred = model.predict(X_test_scaled)
y_test_prob = model.predict_proba(X_test_scaled)[:, 1]

test_results = pd.DataFrame({
    'molecule_id': test_df['molecule_id'] if 'molecule_id' in test_df.columns else range(1, len(test_df)+1),
    'smiles': test_df['smiles'],
    'predicted_class': y_test_pred,
    'probability_class_1': y_test_prob
})

test_results.to_csv(f'{OUTPUT_DIR}/test_predictions.csv', index=False, encoding='utf-8-sig')
print(f"\nTest predictions saved to: {OUTPUT_DIR}/test_predictions.csv")
print(f"测试集预测已保存至：{OUTPUT_DIR}/test_predictions.csv")

print("\nTest Set Predictions:")
print("测试集预测结果:")
print(test_results.to_string(index=False))


y_pre_pred = model.predict(X_pre_scaled)
y_pre_prob = model.predict_proba(X_pre_scaled)[:, 1]

pre_1_count = sum(y_pre_pred == 1)
pre_0_count = len(y_pre_pred) - pre_1_count
pre_1_ratio = pre_1_count / len(y_pre_pred)

pre_results = pd.DataFrame({
    'smiles': pre_df['smiles'],
    'predicted_class': y_pre_pred,
    'probability_class_1': y_pre_prob
})

pre_results.to_csv(f'{OUTPUT_DIR}/pre_predictions.csv', index=False, encoding='utf-8-sig')
print(f"\nPre set predictions saved to: {OUTPUT_DIR}/pre_predictions.csv")
print(f"预测集预测已保存至：{OUTPUT_DIR}/pre_predictions.csv")

print(f"\nPre Set Statistics:")
print(f"预测集统计:")
print(f"  Total molecules: {len(y_pre_pred)}")
print(f"  总分子数：{len(y_pre_pred)}")
print(f"  Class 0 count: {pre_0_count} ({(1-pre_1_ratio)*100:.2f}%)")
print(f"  类别 0 数量：{pre_0_count} ({(1-pre_1_ratio)*100:.2f}%)")
print(f"  Class 1 count: {pre_1_count} ({pre_1_ratio*100:.2f}%)")
print(f"  类别 1 数量：{pre_1_count} ({pre_1_ratio*100:.2f}%)")

# =============================================================================
# Save Model and Processors
# 保存模型和处理器
# =============================================================================
print("\n[Saving Models and Processors...]")
print("[保存模型和处理器...]")

joblib.dump(model, f'{OUTPUT_DIR}/best_model.joblib')
joblib.dump(scaler, f'{OUTPUT_DIR}/scaler.joblib')
joblib.dump(imputer, f'{OUTPUT_DIR}/imputer.joblib')

print(f"Model saved to: {OUTPUT_DIR}/best_model.joblib")
print(f"模型已保存至：{OUTPUT_DIR}/best_model.joblib")
print(f"Scaler saved to: {OUTPUT_DIR}/scaler.joblib")
print(f"缩放器已保存至：{OUTPUT_DIR}/scaler.joblib")
print(f"Imputer saved to: {OUTPUT_DIR}/imputer.joblib")
print(f"填充器已保存至：{OUTPUT_DIR}/imputer.joblib")

# =============================================================================
# Final Summary
# 最终总结
# =============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("最终总结")
print("=" * 80)


print(f"\nModel Configuration:")
print(f"模型配置:")
print(f"  Model: KNeighborsClassifier")
print(f"  n_neighbors: {KNN_N_NEIGHBORS}")
print(f"  weights: {KNN_WEIGHTS}")
print(f"  random_seed: {RANDOM_SEED}")
print(f"  SMOTE_ratio: {SMOTE_RATIO}")


print(f"\nOutput Files:")
print(f"输出文件:")
print(f"  - {OUTPUT_DIR}/model_metrics.csv")
print(f"  - {OUTPUT_DIR}/test_predictions.csv")
print(f"  - {OUTPUT_DIR}/pre_predictions.csv")
print(f"  - {OUTPUT_DIR}/summary.csv")
print(f"  - {OUTPUT_DIR}/best_model.joblib")
print(f"  - {OUTPUT_DIR}/scaler.joblib")
print(f"  - {OUTPUT_DIR}/imputer.joblib")

print("\n" + "=" * 80)
print("Pipeline Completed Successfully!")
print("流程成功完成!")
print("=" * 80)
