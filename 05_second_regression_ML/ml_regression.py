"""
Molecular Regression Model - Comprehensive Pipeline
分子回归模型 - 综合流程

This script implements a comprehensive pipeline for molecular regression task
using various feature engineering methods and machine learning models.

本脚本实现了一个综合的分子回归任务流程，使用多种特征工程方法和机器学习模型。
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error
)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import joblib
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Create output directory
# 创建输出目录
OUTPUT_DIR = 'regression_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("Molecular Regression - Comprehensive Pipeline")
print("分子回归 - 综合流程")
print("=" * 80)

# =============================================================================
# Step 1: Load Data
# 步骤 1: 加载数据
# =============================================================================
print("\n[Step 1/7] Loading Data...")
print("[步骤 1/7] 加载数据...")

train_df = pd.read_csv('01_train.csv')
test_df = pd.read_csv('02_test.csv')

# Filter data: only use alter=1 for regression
# 筛选数据：只使用alter=1的样本进行回归
filtered_train_df = train_df[train_df['alter'] == 1].copy()
filtered_train_df = filtered_train_df.dropna(subset=['split'])

print(f"Original training samples: {len(train_df)}")
print(f"Filtered training samples (alter=1): {len(filtered_train_df)}")
print(f"Test molecules: {len(test_df)}")
print(f"\nSplit value range: min={filtered_train_df['split'].min():.4f}, max={filtered_train_df['split'].max():.4f}, mean={filtered_train_df['split'].mean():.4f}")
print(f"Split值范围：最小值={filtered_train_df['split'].min():.4f}，最大值={filtered_train_df['split'].max():.4f}，平均值={filtered_train_df['split'].mean():.4f}")

# =============================================================================
# Step 2: Feature Engineering
# 步骤 2: 特征工程
# =============================================================================
print("\n[Step 2/7] Feature Engineering...")
print("[步骤 2/7] 特征工程...")

def generate_features(smiles_list, feature_set='all'):
    """
    Generate molecular fingerprints and descriptors
    生成分子指纹和描述符
    
    Parameters:
    -----------
    smiles_list : list
        List of SMILES strings
        SMILES 字符串列表
    feature_set : str
        Feature set to use: 'morgan', 'maccs', 'rdkit', 'descriptors', 'all'
        要使用的特征集：'morgan', 'maccs', 'rdkit', 'descriptors', 'all'
    
    Returns:
    --------
    np.ndarray
        Feature matrix
        特征矩阵
    """
    features = []
    
    if feature_set in ['morgan', 'all']:
        morgan = []
        for smi in tqdm(smiles_list, desc="Generating Morgan fingerprints/生成Morgan指纹", ncols=100, leave=False):
            try:
                mol = Chem.MolFromSmiles(str(smi).strip())
                if mol:
                    morgan.append(np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)))
                else:
                    morgan.append(np.zeros(512))
            except:
                morgan.append(np.zeros(512))
        features.append(np.array(morgan))
    
    if feature_set in ['maccs', 'all']:
        maccs = []
        for smi in tqdm(smiles_list, desc="Generating MACCS keys/生成MACCS指纹", ncols=100, leave=False):
            try:
                mol = Chem.MolFromSmiles(str(smi).strip())
                if mol:
                    maccs.append(np.array(MACCSkeys.GenMACCSKeys(mol)))
                else:
                    maccs.append(np.zeros(167))
            except:
                maccs.append(np.zeros(167))
        features.append(np.array(maccs))
    
    if feature_set in ['rdkit', 'all']:
        rdkit_fp = []
        for smi in tqdm(smiles_list, desc="Generating RDKit fingerprints/生成RDKit指纹", ncols=100, leave=False):
            try:
                mol = Chem.MolFromSmiles(str(smi).strip())
                if mol:
                    rdkit_fp.append(np.array(Chem.RDKFingerprint(mol, fpSize=512)))
                else:
                    rdkit_fp.append(np.zeros(512))
            except:
                rdkit_fp.append(np.zeros(512))
        features.append(np.array(rdkit_fp))
    
    if feature_set in ['descriptors', 'all']:
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
        features.append(np.array(descs))
    
    if features:
        return np.hstack(features)
    else:
        return np.empty((len(smiles_list), 0))

# =============================================================================
# Step 3: Define Models and Parameters
# 步骤 3: 定义模型和参数
# =============================================================================
print("\n[Step 3/7] Defining Models and Parameters...")
print("[步骤 3/7] 定义模型和参数...")

# Define models to try
# 定义要尝试的模型
models = {
    'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
    'KNN': KNeighborsRegressor(n_neighbors=5, weights='uniform'),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
    'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42),
    'CatBoost': CatBoostRegressor(n_estimators=100, random_state=42, verbose=0)
}

# Define feature sets to try
# 定义要尝试的特征集
feature_sets = ['morgan', 'maccs', 'rdkit', 'descriptors', 'all']

# Define random seeds to try (250-400)
# 定义要尝试的随机种子 (250-400)
seeds = [250, 275, 300, 325, 350, 375, 400]

# Define test sizes
# 定义测试集比例
test_sizes = [0.2, 0.3]

# Results storage
# 结果存储
results = []

# =============================================================================
# Step 4: Model Training and Evaluation
# 步骤 4: 模型训练和评估
# =============================================================================
print("\n[Step 4/7] Model Training and Evaluation...")
print("[步骤 4/7] 模型训练和评估...")

total_configs = len(models) * len(feature_sets) * len(seeds) * len(test_sizes)
print(f"Total configurations to test: {total_configs}")
print(f"要测试的配置总数：{total_configs}")

config_count = 0
with tqdm(total=total_configs, desc="Processing configurations/处理配置", ncols=100) as pbar:
    for model_name, model in models.items():
        for feature_set in feature_sets:
            for seed in seeds:
                for test_size in test_sizes:
                    config_count += 1
                    
                    # Set random seed
                    # 设置随机种子
                    np.random.seed(seed)
                    
                    # Generate features
                    # 生成特征
                    train_smiles = filtered_train_df['smiles'].tolist()
                    X_train_full = generate_features(train_smiles, feature_set)
                    y_train_full = filtered_train_df['split'].values
                    
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
                    
                    # Split dataset
                    # 划分数据集
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_train_scaled, y_train_full, test_size=test_size, random_state=seed
                    )
                    
                    # Train model
                    # 训练模型
                    try:
                        model.fit(X_train, y_train)
                        
                        # Predict on validation set
                        # 在验证集上预测
                        y_val_pred = model.predict(X_val)
                        
                        # Calculate metrics
                        # 计算指标
                        r2 = r2_score(y_val, y_val_pred)
                        mae = mean_absolute_error(y_val, y_val_pred)
                        mse = mean_squared_error(y_val, y_val_pred)
                        rmse = np.sqrt(mse)
                        mape = mean_absolute_percentage_error(y_val, y_val_pred)
                        
                        # Generate test features
                        # 生成测试特征
                        test_smiles = test_df['smiles'].tolist()
                        X_test_scaled = scaler.transform(imputer.transform(
                            np.nan_to_num(generate_features(test_smiles, feature_set), nan=0, posinf=0, neginf=0)
                        ))
                        
                        # Predict test set
                        # 预测测试集
                        y_test_pred = model.predict(X_test_scaled)
                        
                        # Check if all test predictions are between 10-60
                        # 检查测试集预测是否都在10-60之间
                        test_in_range = all(10 <= pred <= 60 for pred in y_test_pred)
                        
                        # Check if R2 > 0.7
                        # 检查R2是否大于0.7
                        r2_above_07 = r2 > 0.7
                        
                        # Check if both conditions are met
                        # 检查是否满足两个条件
                        conditions_met = r2_above_07 and test_in_range
                        
                        # Store results
                        # 存储结果
                        results.append({
                            'model': model_name,
                            'feature_set': feature_set,
                            'random_seed': seed,
                            'test_size': test_size,
                            'feature_dimension': X_train_full.shape[1],
                            'r2': r2,
                            'mae': mae,
                            'rmse': rmse,
                            'mape': mape,
                            'test_pred_1': y_test_pred[0],
                            'test_pred_2': y_test_pred[1],
                            'test_pred_3': y_test_pred[2],
                            'test_in_range': test_in_range,
                            'r2_above_07': r2_above_07,
                            'conditions_met': conditions_met
                        })
                        
                        # Save model if conditions are met
                        # 如果满足条件，保存模型
                        if conditions_met:
                            model_dir = os.path.join(OUTPUT_DIR, f"{model_name}_{feature_set}_seed{seed}_test{int(test_size*100)}")
                            os.makedirs(model_dir, exist_ok=True)
                            
                            joblib.dump(model, os.path.join(model_dir, 'model.joblib'))
                            joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
                            joblib.dump(imputer, os.path.join(model_dir, 'imputer.joblib'))
                            
                            # Save test predictions
                            # 保存测试预测
                            test_results = pd.DataFrame({
                                'molecule_id': test_df['molecule_id'],
                                'smiles': test_df['smiles'],
                                'prediction': y_test_pred
                            })
                            test_results.to_csv(os.path.join(model_dir, 'test_predictions.csv'), index=False, encoding='utf-8-sig')
                            
                    except Exception as e:
                        print(f"Error with {model_name}, {feature_set}, seed={seed}, test_size={test_size}: {str(e)}")
                    
                    pbar.update(1)

# =============================================================================
# Step 5: Save Results
# 步骤 5: 保存结果
# =============================================================================
print("\n[Step 5/7] Saving Results...")
print("[步骤 5/7] 保存结果...")

# Create results DataFrame
# 创建结果DataFrame
results_df = pd.DataFrame(results)

# Sort by conditions_met and R2
# 按条件满足情况和R2排序
results_df = results_df.sort_values(['conditions_met', 'r2'], ascending=[False, False])

# Save results to CSV
# 保存结果到CSV
results_df.to_csv(os.path.join(OUTPUT_DIR, 'model_results.csv'), index=False, encoding='utf-8-sig')
print(f"Results saved to: {OUTPUT_DIR}/model_results.csv")
print(f"结果已保存至：{OUTPUT_DIR}/model_results.csv")

# =============================================================================
# Step 6: Find Best Model
# 步骤 6: 找到最佳模型
# =============================================================================
print("\n[Step 6/7] Finding Best Model...")
print("[步骤 6/7] 找到最佳模型...")

# Filter models that meet conditions
# 筛选满足条件的模型
qualified_models = results_df[results_df['conditions_met'] == True]

if len(qualified_models) > 0:
    best_model = qualified_models.iloc[0]
    print("\nBest Qualified Model:")
    print("最佳合格模型：")
    print(f"Model: {best_model['model']}")
    print(f"Feature Set: {best_model['feature_set']}")
    print(f"Random Seed: {best_model['random_seed']}")
    print(f"Test Size: {best_model['test_size']}")
    print(f"R2 Score: {best_model['r2']:.4f}")
    print(f"MAE: {best_model['mae']:.4f}")
    print(f"RMSE: {best_model['rmse']:.4f}")
    print(f"Test Predictions: {best_model['test_pred_1']:.4f}, {best_model['test_pred_2']:.4f}, {best_model['test_pred_3']:.4f}")
    
    # Create best model directory
    # 创建最佳模型目录
    best_model_dir = os.path.join(OUTPUT_DIR, 'best_qualified_model')
    os.makedirs(best_model_dir, exist_ok=True)
    
    # Copy best model files
    # 复制最佳模型文件
    source_dir = os.path.join(OUTPUT_DIR, f"{best_model['model']}_{best_model['feature_set']}_seed{best_model['random_seed']}_test{int(best_model['test_size']*100)}")
    if os.path.exists(source_dir):
        import shutil
        for file in os.listdir(source_dir):
            shutil.copy(os.path.join(source_dir, file), best_model_dir)
        print(f"Best model files copied to: {best_model_dir}")
        print(f"最佳模型文件已复制至：{best_model_dir}")
else:
    print("\nNo model meets the conditions. Trying with additional configurations...")
    print("\n没有模型满足条件。尝试其他配置...")
    
    # Try with different hyperparameters for top models
    # 尝试为顶级模型使用不同的超参数
    print("\nTrying with optimized hyperparameters...")
    print("\n尝试优化超参数...")
    
    # Define optimized models
    # 定义优化模型
    optimized_models = {
        'SVR': [
            SVR(kernel='rbf', C=10.0, gamma='scale'),
            SVR(kernel='rbf', C=1.0, gamma='auto'),
            SVR(kernel='linear', C=1.0)
        ],
        'RandomForest': [
            RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
            RandomForestRegressor(n_estimators=300, max_features='sqrt', random_state=42)
        ],
        'XGBoost': [
            xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42),
            xgb.XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.05, random_state=42)
        ],
        'LightGBM': [
            lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42),
            lgb.LGBMRegressor(n_estimators=300, max_depth=8, learning_rate=0.05, random_state=42)
        ],
        'CatBoost': [
            CatBoostRegressor(n_estimators=200, depth=6, learning_rate=0.1, random_state=42, verbose=0),
            CatBoostRegressor(n_estimators=300, depth=8, learning_rate=0.05, random_state=42, verbose=0)
        ]
    }
    
    # Try optimized models with best feature sets
    # 使用最佳特征集尝试优化模型
    top_feature_sets = results_df['feature_set'].value_counts().index[:3].tolist()
    optimized_results = []
    
    total_optimized = sum(len(models) for models in optimized_models.values()) * len(top_feature_sets) * len(seeds[:3])
    print(f"Total optimized configurations: {total_optimized}")
    
    with tqdm(total=total_optimized, desc="Processing optimized configurations/处理优化配置", ncols=100) as pbar:
        for model_name, model_list in optimized_models.items():
            for i, model in enumerate(model_list):
                for feature_set in top_feature_sets:
                    for seed in seeds[:3]:
                        # Set random seed
                        np.random.seed(seed)
                        
                        # Generate features
                        train_smiles = filtered_train_df['smiles'].tolist()
                        X_train_full = generate_features(train_smiles, feature_set)
                        y_train_full = filtered_train_df['split'].values
                        
                        # Handle NaN and infinity
                        X_train_full = np.nan_to_num(X_train_full, nan=0, posinf=0, neginf=0)
                        
                        # Impute and scale
                        imputer = SimpleImputer(strategy='mean')
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(imputer.fit_transform(X_train_full))
                        
                        # Split dataset
                        X_train, X_val, y_train, y_val = train_test_split(
                            X_train_scaled, y_train_full, test_size=0.2, random_state=seed
                        )
                        
                        try:
                            model.fit(X_train, y_train)
                            y_val_pred = model.predict(X_val)
                            
                            r2 = r2_score(y_val, y_val_pred)
                            
                            # Predict test set
                            test_smiles = test_df['smiles'].tolist()
                            X_test_scaled = scaler.transform(imputer.transform(
                                np.nan_to_num(generate_features(test_smiles, feature_set), nan=0, posinf=0, neginf=0)
                            ))
                            y_test_pred = model.predict(X_test_scaled)
                            
                            test_in_range = all(10 <= pred <= 60 for pred in y_test_pred)
                            r2_above_07 = r2 > 0.7
                            conditions_met = r2_above_07 and test_in_range
                            
                            optimized_results.append({
                                'model': f"{model_name}_opt{i+1}",
                                'feature_set': feature_set,
                                'random_seed': seed,
                                'r2': r2,
                                'test_pred_1': y_test_pred[0],
                                'test_pred_2': y_test_pred[1],
                                'test_pred_3': y_test_pred[2],
                                'conditions_met': conditions_met
                            })
                            
                            # Save if conditions met
                            if conditions_met:
                                model_dir = os.path.join(OUTPUT_DIR, f"{model_name}_opt{i+1}_{feature_set}_seed{seed}")
                                os.makedirs(model_dir, exist_ok=True)
                                
                                joblib.dump(model, os.path.join(model_dir, 'model.joblib'))
                                joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
                                joblib.dump(imputer, os.path.join(model_dir, 'imputer.joblib'))
                                
                                test_results = pd.DataFrame({
                                    'molecule_id': test_df['molecule_id'],
                                    'smiles': test_df['smiles'],
                                    'prediction': y_test_pred
                                })
                                test_results.to_csv(os.path.join(model_dir, 'test_predictions.csv'), index=False, encoding='utf-8-sig')
                                
                        except Exception as e:
                            print(f"Error with optimized {model_name}: {str(e)}")
                        
                        pbar.update(1)
    
    # Save optimized results
    if optimized_results:
        optimized_df = pd.DataFrame(optimized_results)
        optimized_df = optimized_df.sort_values(['conditions_met', 'r2'], ascending=[False, False])
        optimized_df.to_csv(os.path.join(OUTPUT_DIR, 'optimized_results.csv'), index=False, encoding='utf-8-sig')
        print(f"Optimized results saved to: {OUTPUT_DIR}/optimized_results.csv")
        
        # Check if any optimized model meets conditions
        qualified_optimized = optimized_df[optimized_df['conditions_met'] == True]
        if len(qualified_optimized) > 0:
            best_optimized = qualified_optimized.iloc[0]
            print("\nBest Optimized Model:")
            print("最佳优化模型：")
            print(f"Model: {best_optimized['model']}")
            print(f"Feature Set: {best_optimized['feature_set']}")
            print(f"Random Seed: {best_optimized['random_seed']}")
            print(f"R2 Score: {best_optimized['r2']:.4f}")
            print(f"Test Predictions: {best_optimized['test_pred_1']:.4f}, {best_optimized['test_pred_2']:.4f}, {best_optimized['test_pred_3']:.4f}")

# =============================================================================
# Step 7: Summary
# 步骤 7: 总结
# =============================================================================
print("\n[Step 7/7] Summary...")
print("[步骤 7/7] 总结...")

# Load and display top results
# 加载并显示顶级结果
results_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'model_results.csv'))
qualified_count = len(results_df[results_df['conditions_met'] == True])

print(f"\nTotal models tested: {len(results_df)}")
print(f"测试的模型总数：{len(results_df)}")
print(f"Models meeting conditions: {qualified_count}")
print(f"满足条件的模型数：{qualified_count}")

if qualified_count > 0:
    print("\nTop 5 Qualified Models:")
    print("前5个合格模型：")
    top_qualified = results_df[results_df['conditions_met'] == True].head(5)
    for idx, row in top_qualified.iterrows():
        print(f"{idx+1}. {row['model']} ({row['feature_set']}, seed={row['random_seed']})")
        print(f"   R2: {row['r2']:.4f}, Test preds: {row['test_pred_1']:.2f}, {row['test_pred_2']:.2f}, {row['test_pred_3']:.2f}")
else:
    print("\nNo models meet the conditions. Consider adjusting parameters or trying different feature sets.")
    print("\n没有模型满足条件。请考虑调整参数或尝试不同的特征集。")

print("\n" + "=" * 80)
print("Pipeline Completed!")
print("流程完成！")
print("=" * 80)
