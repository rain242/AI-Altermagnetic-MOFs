"""
SHAP Analysis for CatBoost Regression Model - Top 10 Features
使用全部训练数据进行 CatBoost 回归模型的 SHAP 分析（前 10 个特征）

This script performs SHAP analysis on the CatBoost regression model using all training samples.
Features: MACCS fingerprints (167 bits)
Model: CatBoost Regressor (R2=0.956675, seed=350, test_size=0.2)

Generates:
1. Top 10 features summary bar plot
2. Top 10 features impact direction plot
3. Feature correlation heatmap
4. ALL molecules containing each of the top 10 features (standard coloring: N=blue, O=red, etc.)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from catboost import CatBoostRegressor
import shap
import multiprocessing
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set number of CPU cores for parallel computation
NUM_CPU_CORES = 12
print(f"Using {NUM_CPU_CORES} CPU cores for parallel computation")

# =============================================================================
# Configuration
# =============================================================================

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

MACARON_PURPLE = '#C3B1E1'
COLOR_POSITIVE = (253/255, 192/255, 247/255)
COLOR_NEGATIVE = (192/255, 226/255, 253/255)

# Create output directories
base_dir = 'shap_analysis_catboost'
output_dir = os.path.join(base_dir, 'visualization_output')
structure_dir = os.path.join(output_dir, 'feature_structures')

os.makedirs(output_dir, exist_ok=True)
os.makedirs(structure_dir, exist_ok=True)

print("=" * 80)
print("SHAP Analysis for CatBoost Regression Model - Top 10 Features")
print("CatBoost 回归模型的 SHAP 分析 - 前 10 个特征")
print("=" * 80)

# =============================================================================
# Step 1: Load Data
# =============================================================================

print("\n[Step 1/5] Loading Data...")
print("[步骤 1/5] 加载数据...")

train_df = pd.read_csv('../01_train.csv')
test_df = pd.read_csv('../02_test.csv')

filtered_train_df = train_df[train_df['alter'] == 1].copy()
filtered_train_df = filtered_train_df.dropna(subset=['split'])

print(f"Filtered training samples (alter=1): {len(filtered_train_df)}")
print(f"Test molecules: {len(test_df)}")
print(f"Split value range: min={filtered_train_df['split'].min():.4f}, max={filtered_train_df['split'].max():.4f}, mean={filtered_train_df['split'].mean():.4f}")

# =============================================================================
# Step 2: Feature Engineering
# =============================================================================

print("\n[Step 2/5] Generating Features...")
print("[步骤 2/5] 生成特征...")

def generate_features(smiles_list, feature_set='maccs'):
    features = []
    if feature_set in ['maccs', 'all']:
        maccs = []
        for smi in tqdm(smiles_list, desc="Generating MACCS/生成 MACCS 指纹", ncols=100, leave=False):
            try:
                mol = Chem.MolFromSmiles(str(smi).strip())
                if mol:
                    maccs.append(np.array(MACCSkeys.GenMACCSKeys(mol)))
                else:
                    maccs.append(np.zeros(167))
            except:
                maccs.append(np.zeros(167))
        features.append(np.array(maccs))
    return np.hstack(features) if features else np.empty((len(smiles_list), 0))

def get_feature_names(feature_set='maccs'):
    names = []
    if feature_set in ['maccs', 'all']:
        names.extend([f'MACCS_{i}' for i in range(167)])
    return names

feature_set = 'maccs'
feature_names = get_feature_names(feature_set)

train_smiles = filtered_train_df['smiles'].tolist()
train_y = filtered_train_df['split'].values

print("\nGenerating training features...")
X_train_full = generate_features(train_smiles, feature_set)
print(f"Feature matrix shape: {X_train_full.shape}")

X_train_full = np.nan_to_num(X_train_full, nan=0, posinf=0, neginf=0)

imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train_full)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)

# =============================================================================
# Step 3: Load CatBoost Model
# =============================================================================

print("\n[Step 3/5] Loading CatBoost Model...")
print("[步骤 3/5] 加载 CatBoost 模型...")

best_model_dir = '../regression_results/best_qualified_model'
model = joblib.load(os.path.join(best_model_dir, 'model.joblib'))
scaler = joblib.load(os.path.join(best_model_dir, 'scaler.joblib'))
imputer = joblib.load(os.path.join(best_model_dir, 'imputer.joblib'))

print("Model loaded successfully!")
print("模型加载成功!")

# =============================================================================
# Step 4: SHAP Analysis
# =============================================================================

print("\n[Step 4/5] Performing SHAP Analysis...")
print("[步骤 4/5] 执行 SHAP 分析...")

background_size = len(X_train_scaled)
background_data = X_train_scaled[np.arange(background_size)]

print(f"\nUsing ALL training samples for SHAP analysis: {background_size} samples")
print(f"使用全部训练样本进行 SHAP 分析：{background_size} 个样本")
print(f"Using {NUM_CPU_CORES} CPU cores for parallel SHAP computation")
print(f"使用 {NUM_CPU_CORES} 个 CPU 核心进行并行 SHAP 计算")

print("\nCreating SamplingExplainer...")
print("创建 SamplingExplainer...")
explainer = shap.SamplingExplainer(model.predict, background_data)

print(f"\nComputing SHAP values for ALL {len(X_train_scaled)} training samples (parallel with {NUM_CPU_CORES} cores)...")
print(f"计算全部 {len(X_train_scaled)} 个训练样本的 SHAP 值（并行使用 {NUM_CPU_CORES} 个核心）...")

shap_values_train = explainer.shap_values(X_train_scaled, n_jobs=NUM_CPU_CORES)

print(f"Training SHAP values shape: {shap_values_train.shape}")

# =============================================================================
# Step 5: Identify Top 10 Features and Generate Visualizations
# =============================================================================

print("\n[Step 5/5] Identifying Top 10 Features and Generating Visualizations...")
print("[步骤 5/5] 识别前 10 个特征并生成可视化...")

mean_abs_shap = np.mean(np.abs(shap_values_train), axis=0)
top10_indices = np.argsort(mean_abs_shap)[-10:][::-1]
top10_features = [feature_names[i] for i in top10_indices]
top10_shap_train = shap_values_train[:, top10_indices]

print(f"\nTop 10 features identified!")
print(f"已识别前 10 个特征!")
for i, feat in enumerate(top10_features):
    print(f"  {i+1}. {feat}: {mean_abs_shap[top10_indices[i]]:.6f}")

stats_data = []
for i, idx in enumerate(top10_indices):
    shap_vals = shap_values_train[:, idx]
    stats_data.append({
        'feature': feature_names[idx],
        'mean_abs_shap': np.mean(np.abs(shap_vals)),
        'mean_shap': np.mean(shap_vals),
        'std_shap': np.std(shap_vals),
    })

stats_df = pd.DataFrame(stats_data)
stats_df = stats_df.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

stats_df.to_csv(os.path.join(output_dir, 'top10_shap_statistics.csv'), index=False)
print(f"\nStatistics saved to: top10_shap_statistics.csv")

# =============================================================================
# Generate Main Visualizations
# =============================================================================

print("\n" + "=" * 80)
print("Generating Visualizations")
print("生成可视化")
print("=" * 80)

# Figure 1: Summary Bar Plot
print("\n  [1/3] Creating Figure 1: Summary Bar Plot...")
print("  [1/3] 创建图 1：摘要条形图...")

fig, ax = plt.subplots(figsize=(14, 10))
y_pos = np.arange(len(stats_df))
mean_abs_values = stats_df['mean_abs_shap'].values

bars = ax.barh(y_pos, mean_abs_values, color=MACARON_PURPLE, edgecolor='white',
               linewidth=2, alpha=0.9, height=0.85)

ax.set_yticks(y_pos)
ax.set_yticklabels(stats_df['feature'], fontsize=16, fontweight='bold', fontfamily='Arial')
ax.invert_yaxis()
ax.set_xlabel('Mean |SHAP Value|', fontsize=16, fontweight='bold', labelpad=15, fontfamily='Arial')
ax.set_title('Figure 1 | Top 10 Most Important Features', fontsize=20, fontweight='bold', pad=20, fontfamily='Arial')

for i, (bar, val) in enumerate(zip(bars, mean_abs_values)):
    ax.text(val + 0.0001, bar.get_y() + bar.get_height()/2, f'{val:.6f}',
            va='center', ha='left', fontsize=12, fontweight='bold', color='#444444', fontfamily='Arial')

ax.grid(axis='x', alpha=0.2, linestyle='--', linewidth=1, color='#CCCCCC')
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure1_summary_bar_plot.png'), dpi=600, bbox_inches='tight', facecolor='white')
plt.close()
print("    ✓ Saved: figure1_summary_bar_plot.png")

# Figure 2: Impact Direction Plot
print("\n  [2/3] Creating Figure 2: Impact Direction Plot...")
print("  [2/3] 创建图 2：影响方向图...")

fig, ax = plt.subplots(figsize=(14, 10))
y_pos = np.arange(len(stats_df))
mean_shap_values = stats_df['mean_shap'].values

colors_dir = [COLOR_POSITIVE if val > 0 else COLOR_NEGATIVE for val in mean_shap_values]

bars = ax.barh(y_pos, mean_shap_values, color=colors_dir, edgecolor='white', linewidth=2, alpha=0.9, height=0.85)
ax.axvline(x=0, color='gray', linestyle='--', linewidth=2, alpha=0.6)

ax.set_yticks(y_pos)
ax.set_yticklabels(stats_df['feature'], fontsize=16, fontweight='bold', fontfamily='Arial')
ax.invert_yaxis()
ax.set_xlabel('Mean SHAP Value', fontsize=16, fontweight='bold', labelpad=15, fontfamily='Arial')
ax.set_title('Figure 2 | Feature Impact Direction', fontsize=20, fontweight='bold', pad=20, fontfamily='Arial')

for i, (bar, val) in enumerate(zip(bars, mean_shap_values)):
    if val > 0:
        ax.text(val + 0.00015, bar.get_y() + bar.get_height()/2, f'{val:.6f}',
                va='center', ha='left', fontsize=12, fontweight='bold', color='#444444', fontfamily='Arial')
    else:
        ax.text(val - 0.00015, bar.get_y() + bar.get_height()/2, f'{val:.6f}',
                va='center', ha='right', fontsize=12, fontweight='bold', color='#444444', fontfamily='Arial')

ax.grid(axis='x', alpha=0.2, linestyle='--', linewidth=1, color='#CCCCCC')
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure2_impact_direction.png'), dpi=600, bbox_inches='tight', facecolor='white')
plt.close()
print("    ✓ Saved: figure2_impact_direction.png")

# Figure 3: Feature Correlation Heatmap
print("\n  [3/3] Creating Figure 3: Feature Correlation Heatmap...")
print("  [3/3] 创建图 3：特征相关性热图...")

top10_shap_df = pd.DataFrame(top10_shap_train, columns=top10_features)
corr_matrix = top10_shap_df.corr(method='pearson')

# Create RF_01 to RF_10 labels
rf_labels = [f'RF_{i:02d}' for i in range(1, 11)]

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True,
            linewidths=2, linecolor='white', cbar_kws={'shrink': 0.8, 'pad': 0.02},
            annot_kws={'size': 10, 'weight': 'bold'}, ax=ax, vmin=-1, vmax=1)

ax.set_xticklabels(rf_labels, rotation=45, ha='right', fontsize=12, fontweight='bold', fontfamily='Arial')
ax.set_yticklabels(rf_labels, rotation=0, fontsize=12, fontweight='bold', fontfamily='Arial')
ax.set_title('Figure 3 | Feature Correlation Analysis', fontsize=20, fontweight='bold', pad=20, fontfamily='Arial')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure3_correlation_heatmap.png'), dpi=600, bbox_inches='tight', facecolor='white')
plt.close()
print("    ✓ Saved: figure3_correlation_heatmap.png")

# =============================================================================
# Generate ALL Molecules for Each Feature (Standard Coloring)
# =============================================================================

print("\n" + "=" * 80)
print("Generating ALL Molecules for Each Feature (Standard Coloring: N=Blue, O=Red, etc.)")
print("为每个特征生成所有分子结构（标准配色：N=蓝，O=红，等）")
print("=" * 80)

def find_all_molecules_with_maccs_feature(smiles_list, bit_idx):
    """Find ALL molecules that have the specified MACCS key active"""
    molecules = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(str(smi).strip())
            if not mol:
                continue
            maccs = MACCSkeys.GenMACCSKeys(mol)
            if maccs[bit_idx]:
                molecules.append((mol, smi))
        except:
            continue
    return molecules

def draw_molecule_standard(mol, smi, filename):
    """Draw a molecule with standard RDKit coloring (N=blue, O=red, etc.)"""
    try:
        # Generate 2D coordinates
        mol_2d = Chem.Mol(mol)
        Chem.rdDepictor.Compute2DCoords(mol_2d)
        
        # Use RDKit's standard drawing function with colored atoms
        # RDKit automatically colors atoms: N=blue, O=red, S=yellow, etc.
        drawer = rdMolDraw2D.MolDraw2DCairo(600, 450)
        draw_opts = drawer.drawOptions()
        draw_opts.bondLineWidth = 2
        
        drawer.DrawMolecule(mol_2d)
        drawer.FinishDrawing()
        
        filepath = os.path.join(structure_dir, filename)
        with open(filepath, 'wb') as f:
            f.write(drawer.GetDrawingText())
        
        return filename
        
    except Exception as e:
        print(f"  Error drawing molecule: {str(e)}")
        return None

structure_doc = []

print("\n  Finding and drawing ALL molecules for each feature...")
print("  为每个特征寻找并绘制所有分子...")

for i, feature in enumerate(stats_df['feature']):
    feature_type = feature.split('_')[0] if '_' in feature else 'Descriptor'
    bit_idx = int(feature.split('_')[1]) if '_' in feature else None
    
    structure_info = {
        'feature': feature,
        'type': feature_type,
        'bit_index': bit_idx,
        'total_molecules': 0,
        'molecule_files': []
    }
    
    if feature_type == 'MACCS' and bit_idx is not None:
        # Find ALL molecules with this MACCS key
        all_molecules = find_all_molecules_with_maccs_feature(train_smiles, bit_idx)
        
        if all_molecules:
            structure_info['total_molecules'] = len(all_molecules)
            print(f"\n  Feature {i+1}: {feature} - Found {len(all_molecules)} molecules")
            
            # Draw each molecule with standard coloring
            for mol_idx, (mol, smi) in enumerate(all_molecules):
                filename = f"feature_{i+1:02d}_{feature}_mol{mol_idx+1:03d}.png"
                drawn_file = draw_molecule_standard(mol, smi, filename)
                
                if drawn_file:
                    structure_info['molecule_files'].append({
                        'filename': drawn_file,
                        'smiles': smi
                    })
                    print(f"    ✓ Drew molecule {mol_idx+1}/{len(all_molecules)}")
        else:
            structure_info['total_molecules'] = 0
            print(f"\n  Feature {i+1}: {feature} - No molecules found")
    
    structure_doc.append(structure_info)

# Save structure information
structure_df = pd.DataFrame([{
    'feature': item['feature'],
    'type': item['type'],
    'bit_index': item['bit_index'],
    'total_molecules': item['total_molecules'],
    'molecule_count': len(item['molecule_files'])
} for item in structure_doc])
structure_df.to_csv(os.path.join(structure_dir, 'feature_molecules_summary.csv'), index=False)

print("\n    ✓ Saved: feature_molecules_summary.csv")

# Create summary visualization for each feature (grid of all molecules)
print("\n  Creating summary grids for each feature...")
print("  为每个特征创建汇总网格图...")

for i, struct_info in enumerate(structure_doc):
    if struct_info['total_molecules'] > 0 and len(struct_info['molecule_files']) > 0:
        n_mols = len(struct_info['molecule_files'])
        
        # Calculate grid size
        n_cols = min(5, n_mols)
        n_rows = (n_mols + n_cols - 1) // n_cols if n_cols > 0 else 1
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        if n_mols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        axes_flat = axes.flatten()
        
        for mol_idx, mol_info in enumerate(struct_info['molecule_files']):
            ax = axes_flat[mol_idx]
            img_path = os.path.join(structure_dir, mol_info['filename'])
            
            if os.path.exists(img_path):
                img = plt.imread(img_path)
                ax.imshow(img)
                ax.set_title(f"Molecule {mol_idx+1}\n{mol_info['smiles'][:50]}...",
                            fontsize=8, fontweight='bold', fontfamily='Arial', pad=5)
                ax.axis('off')
            else:
                ax.axis('off')
        
        # Hide empty subplots
        for j in range(n_mols, len(axes_flat)):
            axes_flat[j].axis('off')
        
        plt.suptitle(f'Feature {i+1}: {struct_info["feature"]}\nTotal Molecules: {struct_info["total_molecules"]}',
                     fontsize=14, fontweight='bold', fontfamily='Arial', y=1.02)
        plt.tight_layout()
        
        summary_file = os.path.join(structure_dir, f'feature_{i+1:02d}_{struct_info["feature"]}_all_molecules.png')
        plt.savefig(summary_file, dpi=600, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"    ✓ Saved: feature_{i+1:02d}_{struct_info['feature']}_all_molecules.png ({n_mols} molecules)")

print("\n" + "=" * 80)
print("✅ SHAP Analysis Successfully Generated!")
print("✅ SHAP 分析成功生成!")
print("=" * 80)
print(f"\nOutput directory: {output_dir}")
print(f"\nGenerated files:")
print("• 3 main figures (Figure 1-3)")
print(f"• All molecules for each of the top 10 features (standard coloring)")
print("• Statistics CSV file")
print("=" * 80)
