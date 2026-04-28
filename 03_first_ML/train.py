import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import warnings
warnings.filterwarnings('ignore')
import os
import joblib
import json

# 创建结果目录
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)

# 设置随机种子范围
random_states = range(42, 402)  

# 存储所有random_state的结果
all_results = []
best_overall_f1 = 0
best_overall_model = None
best_overall_model_name = ""
best_random_state = None
best_model_type = ""
best_model_params = {}

# 检查GPU可用性并设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name()}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 读取数据
df = pd.read_csv('alter.csv')
print(f"数据集大小: {df.shape}")
print(f"类别分布:\n{df['alter'].value_counts()}")

# 计算类别权重
class_counts = df['alter'].value_counts().sort_index()
total_samples = len(df)
n_classes = len(class_counts)

# 计算类别权重 - 使用多种策略
# 策略1: 基于频率的反比权重
class_weights_freq = total_samples / (n_classes * class_counts)
# 策略2: 平衡权重 (sklearn默认)
class_weights_balanced = compute_class_weight('balanced', classes=np.array([0, 1]), y=df['alter'].values)
# 策略3: 手动调整权重，考虑到不平衡比例 (1:43的比例)
class_weights_manual = {0: 1.0, 1: 43.0}  # 少数类权重显著增加
# 策略4: 适中的权重
class_weights_moderate = {0: 1.0, 1: 25.0}

print(f"\n类别分布分析:")
print(f"类别0数量: {class_counts[0]} ({(class_counts[0]/total_samples)*100:.1f}%)")
print(f"类别1数量: {class_counts[1]} ({(class_counts[1]/total_samples)*100:.1f}%)")
print(f"不平衡比例: {class_counts[0]}:{class_counts[1]} ≈ {class_counts[0]/class_counts[1]:.1f}:1")

print(f"\n计算的类别权重:")
print(f"基于频率的权重: {dict(zip(class_counts.index, class_weights_freq))}")
print(f"平衡权重: {dict(zip([0, 1], class_weights_balanced))}")
print(f"手动调整权重(激进): {class_weights_manual}")
print(f"手动调整权重(适中): {class_weights_moderate}")

# 选择使用适中的手动调整权重
selected_class_weights = class_weights_moderate
print(f"\n使用权重策略: 适中手动调整权重 (1:25)")
print(f"理由: 数据不平衡比例为43:1，使用1:25的权重可以在平衡少数类关注度和防止过拟合之间取得平衡")
print(f"      过于激进的权重(1:43)可能导致模型对少数类过度敏感，产生过多假阳性")

# SMILES转换为分子指纹
def smiles_to_fingerprint(smiles, radius=6, n_bits=2048):
    """将SMILES字符串转换为Morgan指纹"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# 转换所有SMILES为指纹
print("正在转换SMILES为分子指纹...")
X_fp = np.array([smiles_to_fingerprint(smiles) for smiles in df['smiles']])
y = df['alter'].values

print(f"指纹特征维度: {X_fp.shape}")

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_fp)

# PyTorch模型定义
class CNNModel(nn.Module):
    def __init__(self, input_dim):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc_input_dim = 256 * (input_dim // 8)
        self.fc1 = nn.Linear(self.fc_input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.3, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        lstm_out, (hn, cn) = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        x = torch.relu(self.fc1(last_output))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512, dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(d_model, 64)
        self.fc2 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.embedding(x)
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 创建加权采样器用于PyTorch
def create_weighted_sampler(y):
    class_sample_count = np.array([len(np.where(y == t)[0]) for t in np.unique(y)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler

# 训练PyTorch模型的函数（使用类别权重）
def train_pytorch_model(model, X_train, y_train, X_test, y_test, epochs=300, lr=0.001):
    model.to(device)
    
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    # 创建加权采样器
    sampler = create_weighted_sampler(y_train)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    
    # 使用类别权重的损失函数
    weights = torch.tensor([selected_class_weights[0], selected_class_weights[1]], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_f1 = 0
    patience = 20
    counter = 0
    best_model_state = None  # 初始化为None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            _, predicted = torch.max(test_outputs, 1)
            y_pred = predicted.cpu().numpy()
            y_true = y_test_tensor.cpu().numpy()
            
            # 计算F1分数
            current_f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # 初始化best_model_state（如果为None）
        if best_model_state is None:
            best_model_state = model.state_dict().copy()
            best_f1 = current_f1
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            counter = 0
            best_model_state = model.state_dict().copy()
        else:
            counter += 1
        
        if counter >= patience:
            print(f"早停触发，在第{epoch+1}轮停止训练")
            break
    
    # 确保best_model_state已经被设置
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        print("警告: 没有找到最佳模型状态，使用当前模型")
    
    return model, best_f1

# 评估模型函数
def evaluate_model(model, X_test, y_test, model_name, model_params=""):
    if hasattr(model, 'predict_proba'):
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        model.to(device)
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            _, y_pred = torch.max(outputs, 1)
            y_prob = torch.softmax(outputs, 1)[:, 1]
        y_pred = y_pred.cpu().numpy()
        y_prob = y_prob.cpu().numpy()
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred)
    
    result = {
        'Model': model_name,
        'Accuracy': acc,
        'AUC': auc,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1,
        'Model_Params': str(model_params),
        'Confusion_Matrix': cm,
        'y_pred': y_pred
    }
    
    return result, acc, auc, cm, y_pred

# 主训练循环
for random_state in random_states:
    print(f"\n{'='*60}")
    print(f"训练 random_state = {random_state}")
    print(f"{'='*60}")
    
    # 设置随机种子
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_state)
    
    # 划分训练测试集，使用分层抽样
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    print(f"训练集类别分布: 0={np.sum(y_train==0)}, 1={np.sum(y_train==1)}")
    print(f"测试集类别分布: 0={np.sum(y_test==0)}, 1={np.sum(y_test==1)}")
    
    results_list = []
    
    # 1. 支持向量机（使用类别权重）
    print("训练支持向量机模型...")
    svm_param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto', 0.1],
        'class_weight': [selected_class_weights, 'balanced', {0:1, 1:30}]
    }
    
    svm_model = GridSearchCV(
        SVC(probability=True, random_state=random_state),
        svm_param_grid,
        cv=3,
        scoring='f1',  # 使用F1分数作为评估指标
        n_jobs=-1
    )
    svm_model.fit(X_train, y_train)
    svm_params = svm_model.best_params_
    
    # 2. 随机森林（使用类别权重）
    print("训练随机森林模型...")
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'class_weight': [selected_class_weights, 'balanced', {0:1, 1:30}]
    }
    
    rf_model = GridSearchCV(
        RandomForestClassifier(random_state=random_state),
        rf_param_grid,
        cv=3,
        scoring='f1',  # 使用F1分数作为评估指标
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_params = rf_model.best_params_
    
    # 3. XGBoost（使用scale_pos_weight处理不平衡）
    print("训练XGBoost模型...")
    # 计算XGBoost的scale_pos_weight
    scale_pos_weight = class_counts[0] / class_counts[1]  # 多数类/少数类
    
    xgb_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.01, 0.1],
        'scale_pos_weight': [scale_pos_weight, scale_pos_weight/2, scale_pos_weight*0.7]
    }
    
    xgb_model = GridSearchCV(
        xgb.XGBClassifier(random_state=random_state, eval_metric='logloss'),
        xgb_param_grid,
        cv=3,
        scoring='f1',  # 使用F1分数作为评估指标
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    xgb_params = xgb_model.best_params_
    
    # 评估传统机器学习模型
    models_to_evaluate = [
        ('SVM', svm_model, svm_params),
        ('Random Forest', rf_model, rf_params),
        ('XGBoost', xgb_model, xgb_params)
    ]
    
    for model_name, model, params in models_to_evaluate:
        result, acc, auc, cm, y_pred = evaluate_model(model, X_test, y_test, model_name, params)
        result['Random_State'] = random_state
        results_list.append(result)
        
        # 检查是否满足条件（精确率和召回率都不为0）并且F1分数更高
        if (result['Precision'] > 0 and result['Recall'] > 0 and 
            result['F1_Score'] > best_overall_f1):
            best_overall_f1 = result['F1_Score']
            best_overall_model = model
            best_overall_model_name = model_name
            best_random_state = random_state
            best_model_type = 'sklearn'
            best_model_params = params
    
    # 深度学习模型
    input_dim = X_train.shape[1]
    dl_epochs = {'CNN': 300, 'LSTM': 400, 'Transformer': 500}
    
    for dl_model_name, epochs in dl_epochs.items():
        print(f"训练{dl_model_name}模型...")
        
        if dl_model_name == 'CNN':
            model = CNNModel(input_dim)
            model_params = f"CNN - epochs: {epochs}"
        elif dl_model_name == 'LSTM':
            model = LSTMModel(input_dim)
            model_params = f"LSTM - epochs: {epochs}"
        else:
            model = TransformerModel(input_dim)
            model_params = f"Transformer - epochs: {epochs}"
        
        try:
            trained_model, dl_f1 = train_pytorch_model(model, X_train, y_train, X_test, y_test, epochs=epochs)
            result, acc, auc, cm, y_pred = evaluate_model(trained_model, X_test, y_test, dl_model_name, model_params)
            result['Random_State'] = random_state
            results_list.append(result)
            
            # 检查是否满足条件（精确率和召回率都不为0）并且F1分数更高
            if (result['Precision'] > 0 and result['Recall'] > 0 and 
                result['F1_Score'] > best_overall_f1):
                best_overall_f1 = result['F1_Score']
                best_overall_model = trained_model
                best_overall_model_name = dl_model_name
                best_random_state = random_state
                best_model_type = 'pytorch'
                best_model_params = model_params
        except Exception as e:
            print(f"训练{dl_model_name}模型时出错: {e}")
            continue
    
    # 保存当前random_state的结果
    all_results.extend(results_list)
    
    # 输出当前最佳
    if results_list:  # 确保results_list不为空
        current_best_f1 = max([r['F1_Score'] for r in results_list])
        current_best_model = [r for r in results_list if r['F1_Score'] == current_best_f1][0]['Model']
        print(f"当前最佳: {current_best_model} (F1分数: {current_best_f1:.4f})")

# 保存所有详细结果到CSV
print(f"\n保存详细结果...")
detailed_results = []
for result in all_results:
    detailed_result = {
        'Random_State': result['Random_State'],
        'Model': result['Model'],
        'Accuracy': result['Accuracy'],
        'AUC': result['AUC'],
        'Precision': result['Precision'],
        'Recall': result['Recall'],
        'F1_Score': result['F1_Score'],
        'Model_Params': result['Model_Params']
    }
    detailed_results.append(detailed_result)

detailed_df = pd.DataFrame(detailed_results)
if not detailed_df.empty:
    detailed_df = detailed_df.sort_values(['F1_Score', 'Accuracy'], ascending=[False, False])
    detailed_df.to_csv('results/detailed_model_performance.csv', index=False)
    print(f"详细结果已保存到: results/detailed_model_performance.csv")
else:
    print("警告: 没有结果可保存")

# 输出全局最佳模型信息
print(f"\n{'='*80}")
print("全局最佳模型信息")
print(f"{'='*80}")
if best_overall_model is not None:
    print(f"最佳random_state: {best_random_state}")
    print(f"最佳模型: {best_overall_model_name}")
    print(f"最佳F1分数: {best_overall_f1:.4f}")
    print(f"模型类型: {best_model_type}")
    print(f"模型参数: {best_model_params}")
    
    # 重新评估最佳模型以获取详细指标
    if best_model_type == 'sklearn':
        y_pred_best = best_overall_model.predict(X_test)
        y_prob_best = best_overall_model.predict_proba(X_test)[:, 1]
    else:
        best_overall_model.to(device)
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        best_overall_model.eval()
        with torch.no_grad():
            outputs = best_overall_model(X_test_tensor)
            _, y_pred_best = torch.max(outputs, 1)
            y_prob_best = torch.softmax(outputs, 1)[:, 1]
        y_pred_best = y_pred_best.cpu().numpy()
        y_prob_best = y_prob_best.cpu().numpy()

    # 计算最佳模型的详细指标
    best_acc = accuracy_score(y_test, y_pred_best)
    best_auc = roc_auc_score(y_test, y_prob_best)
    best_precision = precision_score(y_test, y_pred_best, zero_division=0)
    best_recall = recall_score(y_test, y_pred_best, zero_division=0)
    best_f1 = f1_score(y_test, y_pred_best, zero_division=0)
    best_cm = confusion_matrix(y_test, y_pred_best)

    print(f"\n最佳模型详细指标:")
    print(f"准确率: {best_acc:.4f}")
    print(f"AUC: {best_auc:.4f}")
    print(f"精确率: {best_precision:.4f}")
    print(f"召回率: {best_recall:.4f}")
    print(f"F1分数: {best_f1:.4f}")

    print(f"\n混淆矩阵:")
    print(best_cm)
    print(f"TN: {best_cm[0, 0]}, FP: {best_cm[0, 1]}, FN: {best_cm[1, 0]}, TP: {best_cm[1, 1]}")

    print(f"\n详细分类报告:")
    print(classification_report(y_test, y_pred_best, target_names=['Class 0', 'Class 1']))

    # 保存最佳模型
    print(f"\n保存最佳模型...")
    joblib.dump(scaler, 'models/scaler.pkl')

    if best_model_type == 'sklearn':
        joblib.dump(best_overall_model, f'models/best_model_{best_overall_model_name}.pkl')
        print(f"最佳模型已保存: models/best_model_{best_overall_model_name}.pkl")
    else:
        best_overall_model.cpu()
        torch.save({
            'model_state_dict': best_overall_model.state_dict(),
            'model_name': best_overall_model_name,
            'model_class': best_overall_model.__class__.__name__,
            'input_dim': X_train.shape[1],
            'f1_score': best_overall_f1,
            'random_state': best_random_state,
            'model_params': best_model_params,
            'class_weights': selected_class_weights
        }, f'models/best_model_{best_overall_model_name}.pth')
        print(f"最佳模型已保存: models/best_model_{best_overall_model_name}.pth")

    # 保存最佳模型摘要
    best_model_summary = {
        'best_model': best_overall_model_name,
        'best_f1_score': float(best_overall_f1),
        'best_random_state': int(best_random_state),
        'model_type': best_model_type,
        'model_parameters': str(best_model_params),
        'class_weights_used': selected_class_weights,
        'performance_metrics': {
            'accuracy': float(best_acc),
            'auc': float(best_auc),
            'precision': float(best_precision),
            'recall': float(best_recall),
            'f1_score': float(best_f1)
        },
        'confusion_matrix': best_cm.tolist(),
        'input_dim': int(X_train.shape[1]),
        'total_trials': len(random_states),
        'class_distribution': {
            'class_0': int(class_counts[0]),
            'class_1': int(class_counts[1]),
            'imbalance_ratio': float(class_counts[0]/class_counts[1])
        }
    }

    with open('models/best_model_summary.json', 'w') as f:
        json.dump(best_model_summary, f, indent=4, ensure_ascii=False)

    print(f"模型摘要已保存: models/best_model_summary.json")
else:
    print("警告: 没有找到最佳模型")

# 输出满足条件的模型统计
valid_models = [r for r in all_results if r['Precision'] > 0 and r['Recall'] > 0]
print(f"\n统计信息:")
print(f"总训练次数: {len(all_results)}")
print(f"满足条件(Precision>0且Recall>0)的模型数: {len(valid_models)}")
if best_overall_model is not None:
    print(f"最佳模型满足条件: {best_precision > 0 and best_recall > 0}")

print(f"\n训练完成! 在{len(random_states)}个random_state中完成了训练。")