import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
import os
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import random

# ==============================================================================
# CẤU HÌNH REPRODUCIBILITY
# ==============================================================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==============================================================================
# 1. CẤU HÌNH MÔ HÌNH MF-MGCN (OPTIMAL VERSION K=16)
# ==============================================================================
class MF_MGCN(nn.Module):
    def __init__(self, num_nodes=19, num_bands=5):
        super(MF_MGCN, self).__init__()
        
        #  Bài báo nói K=16 cho kết quả tốt nhất (Section 5.4)
        # Mặc dù Table 1 ghi 32, nhưng thực nghiệm của họ chọn 16.
        K_HIDDEN = 16 
        
        # --- LAYER 1: FUNCTIONAL CONNECTIVITY (Riêng biệt từng band) ---
        # Input: 1 feature (DE value) -> Output: K features
        self.conv1_layers = nn.ModuleList([GCNConv(1, K_HIDDEN) for _ in range(num_bands)])
        self.bn1_layers = nn.ModuleList([nn.BatchNorm1d(K_HIDDEN) for _ in range(num_bands)])

        # --- LAYER 2: STRUCTURAL CONNECTIVITY (Chia sẻ không gian) ---
        # Input: K -> Output: 2 features 
        self.conv2_layers = nn.ModuleList([GCNConv(K_HIDDEN, 2) for _ in range(num_bands)])
        self.bn2_layers = nn.ModuleList([nn.BatchNorm1d(2) for _ in range(num_bands)])

        # --- FULLY CONNECTED LAYERS ---
        # Flatten: 19 nodes * 2 features * 5 bands = 190
        self.flatten_dim = num_nodes * 2 * num_bands 
        
        # S = 128, F = 32 [cite: 599]
        self.lin1 = nn.Linear(self.flatten_dim, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.lin2 = nn.Linear(128, 32)
        self.lin3 = nn.Linear(32, 2) 

        # Khởi tạo trọng số (Kaiming Init)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, data, use_dropout=True):
        x_all = data.x # Shape: [Total_Nodes_in_Batch, 5]
        batch_size = data.num_graphs
        band_outputs = []
        
        edge_indices = [
            data.edge_index_b0, data.edge_index_b1, data.edge_index_b2, 
            data.edge_index_b3, data.edge_index_b4
        ]
        edge_weights = [
            data.edge_weight_b0, data.edge_weight_b1, data.edge_weight_b2, 
            data.edge_weight_b3, data.edge_weight_b4
        ]
        
        for i in range(5):
            # 1. Feature Selection
            x_band = x_all[:, i].unsqueeze(-1) # [Nodes, 1]
            
            # 2. Layer 1: Functional GCN
            x = self.conv1_layers[i](x_band, edge_indices[i], edge_weights[i])
            x = self.bn1_layers[i](x)
            x = F.relu(x)
            if use_dropout: x = F.dropout(x, p=0.3, training=self.training)

            # 3. Layer 2: Structural GCN
            x = self.conv2_layers[i](x, data.edge_index_struct)
            x = self.bn2_layers[i](x)
            x = F.relu(x)
            if use_dropout: x = F.dropout(x, p=0.3, training=self.training)
            
            # 4. Reshape
            x_reshaped = x.view(batch_size, -1) 
            band_outputs.append(x_reshaped)
            
        # --- FLATTEN & CONCATENATE ---
        x_concat = torch.cat(band_outputs, dim=1)
        
        # --- CLASSIFICATION HEAD ---
        x = self.lin1(x_concat)
        x = self.bn3(x)
        x = F.relu(x)
        # Dropout ở FC layer thường cao hơn (0.5) [cite: 599] (implicitly implied by standard practices)
        if use_dropout: x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.lin2(x)
        x = F.relu(x)
        
        x = self.lin3(x)
        return x

# ==============================================================================
# 2. DATA LOADER (GIỮ NGUYÊN BẢN ROBUST)
# ==============================================================================
def load_raw_data_dict():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
    
    # --- Load nhãn ---
    tsv_path = os.path.join(BASE_DIR, "participants.tsv")
    try: df = pd.read_csv(tsv_path, sep='\t')
    except: df = pd.read_csv(tsv_path, sep='\s+')
    
    sub_label_map = {}
    for _, row in df.iterrows():
        sub = row['participant_id']
        grp = str(row['Group']).strip()
        sub_label_map[sub] = 'AD' if grp == 'A' else 'NC'

    # --- Load Structural Adjacency ---
    struct_path = os.path.join(PROCESSED_DIR, "structural_adjacency.csv")
    if not os.path.exists(struct_path):
        raise FileNotFoundError("Chưa chạy processing.py! Hãy chạy processing.py trước.")
        
    struct_adj = pd.read_csv(struct_path, header=None).values
    struct_edge_index = torch.tensor(struct_adj, dtype=torch.float32).nonzero().t().contiguous()

    # --- Load Features ---
    raw_data_dict = {'AD': [], 'NC': []}
    files = sorted([f for f in os.listdir(PROCESSED_DIR) if f.endswith('_features.csv')])
    
    print(f">>> Đang tải dữ liệu từ {len(files)} subjects...")
    
    for f in files:
        sub_id = f.split('_')[0]
        if sub_id not in sub_label_map: continue
        
        label_str = sub_label_map[sub_id]
        
        # 1. Load Features
        feat_path = os.path.join(PROCESSED_DIR, f)
        features_raw = pd.read_csv(feat_path, header=None).values 
        features_raw = np.nan_to_num(features_raw, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 2. Load 5-Band Adjacency
        adj_path = os.path.join(PROCESSED_DIR, f"{sub_id}_adj_multiband.npy")
        if not os.path.exists(adj_path): continue
        adj_5bands = np.load(adj_path) 
        adj_5bands = np.nan_to_num(adj_5bands, nan=0.0, posinf=0.0, neginf=0.0)
        
        raw_data_dict[label_str].append({
            'features': features_raw,
            'adj': adj_5bands,
            'id': sub_id
        })
        
    return raw_data_dict, struct_edge_index

def create_dataloaders(raw_data_dict, struct_edge_index, train_indices, val_indices, batch_size=10):
    scaler = StandardScaler()
    
    ad_list = raw_data_dict['AD']
    nc_list = raw_data_dict['NC']
    ad_ids = {item['id'] for item in ad_list}
    
    def get_subs(indices, source_list):
        return [source_list[i] for i in indices if i < len(source_list)]

    train_subs = get_subs(train_indices['AD'], ad_list) + get_subs(train_indices['NC'], nc_list)
    val_subs = get_subs(val_indices['AD'], ad_list) + get_subs(val_indices['NC'], nc_list)

    if not train_subs: return None, None

    # Fit Scaler
    train_feats_all = []
    for sub in train_subs:
        train_feats_all.append(sub['features'])
    
    all_train_data = np.vstack(train_feats_all)
    scaler.fit(all_train_data)
    scaler.scale_ = np.maximum(scaler.scale_, 1e-5)
    
    def convert_to_pyg(sub_list, is_train=False):
        data_list = []
        for sub in sub_list:
            feats_scaled = scaler.transform(sub['features'])
            feats_scaled = np.nan_to_num(feats_scaled, nan=0.0)
            
            n_segs = feats_scaled.shape[0]
            features_3d = feats_scaled.reshape(n_segs, 19, 5)
            
            y_label = 0 if sub['id'] in ad_ids else 1 
            
            adj_5bands = sub['adj']
            edge_indices = []
            edge_weights = []
            
            for b in range(5):
                adj_t = torch.tensor(adj_5bands[b], dtype=torch.float32)
                
                # --- QUAN TRỌNG: LẤY TRỊ TUYỆT ĐỐI ---
                abs_adj = torch.abs(adj_t)
                
                # Ngưỡng lọc nhiễu 0.5 (Có thể thử giảm xuống 0.3 nếu data quá thưa)
                mask = abs_adj > 0.5
                idx = mask.nonzero().t().contiguous()
                w = abs_adj[idx[0], idx[1]].unsqueeze(-1)
                
                edge_indices.append(idx)
                edge_weights.append(w)

            for i in range(n_segs):
                x = torch.tensor(features_3d[i], dtype=torch.float32)
                y = torch.tensor([y_label], dtype=torch.long)
                
                data = Data(x=x, y=y)
                data.edge_index_struct = struct_edge_index
                
                data.edge_index_b0, data.edge_weight_b0 = edge_indices[0], edge_weights[0]
                data.edge_index_b1, data.edge_weight_b1 = edge_indices[1], edge_weights[1]
                data.edge_index_b2, data.edge_weight_b2 = edge_indices[2], edge_weights[2]
                data.edge_index_b3, data.edge_weight_b3 = edge_indices[3], edge_weights[3]
                data.edge_index_b4, data.edge_weight_b4 = edge_indices[4], edge_weights[4]
                
                data_list.append(data)
        return data_list

    train_data = convert_to_pyg(train_subs, is_train=True)
    val_data = convert_to_pyg(val_subs, is_train=False)
    
    if len(train_data) == 0: return None, None

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# ==============================================================================
# 3. TRAINING LOOP
# ==============================================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        out = model(data, use_dropout=True)
        loss = criterion(out, data.y)
        
        if torch.isnan(loss): continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1)
        total_correct += (pred == data.y).sum().item()
        total_samples += data.num_graphs
        
    if total_samples == 0: return 0.0, 0.0
    return total_loss / total_samples, total_correct / total_samples

def evaluate(model, loader, criterion, device):
    model.eval()
    total_correct, total_samples = 0, 0
    all_targets, all_preds = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data, use_dropout=False)
            pred = out.argmax(dim=1)
            total_correct += (pred == data.y).sum().item()
            total_samples += data.num_graphs
            all_targets.extend(data.y.cpu().tolist())
            all_preds.extend(pred.cpu().tolist())
            
    acc = total_correct / total_samples if total_samples > 0 else 0
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    return acc, f1

# ==============================================================================
# 4. MAIN
# ==============================================================================
def main():
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        raw_data_dict, struct_edge_index = load_raw_data_dict()
    except Exception as e:
        print(f"Lỗi load data: {e}")
        return

    struct_edge_index = struct_edge_index.to(device)
    
    # Định nghĩa Folds
    folds_indices = [
        (
            [21, 2, 24, 11, 19, 30, 14, 35, 27, 0, 3, 10, 6, 12, 31, 4, 32, 25, 33, 13, 7, 28, 26, 9, 18, 8, 23, 1, 5], 
            [15, 22, 17, 20, 34, 16, 29], 
            [17, 25, 19, 0, 3, 16, 10, 6, 12, 2, 4, 22, 13, 7, 9, 18, 28, 8, 1, 5, 23, 14, 26], 
            [11, 21, 27, 15, 20, 24]
        ),
    ]

    print(f"Bắt đầu Training MF-MGCN (Optimal K=16) - 200 Epochs...")

    for fold_idx, (tr_ad, val_ad, tr_nc, val_nc) in enumerate(folds_indices):
        print(f'\n=== FOLD {fold_idx + 1} ===')
        
        train_indices = {'AD': tr_ad, 'NC': tr_nc}
        val_indices = {'AD': val_ad, 'NC': val_nc}
        
        train_loader, val_loader = create_dataloaders(
            raw_data_dict, struct_edge_index.cpu(), train_indices, val_indices, batch_size=10
        )
        
        if not train_loader:
            print("Skip fold: Không có dữ liệu train.")
            continue

        model = MF_MGCN().to(device)
        
        # [QUAN TRỌNG] Tăng lại Learning Rate lên 0.001 vì data đã sạch (theo Table 1)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # StepLR: Giảm LR sau mỗi 50 epoch (cho 200 epoch tổng)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        best_f1 = 0.0
        
        for epoch in range(200): # Chạy đủ 200 Epochs
            t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            v_acc, v_f1 = evaluate(model, val_loader, criterion, device)
            
            if v_acc > best_acc:
                best_acc = v_acc
                best_f1 = v_f1
            
            # Print tiến độ chi tiết hơn
            if epoch % 5 == 0 or epoch == 199:
                print(f"Ep {epoch:03d} | Loss: {t_loss:.4f} | Tr_Acc: {t_acc:.4f} | Val_Acc: {v_acc:.4f} | Best: {best_acc:.4f}")
            
            scheduler.step()

        print(f"--> Kết quả Fold {fold_idx+1}: Accuracy = {best_acc:.4f}, F1-Score = {best_f1:.4f}")

if __name__ == '__main__':
    main()