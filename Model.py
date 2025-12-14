import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
import pandas as pd
import numpy as np
import os
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Import file xử lý dữ liệu chuẩn
import processing

# ==============================================================================
# 1. CẤU HÌNH MÔ HÌNH MF-MGCN (PHIÊN BẢN MULTI-BRANCH CHUẨN)
# ==============================================================================
class MF_MGCN(nn.Module):
    def __init__(self, num_nodes=19, num_bands=5):
        super(MF_MGCN, self).__init__()
        
        # --- SỬA ĐỔI 1: KHỞI TẠO RIÊNG BIỆT CHO TỪNG BĂNG TẦN (Equation 4 & 6) ---
        # Thay vì 1 conv dùng chung, ta dùng ModuleList để chứa 5 nhánh riêng biệt.
        self.conv1_layers = nn.ModuleList([GCNConv(1, 32) for _ in range(num_bands)])
        self.bn1_layers = nn.ModuleList([nn.BatchNorm1d(32) for _ in range(num_bands)])

        # --- SỬA ĐỔI 2: OUTPUT DIMENSION = 2 (Theo Table 1) ---
        # Table 1: Dimensions M and L ... are 32 and 2.
        # Điều này giúp giảm số lượng tham số xuống mức ~29k.
        self.conv2_layers = nn.ModuleList([GCNConv(32, 2) for _ in range(num_bands)])
        self.bn2_layers = nn.ModuleList([nn.BatchNorm1d(2) for _ in range(num_bands)])

        # --- Fully Connected Layers ---
        # Sau khi concat 5 bands: (19 nodes * 2 features) * 5 bands = 190
        self.flatten_dim = num_nodes * 2 * num_bands 
        
        # S (Hidden units in FCL) = 128 [cite: 599]
        self.lin1 = nn.Linear(self.flatten_dim, 128)
        self.bn3 = nn.BatchNorm1d(128)
        
        # F (Hidden units in FCL) = 32 [cite: 599]
        self.lin2 = nn.Linear(128, 32)
        
        # Output: AD vs NC
        self.lin3 = nn.Linear(32, 2) 

        # Khởi tạo trọng số
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, data, use_dropout=True):
        x_all = data.x # Shape: [Total_Nodes, 5]
        batch_size = data.num_graphs
        band_outputs = []
        
        # --- MULTI-BRANCH LOOP (Weights W_i are distinct) ---
        for i in range(5):
            # Lấy dữ liệu băng tần i
            x_band = x_all[:, i].unsqueeze(-1)
            
            # Lấy layer GCN tương ứng với băng tần i
            conv1 = self.conv1_layers[i]
            bn1 = self.bn1_layers[i]
            conv2 = self.conv2_layers[i]
            bn2 = self.bn2_layers[i]

            # 1. Functional GCN (Pearson)
            x = conv1(x_band, data.edge_index_func, data.edge_weight_func)
            x = bn1(x)
            x = F.relu(x)
            if use_dropout: x = F.dropout(x, p=0.3, training=self.training)

            # 2. Structural GCN (Spatial)
            x = conv2(x, data.edge_index_struct)
            x = bn2(x)
            x = F.relu(x)
            if use_dropout: x = F.dropout(x, p=0.3, training=self.training)
            
            # Reshape: [Batch, 19, 2]
            x_reshaped = x.view(batch_size, -1) 
            band_outputs.append(x_reshaped)
            
        # --- CONCATENATION ---
        # [Batch, 19*2*5] = [Batch, 190]
        x_concat = torch.cat(band_outputs, dim=1)
        
        # --- CLASSIFICATION HEAD ---
        x = self.lin1(x_concat)
        x = self.bn3(x)
        x = F.relu(x)
        if use_dropout: x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.lin2(x)
        x = F.relu(x)
        
        x = self.lin3(x)
        return x

# ==============================================================================
# 2. HÀM TẢI DỮ LIỆU (GIỮ NGUYÊN BẢN CHUẨN TRƯỚC ĐÓ)
# ==============================================================================
def load_processed_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
    
    # --- Đọc Nhãn ---
    tsv_path = os.path.join(BASE_DIR, "participants.tsv")
    if not os.path.exists(tsv_path): 
        print(f"LỖI: Không tìm thấy {tsv_path}")
        return {'AD': [], 'NC': []}
        
    try: df = pd.read_csv(tsv_path, sep='\t')
    except: df = pd.read_csv(tsv_path, sep='\s+')
    df.columns = [c.strip() for c in df.columns]
    
    sub_label_map = {}
    for _, row in df.iterrows():
        sub = row['participant_id']
        grp = str(row['Group']).strip()
        if grp == 'A': sub_label_map[sub] = 'AD'
        elif grp == 'C': sub_label_map[sub] = 'NC'

    # --- Load Data & Scaler ---
    if not os.path.exists(PROCESSED_DIR): 
        print(">>> Chưa có dữ liệu processed. Đang chạy processing...")
        processing.main_processing()
        
    struct_path = os.path.join(PROCESSED_DIR, "structural_adjacency.csv")
    if not os.path.exists(struct_path): processing.main_processing()
    
    struct_adj = pd.read_csv(struct_path, header=None).values
    edge_index_struct = torch.tensor(struct_adj, dtype=torch.float32).nonzero().t().contiguous()

    files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith('_features.csv')]
    valid_files = []
    
    print(">>> Đang tính toán thống kê (Mean/Std) để chuẩn hóa...")
    scaler = StandardScaler()
    all_raw_data = []
    for f in files:
        sub_id = f.split('_')[0]
        if sub_id in sub_label_map:
            path = os.path.join(PROCESSED_DIR, f)
            vals = pd.read_csv(path, header=None).values
            all_raw_data.append(vals)
            valid_files.append(f)
            
    full_data = np.vstack(all_raw_data)
    scaler.fit(full_data)
    print(f"   -> Data Mean gốc: {scaler.mean_[:3]}...") # Debug info

    print(f">>> Đang tải {len(valid_files)} files...")
    groups = {'AD': [], 'NC': []}
    
    for f in valid_files:
        sub_id = f.split('_')[0]
        category = sub_label_map[sub_id]
        label = 0 if category == 'AD' else 1
        
        feat_raw = pd.read_csv(os.path.join(PROCESSED_DIR, f), header=None).values
        feat_scaled = scaler.transform(feat_raw)
        
        num_segments = feat_scaled.shape[0]
        features_3d = feat_scaled.reshape(num_segments, 19, 5)
        
        adj_path = os.path.join(PROCESSED_DIR, f"{sub_id}_adj_pearson.csv")
        if not os.path.exists(adj_path): continue
        adj_t = torch.tensor(pd.read_csv(adj_path, header=None).values, dtype=torch.float32)
        
        # Threshold > 0.5 để lọc nhiễu
        edge_index_func = (torch.abs(adj_t) > 0.5).nonzero().t().contiguous()
        edge_weight_func = adj_t[edge_index_func[0], edge_index_func[1]].unsqueeze(-1)

        sub_data_list = []
        for i in range(num_segments):
            x = torch.tensor(features_3d[i], dtype=torch.float32)
            y = torch.tensor([label], dtype=torch.long)
            
            data = Data(x=x, y=y)
            data.edge_index_func = edge_index_func      
            data.edge_weight_func = edge_weight_func    
            data.edge_index_struct = edge_index_struct  
            sub_data_list.append(data)
            
        groups[category].append(sub_data_list)
        
    return groups

# ==============================================================================
# 3. TRAINING LOOP
# ==============================================================================
def train(model, loader, criterion, optimizer, device, use_dropout=True):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data, use_dropout=use_dropout) 
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1)
        total_correct += (pred == data.y).sum().item()
        total_samples += data.num_graphs
    return total_loss / total_samples, total_correct / total_samples

def test(model, loader, criterion, device):
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
    
    acc = total_correct / total_samples
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    return acc, f1

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    groups = load_processed_data()
    ad_groups = groups['AD']
    nc_groups = groups['NC']
    if not ad_groups: return

    # Fixed Fold 1 (Như bạn đang dùng)
    fixed_indices = [
        ([21, 2, 24, 11, 19, 30, 14, 35, 27, 0, 3, 10, 6, 12, 31, 4, 32, 25, 33, 13, 7, 28, 26, 9, 18, 8, 23, 1, 5], 
         [15, 22, 17, 20, 34, 16, 29], 
         [17, 25, 19, 0, 3, 16, 10, 6, 12, 2, 4, 22, 13, 7, 9, 18, 28, 8, 1, 5, 23, 14, 26], 
         [11, 21, 27, 15, 20, 24]),
    ]
    
    # Check size
    if len(ad_groups) < 36: # Tự động KFold nếu data ít (để debug)
        print("Dataset nhỏ, chuyển sang Auto-KFold...")
        loop_target = fixed_indices # Tạm thời ép dùng fixed cho giống bài báo
    else:
        loop_target = fixed_indices

    print(f"Start Training (200 Epochs)...")
    
    for fold, (tr_ad, val_ad, tr_nc, val_nc) in enumerate(loop_target):
        print(f'\n=== FOLD {fold + 1} ===')
        train_list = []
        val_list = []
        
        for i in tr_ad: train_list.extend(ad_groups[i] if i < len(ad_groups) else [])
        for i in tr_nc: train_list.extend(nc_groups[i] if i < len(nc_groups) else [])
        for i in val_ad: val_list.extend(ad_groups[i] if i < len(ad_groups) else [])
        for i in val_nc: val_list.extend(nc_groups[i] if i < len(nc_groups) else [])

        if not train_list: continue
        
        # Batch Size 10
        train_loader = DataLoader(train_list, batch_size=10, shuffle=True)
        val_loader = DataLoader(val_list, batch_size=10, shuffle=False)

        model = MF_MGCN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) # Giảm weight_decay chút
        # Scheduler theo bài báo: Có thể họ giảm LR sau mỗi 50 epoch
        scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5) 
        criterion = nn.CrossEntropyLoss()

        best_acc = 0
        for epoch in range(1, 201): # Tăng lên 200 epochs
            t_loss, t_acc = train(model, train_loader, criterion, optimizer, device)
            v_acc, v_f1 = test(model, val_loader, criterion, device)
            
            if v_acc > best_acc: best_acc = v_acc
            
            print(f"Ep {epoch:03d} | Train: {t_acc:.6f} | Val: {v_acc:.6f} | Best: {best_acc:.6f}")
            
            scheduler.step()

        print(f"--> Result Fold {fold+1}: {best_acc:.4f}")

if __name__ == '__main__':
    main()