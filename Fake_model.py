import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import summary
from collections import defaultdict

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class SimpleGCN(nn.Module):
    def __init__(self, num_node_features=5, num_nodes=19, num_classes=2):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 8)
        self.fc = nn.Linear(num_nodes * 8, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch_size = data.num_graphs

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = x.view(batch_size, -1)

        out = self.fc(x)
        return out
    
def load_raw_data_dict():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
    tsv_path = os.path.join(BASE_DIR, "participants.tsv")

    try:
        df = pd.read_csv(tsv_path, sep='\t')
    except:
        df = pd.read_csv(tsv_path, sep='\s+')
    
    sub_label_map = {}
    for _, row in df.iterrows():
        sub = row['participant_id']
        grp = str(row['Group']).strip()
        
        if grp == 'A':
            sub_label_map[sub] = 'AD'
        elif grp == 'C':
            sub_label_map[sub] = 'NC'
    
    struct_adj = pd.read_csv(os.path.join(PROCESSED_DIR, "structural_adjacency.csv"), header=None).values
    # print(struct_adj)
    struct_edge_index = torch.tensor(struct_adj, dtype=torch.float32).nonzero().t().contiguous()
    # print(struct_edge_index)

    raw_data_dict = {'AD':[], 'NC':[]}
    files = sorted([f for f in os.listdir(PROCESSED_DIR) if f.endswith('_features.csv')])
    
    print(f">>> Đang tải dữ liệu từ {len(files)} subjects...")
    for f in files:
        sub_id = f.split('_')[0]
        if sub_id not in sub_label_map:
            continue
        features_raw = pd.read_csv(os.path.join(PROCESSED_DIR, f), header=None).values
        features_raw = np.nan_to_num(features_raw, nan=0.0)
        raw_data_dict[sub_label_map[sub_id]].append({
            'features': features_raw,
            'id': sub_id
        })
    
    return raw_data_dict, struct_edge_index

def create_dataloaders(raw_data_dict, struct_edge_index, train_indices, val_indices, batch_size=10):
    scaler = StandardScaler()
    ad_list, nc_list = raw_data_dict['AD'], raw_data_dict['NC']
    ad_ids = {item['id'] for item in ad_list}
    
    def get_subs(indices, source_list):
        return [source_list[i] for i in indices if i < len(source_list)]
    
    train_subs = get_subs(train_indices['AD'], ad_list) + get_subs(train_indices['NC'], nc_list)
    val_subs = get_subs(val_indices['AD'], ad_list) + get_subs(val_indices['NC'], nc_list)

    if not train_subs:
        print(f"Data tập Train bị rỗng!!!")
        return None, None

    all_train_data = np.vstack([sub['features'] for sub in train_subs])
    scaler.fit(all_train_data)

    def convert_to_pyg(sub_list):
        data_list = []
        cnt_0, cnt_1 = 0, 0
        for sub in sub_list:
            # print(sub['features'])
            feats_scaled = np.nan_to_num(scaler.transform(sub['features']))
            # print(feats_scaled)

            n_segs = feats_scaled.shape[0]

            features_3d = feats_scaled.reshape(n_segs, 19, 5)

            y_label = 0 if sub['id'] in ad_ids else 1

            if y_label == 0:
                cnt_0 += 1
            else:
                cnt_1 +=1

            for i in range(n_segs):
                x = torch.tensor(features_3d[i], dtype=torch.float32)
                y = torch.tensor([y_label],dtype=torch.long)
                data = Data(x=x, y= y, edge_index=struct_edge_index, sub_id=sub['id'])
                data_list.append(data)
        
        print(f"Label 0: {cnt_0}")
        print(f"Label 1: {cnt_1}")
        
        return data_list
    
    train_loader = DataLoader(
        convert_to_pyg(train_subs),
        batch_size = batch_size,
        shuffle = True
    )
    val_loader = DataLoader(
        convert_to_pyg(val_subs),
        batch_size = batch_size,
        shuffle = False
    )
    return train_loader, val_loader

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        total_correct += (out.argmax(dim=1) == data.y).sum().item()
        total_samples += data.num_graphs

    return total_loss / total_samples, total_correct / total_samples

def evaluate(model, loader, device):
    model.eval()
    
    # patient_preds = defaultdict(list)
    # patient_trues = defaultdict(int)
    
    # with torch.no_grad():
    #     for data in loader:
    #         data = data.to(device)
    #         out = model(data)
    #         preds = out.argmax(dim=1).cpu().tolist()
    #         trues =  data.y.cpu().tolist()

    #         sub_ids = data.sub_id

    #         for i in range(len(sub_ids)):
    #             sub_id = sub_ids[i]
    #             patient_preds[sub_id].append(preds[i])
    #             patient_trues[sub_id] = trues[i]
    
    # final_preds = []
    # final_trues = []

    # for sub_id, preds_list in patient_preds.items():
    #     count_0 = preds_list.count(0)
    #     count_1 = preds_list.count(1)

    #     final_vote = 1 if count_1 > count_0 else 0

    #     final_preds.append(final_vote)
    #     final_trues.append(patient_trues[sub_id])

    # acc = accuracy_score(final_trues, final_preds)
    # f1 = f1_score(final_trues, final_preds, average='macro', zero_division=0)
    # return acc, f1

    total_correct, total_samples = 0, 0
    all_targets, all_preds = [], []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            total_correct += (pred == data.y).sum().item()
            total_samples += data.num_graphs
            all_targets.extend(data.y.cpu().tolist())
            all_preds.extend(pred.cpu().tolist())
        
    acc = total_correct / total_samples
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    return acc, f1

def main():
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng device: {device}")

    raw_data_dict, struct_edge_index = load_raw_data_dict()

    folds_indices = [
        # Fold 1 (Nguyên bản của bạn)
        ([21, 2, 24, 11, 19, 30, 14, 35, 27, 0, 3, 10, 6, 12, 31, 4, 32, 25, 33, 13, 7, 28, 26, 9, 18, 8, 23, 1, 5], 
         [15, 22, 17, 20, 34, 16, 29], 
         [17, 25, 19, 0, 3, 16, 10, 6, 12, 2, 4, 22, 13, 7, 9, 18, 28, 8, 1, 5, 23, 14, 26], 
         [11, 21, 27, 15, 20, 24]),
         
        # Fold 2 (Tạo mới: Val AD có 7 người mới, Val NC có 6 người mới)
        ([15, 22, 17, 20, 34, 16, 29, 35, 27, 0, 3, 10, 6, 12, 31, 4, 32, 25, 33, 13, 7, 28, 26, 9, 18, 8, 23, 1, 5],
         [21, 2, 24, 11, 19, 30, 14],
         [11, 21, 27, 15, 20, 24, 10, 6, 12, 2, 4, 22, 13, 7, 9, 18, 28, 8, 1, 5, 23, 14, 26],
         [17, 25, 19, 0, 3, 16]),

        # Fold 3 (Tạo mới: Val AD có 7 người mới, Val NC có 6 người mới)
        ([15, 22, 17, 20, 34, 16, 29, 21, 2, 24, 11, 19, 30, 14, 31, 4, 32, 25, 33, 13, 7, 28, 26, 9, 18, 8, 23, 1, 5],
         [35, 27, 0, 3, 10, 6, 12],
         [11, 21, 27, 15, 20, 24, 17, 25, 19, 0, 3, 16, 13, 7, 9, 18, 28, 8, 1, 5, 23, 14, 26],
         [10, 6, 12, 2, 4, 22])
    ]

    print("===== Bắt đầu Training =====")
    best_vals = []
    for fold_idx,  (tr_ad, val_ad, tr_nc, val_nc) in enumerate(folds_indices):
        print(f"\n===== Đang xử lý Fold {fold_idx} =====")
        train_loader, val_loader= create_dataloaders(
            raw_data_dict,
            struct_edge_index,
            {'AD': tr_ad, 'NC': tr_nc},
            {'AD': val_ad, 'NC': val_nc}
        )

        model = SimpleGCN().to(device)
        dummy_data = next(iter(train_loader)).to(device)
        
        if fold_idx == 0:
            try:
                # PyG summary chỉ cần truyền model và các tham số đầu vào của hàm forward
                print(summary(model, dummy_data))
            except Exception as e:
                print(f"Lỗi: {e}")
            print("==============================\n")

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0

        for epoch in range(50):
            t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            v_acc, v_f1 = evaluate(model, val_loader, device)

            if v_acc > best_acc:
                best_acc = v_acc
            
            print(f"Epoch {epoch:02d} | Loss: {t_loss:.4f} | Train Acc: {t_acc:.4f} | Val Acc: {v_acc:.4f} | Val F1: {v_f1:.4f} | Best Val: {best_acc:.4f}")

        best_vals.append(best_acc)
    print(f"\nAverage Val Acc: {sum(best_vals) / len(best_vals)}")
if __name__ == '__main__':
    main()