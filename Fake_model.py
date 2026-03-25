import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, recall_score
from torch_geometric.nn import GCNConv, GATConv, BatchNorm, global_mean_pool, global_max_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import summary
from collections import defaultdict

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class SimpleGCN(nn.Module):
    def __init__(self, num_node_features=5, hidden_dim=16, num_classes=2):
        super().__init__()

        self.conv1 = GATConv(num_node_features, hidden_dim, heads=2, concat=True)
        self.bn1 = BatchNorm(hidden_dim * 2)
        # self.conv1 = GCNConv(num_node_features, hidden_dim)

        self.conv2 = GCNConv(hidden_dim * 2, hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)

        # self.conv3 = GCNConv(hidden_dim, hidden_dim)
        # self.bn3 = BatchNorm(hidden_dim)

        # self.fc1 = nn.Linear(hidden_dim * 7, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim * 7, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        batch_size = data.num_graphs

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.35, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.35, training=self.training)
        # x1 = x

        # x = self.conv3(x, edge_index)
        # x = self.bn3(x)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.35, training=self.training)
        # x = x + x1

        x = x.view(batch_size, -1)

        # x_mean = global_mean_pool(x, batch)
        # x_max = global_max_pool(x, batch)

        # x = torch.cat([x_mean, x_max], dim=1)

        # x = self.fc1(x)
        out = self.fc2(x)
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
    files = sorted([f for f in os.listdir(PROCESSED_DIR) if f.endswith('_features.npy')])
    
    print(f">>> Đang tải dữ liệu từ {len(files)} subjects...")
    for f in files:
        sub_id = f.split('_')[0]
        if sub_id not in sub_label_map:
            continue
        features_raw = np.load(os.path.join(PROCESSED_DIR, f))
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

            features_3d = feats_scaled.reshape(n_segs, 7, 5)

            y_label = 0 if sub['id'] in ad_ids else 1

            # y_label = random.choice([0, 1])

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

def evaluate(model, loader, criterion, device):
    model.eval()
    
    # patient_preds = defaultdict(list)
    # patient_trues = defaultdict(int)
    # total_samples, total_loss = 0, 0

    # with torch.no_grad():
    #     for data in loader:
    #         data = data.to(device)
    #         out = model(data)
    #         loss = criterion(out, data.y)
    #         preds = out.argmax(dim=1).cpu().tolist()
    #         trues =  data.y.cpu().tolist()
    #         sub_ids = data.sub_id
    #         total_loss += loss.item() * data.num_graphs
    #         total_samples += data.num_graphs
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

    # loss = total_loss / total_samples
    # acc = accuracy_score(final_trues, final_preds)
    # f1 = f1_score(final_trues, final_preds, average='macro', zero_division=0)
    # recall = recall_score(final_trues, final_preds, average='macro', zero_division=0)
    # return acc, f1, recall

    total_correct, total_samples, total_loss = 0, 0, 0
    all_targets, all_preds = [], []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            pred = out.argmax(dim=1)
            total_loss += loss.item() * data.num_graphs
            total_correct += (pred == data.y).sum().item()
            total_samples += data.num_graphs
            all_targets.extend(data.y.cpu().tolist())
            all_preds.extend(pred.cpu().tolist())
        
    acc = total_correct / total_samples
    loss = total_loss / total_samples
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    return acc, f1, recall, loss

def main():
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng device: {device}")

    raw_data_dict, struct_edge_index = load_raw_data_dict()

    folds_indices = [        
        # Fold 0
        ([21, 2, 24, 11, 19, 30, 14, 35, 27, 0, 3, 10, 6, 12, 31, 4, 32, 25, 33, 13, 7, 28, 26, 9, 18, 8, 23, 1, 5], [15, 22, 17, 20, 34, 16, 29], [17, 25, 19, 0, 3, 16, 10, 6, 12, 2, 4, 22, 13, 7, 9, 18, 28, 8, 1, 5, 23, 14, 26], [11, 21, 27, 15, 20, 24]),
        # Fold 1
        ([18, 35, 32, 11, 2, 7, 5, 4, 22, 25, 13, 14, 6, 21, 26, 29, 9, 19, 1, 23, 12, 17, 20, 24, 15, 10, 31, 30, 0], [34, 16, 3, 28, 27, 33, 8], [8, 13, 5, 4, 21, 23, 16, 28, 14, 6, 22, 18, 9, 19, 1, 12, 17, 20, 24, 15, 10, 0, 7], [11, 3, 27, 26, 25, 2]),
        # Fold 2
        ([0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 35], [2, 6, 10, 15, 21, 32, 34], [0, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 17, 18, 19, 21, 22, 23, 24, 25, 26, 28], [1, 10, 15, 16, 20, 27]),
        # Fold 3
        ([1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35], [0, 7, 13, 16, 17, 20, 33], [0, 1, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 26, 27, 28], [2, 9, 11, 24, 25]),
        # Fold 4
        ([1, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35], [0, 3, 4, 6, 16, 22, 33], [0, 1, 3, 4, 5, 6, 8, 10, 11, 12, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 28], [2, 7, 9, 13, 20, 27]),
        # Fold 5
        ([15, 22, 17, 20, 34, 16, 29, 35, 27, 0, 3, 10, 6, 12, 31, 4, 32, 25, 33, 13, 7, 28, 26, 9, 18, 8, 23, 1, 5], [21, 2, 24, 11, 19, 30, 14], [11, 21, 27, 15, 20, 24, 10, 6, 12, 2, 4, 22, 13, 7, 9, 18, 28, 8, 1, 5, 23, 14, 26], [17, 25, 19, 0, 3, 16]),
        # Fold 6
        ([15, 22, 17, 20, 34, 16, 29, 21, 2, 24, 11, 19, 30, 14, 31, 4, 32, 25, 33, 13, 7, 28, 26, 9, 18, 8, 23, 1, 5], [35, 27, 0, 3, 10, 6, 12], [11, 21, 27, 15, 20, 24, 17, 25, 19, 0, 3, 16, 13, 7, 9, 18, 28, 8, 1, 5, 23, 14, 26], [10, 6, 12, 2, 4, 22])
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

        optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        patience = 15
        patience_counter = 0
        for epoch in range(100):
            t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            v_acc, v_f1, v_recall, v_loss = evaluate(model, val_loader, criterion, device)
            scheduler.step(v_acc)
            if v_acc > best_acc:
                best_acc = v_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:02d}| Lr: {lr:.8f}| Tr Loss: {t_loss:.7f}| Tr Acc: {t_acc:.7f}| Val Loss: {v_loss:.7f}| Val Acc: {v_acc:.7f}| Val F1: {v_f1:.7f}| Val Recall: {v_recall:.7f}| Best Val: {best_acc:.7f}")
            
            if patience_counter >= patience:
                print(f"Early Stopping tại Epoch {epoch}")
                break

        best_vals.append(best_acc)
    print(f"\nAverage Val Acc: {sum(best_vals) / len(best_vals)}")
if __name__ == '__main__':
    main()