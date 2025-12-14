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

# Import file xử lý dữ liệu (đảm bảo file processing.py nằm cùng thư mục)
import processing

# --- CẤU HÌNH MÔ HÌNH ---
class EdgeGCN(nn.Module):
    def __init__(self):
        super(EdgeGCN, self).__init__()
        # CẬP NHẬT: Input features = 5 (tương ứng 5 băng tần: Delta, Theta, Alpha, Beta, Gamma)
        # Code cũ là 32, nhưng dữ liệu thực tế từ bài báo là 5 đặc trưng DE.
        self.conv1 = GCNConv(5, 16) 
        self.conv2 = GCNConv(16, 2)

        # Tính toán kích thước Linear layer đầu tiên
        # Sau khi flatten: 19 nodes * 2 output channels từ conv2 = 38
        self.lin = nn.Linear(19 * 2, 128) 
        self.lin1 = nn.Linear(128, 32)
        self.lin2 = nn.Linear(32, 2) # Output 2 class: AD và NC

        # Khởi tạo trọng số (Weight Initialization)
        nn.init.kaiming_normal_(self.lin.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.lin.bias, 0)
        nn.init.kaiming_normal_(self.lin1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.lin1.bias, 0)
        nn.init.kaiming_normal_(self.lin2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.lin2.bias, 0)

    def forward(self, data, use_dropout=True):
        # PyG DataLoader sẽ ghép (batch) các đồ thị lại thành 1 đồ thị lớn
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        
        # Layer 1: GCN
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)

        # Layer 2: GCN
        # edge_weight > 0.5 để tạo mask nhị phân cho cấu trúc (Structural Connectivity giả lập)
        x = self.conv2(x, edge_index, (edge_weight > 0.5).float())
        x = F.relu(x)

        # Flatten: PyG batching ghép các nodes lại dọc theo dimension 0.
        # Ta cần reshape lại để tách từng graph trong batch ra.
        # data.batch giúp ta biết node nào thuộc graph nào, nhưng ở đây dùng view/reshape nhanh
        # Batch size = -1 (tự tính), mỗi graph có 19 nodes * 2 features
        # Lưu ý: Cách reshape này chỉ đúng nếu số node mỗi graph luôn cố định là 19
        x = x.view(-1, 19 * 2) 

        if use_dropout:
            x = F.dropout(x, p=0.2, training=self.training)

        # Fully Connected Layers
        x = self.lin(x)
        x = F.relu(x)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)

        return x

# --- HÀM TẢI DỮ LIỆU MỚI ---
def load_processed_data():
    # Đường dẫn thư mục chứa dữ liệu đã xử lý
    # (Tự động lấy thư mục cùng cấp với file code này)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
    
    # Kiểm tra xem dữ liệu đã có chưa, nếu chưa thì chạy processing.py
    if not os.path.exists(PROCESSED_DIR) or not os.listdir(PROCESSED_DIR):
        print(">>> Chưa tìm thấy dữ liệu đã xử lý. Đang chạy processing.py...")
        # Gọi hàm main của processing.py để tạo dữ liệu
        try:
            processing.main_processing() 
        except Exception as e:
            print(f"LỖI NGHIÊM TRỌNG khi chạy processing: {e}")
            return {'AD': [], 'NC': []}
        
    print(f">>> Đang tải dữ liệu từ: {PROCESSED_DIR}")
    groups = {'AD': [], 'NC': []}
    
    # Lấy danh sách file features
    files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith('_features.csv')]
    
    if not files:
        print("Lỗi: Không tìm thấy file .csv nào trong folder processed_data!")
        return groups

    for f in files:
        sub_id = f.split('_')[0] # ví dụ: sub-001
        
        # --- LOGIC GÁN NHÃN (LABEL) ---
        # Tạm thời: sub-0xx LẺ là AD, CHẴN là NC (Logic giả định để test code)
        # Bạn cần thay đổi logic này dựa trên file participants.tsv thực tế
        try:
            sub_num = int(sub_id.split('-')[1])
            category = 'AD' if sub_num % 2 != 0 else 'NC'
        except:
            category = 'NC' # Mặc định nếu tên file lạ

        label = 0 if category == 'AD' else 1
        
        # Đọc Features
        feat_path = os.path.join(PROCESSED_DIR, f)
        # Pandas đọc file không header
        features_flat = pd.read_csv(feat_path, header=None).values 
        
        # Reshape lại: [Số đoạn, 19 kênh, 5 bands]
        num_segments = features_flat.shape[0]
        # features_flat có shape (num_segments, 95) vì 19*5=95
        features_3d = features_flat.reshape(num_segments, 19, 5)
        
        # Đọc Adjacency Matrix
        adj_path = os.path.join(PROCESSED_DIR, f"{sub_id}_adjacency.csv")
        if not os.path.exists(adj_path):
            print(f"Thiếu file adjacency cho {sub_id}, bỏ qua.")
            continue
            
        adj_matrix = pd.read_csv(adj_path, header=None).values
        
        # Xử lý Graph (chuyển sang PyTorch Geometric Data)
        sub_data_list = []
        
        # Tạo edge_index từ adjacency matrix (chỉ làm 1 lần cho subject này)
        adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32)
        
        # Lọc các cạnh yếu để giảm nhiễu (Thresholding)
        # Chỉ giữ lại các kết nối có correlation > 0.5 (hoặc 0.3)
        edge_index = (torch.abs(adj_tensor) > 0.5).nonzero().t()
        edge_weight = adj_tensor[edge_index[0], edge_index[1]]
        
        # Đảm bảo edge_weight có shape [num_edges, 1]
        edge_weight = edge_weight.unsqueeze(-1)

        for i in range(num_segments):
            # Đặc trưng của đoạn thứ i
            x = torch.tensor(features_3d[i], dtype=torch.float32) # Shape [19, 5]
            y = torch.tensor([label], dtype=torch.long)
            
            # Tạo object Data
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)
            sub_data_list.append(data)
            
        # Thêm toàn bộ các đoạn cắt của subject này vào nhóm tương ứng
        # Lưu ý: Để tương thích với code chia Fold cũ (dựa trên index subject),
        # ta sẽ lưu theo dạng List lồng nhau: groups['AD'] = [ [sub1_data...], [sub2_data...] ]
        groups[category].append(sub_data_list)
        
    print(f"Đã tải xong: {len(groups['AD'])} subjects AD, {len(groups['NC'])} subjects NC.")
    return groups

# --- HÀM TRAIN & TEST ---
def train(model, train_loader, criterion, optimizer, device, use_dropout=True):
    model.train()
    total_loss, total_correct = 0, 0
    total_samples = 0
    
    for data in train_loader:
        data = data.to(device) # Chuyển dữ liệu sang GPU/CPU
        optimizer.zero_grad()
        
        out = model(data, use_dropout=use_dropout) # Forward pass
        
        loss = criterion(out, data.y) # Tính loss
        loss.backward() # Backward pass
        optimizer.step() # Cập nhật trọng số
        
        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1)
        total_correct += (pred == data.y).sum().item()
        total_samples += data.num_graphs
        
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    # print(f'Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    return avg_loss, accuracy

def test(model, test_loader, criterion, device):
    model.eval()
    total_loss, total_correct = 0, 0
    total_samples = 0
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data, use_dropout=False)
            
            loss = criterion(out, data.y)
            total_loss += loss.item() * data.num_graphs
            
            pred = out.argmax(dim=1)
            total_correct += (pred == data.y).sum().item()
            total_samples += data.num_graphs
            
            all_outputs.append(out)
            all_targets.extend(data.y.cpu().tolist())

    if len(all_outputs) > 0:
        all_outputs = torch.cat(all_outputs, dim=0)
        # Softmax để lấy xác suất cho class 1 (NC hoặc AD tùy quy định)
        all_probs = F.softmax(all_outputs, dim=1)[:, 1].cpu().numpy()
        all_preds = all_outputs.argmax(dim=1).cpu().numpy()
    else:
        all_probs = []
        all_preds = []

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    # Tính các chỉ số sklearn (cần xử lý trường hợp ngoại lệ nếu batch rỗng hoặc chỉ có 1 class)
    try:
        precision = precision_score(all_targets, all_preds, zero_division=0)
        recall = recall_score(all_targets, all_preds, zero_division=0)
        f1 = f1_score(all_targets, all_preds, zero_division=0)
        # AUC chỉ tính được khi có cả 2 class trong tập test
        if len(set(all_targets)) > 1:
            auc = roc_auc_score(all_targets, all_probs)
        else:
            auc = 0.0
    except:
        precision, recall, f1, auc = 0, 0, 0, 0

    return avg_loss, accuracy, precision, recall, f1, auc

# --- MAIN FUNCTION ---
def main():
    # Kiểm tra thiết bị (GPU ưu tiên)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng thiết bị: {device}")

    # 1. Tải dữ liệu đã qua xử lý
    groups = load_processed_data()
    ad_groups = groups['AD']
    nc_groups = groups['NC']

    # Kiểm tra dữ liệu
    if len(ad_groups) == 0 and len(nc_groups) == 0:
        print("KHÔNG CÓ DỮ LIỆU. Vui lòng kiểm tra lại đường dẫn trong processing.py")
        return

    # Danh sách chia Fold thủ công (như code gốc của bạn)
    # Lưu ý: Index trong list này phải nhỏ hơn tổng số subject thực tế bạn có.
    # Ví dụ: Nếu bạn chỉ có 20 sub (sub-001 đến sub-020), mà index gọi tới 35 sẽ lỗi.
    # Tạm thời giữ nguyên list của bạn, nhưng cần đảm bảo dataset đủ lớn.
    fixed_indices = [
        # Fold 1
        ([21, 2, 24, 11, 19, 30, 14, 35, 27, 0, 3, 10, 6, 12, 31, 4, 32, 25, 33, 13, 7, 28, 26, 9, 18, 8, 23, 1, 5], [15, 22, 17, 20, 34, 16, 29], [17, 25, 19, 0, 3, 16, 10, 6, 12, 2, 4, 22, 13, 7, 9, 18, 28, 8, 1, 5, 23, 14, 26], [11, 21, 27, 15, 20, 24]),
        # ... Các fold khác (tôi rút gọn để demo, bạn hãy paste đầy đủ list fold cũ vào đây nếu muốn chạy full)
    ]
    
    # Nếu dataset nhỏ hơn index trong fixed_indices, code sẽ lỗi "list index out of range".
    # Giải pháp an toàn: Tự động chia fold nếu index không khớp hoặc chạy thử 1 fold đơn giản.
    # Ở đây tôi check độ dài dataset trước.
    total_subjects_ad = len(ad_groups)
    total_subjects_nc = len(nc_groups)
    
    # Nếu số lượng subject ít hơn số trong fixed_indices, ta dùng KFold tự động thay thế
    use_auto_kfold = False
    if total_subjects_ad < 36 or total_subjects_nc < 29:
        print(f"Dataset thực tế ({total_subjects_ad} AD, {total_subjects_nc} NC) nhỏ hơn cấu hình Fixed Fold gốc.")
        print("-> Chuyển sang chế độ K-Fold Cross Validation tự động (5 Folds).")
        use_auto_kfold = True
    
    fold_results = []
    
    # Logic chạy lặp qua các Fold
    num_folds = 5
    
    if use_auto_kfold:
        # Tạo danh sách index
        kf = 5 # số fold
        # Chia index cho AD
        ad_indices = list(range(total_subjects_ad))
        nc_indices = list(range(total_subjects_nc))
        
        # Trộn ngẫu nhiên
        np.random.shuffle(ad_indices)
        np.random.shuffle(nc_indices)
        
        # Chia chunk
        ad_chunks = np.array_split(ad_indices, kf)
        nc_chunks = np.array_split(nc_indices, kf)
        
        folds_indices_auto = []
        for i in range(kf):
            val_ad = ad_chunks[i]
            train_ad = np.concatenate([ad_chunks[j] for j in range(kf) if j != i])
            
            val_nc = nc_chunks[i]
            train_nc = np.concatenate([nc_chunks[j] for j in range(kf) if j != i])
            
            folds_indices_auto.append((train_ad, val_ad, train_nc, val_nc))
        
        loop_target = folds_indices_auto
    else:
        loop_target = fixed_indices

    print(f"Bắt đầu huấn luyện trên {len(loop_target)} Folds...")

    for fold, (train_ad_idxs, val_ad_idxs, train_nc_idxs, val_nc_idxs) in enumerate(loop_target):
        print(f'\n=== FOLD {fold + 1} ===')
        
        # Chuẩn bị dữ liệu cho Fold này
        train_list = []
        val_list = []

        # Gộp dữ liệu train (làm phẳng list vì groups[idx] trả về list các segments)
        for idx in train_ad_idxs:
            if idx < len(ad_groups): train_list.extend(ad_groups[idx])
        for idx in train_nc_idxs:
             if idx < len(nc_groups): train_list.extend(nc_groups[idx])
             
        # Gộp dữ liệu val
        for idx in val_ad_idxs:
            if idx < len(ad_groups): val_list.extend(ad_groups[idx])
        for idx in val_nc_idxs:
            if idx < len(nc_groups): val_list.extend(nc_groups[idx])

        # Tạo DataLoader
        if len(train_list) == 0:
            print("Cảnh báo: Train list rỗng, bỏ qua fold này.")
            continue
            
        train_loader = DataLoader(train_list, batch_size=32, shuffle=True) # Batch size lớn hơn chút cho nhanh
        val_loader = DataLoader(val_list, batch_size=32, shuffle=False)

        # Khởi tạo Model, Optimizer, Loss
        model = EdgeGCN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95) # Decay nhẹ hơn
        criterion = nn.CrossEntropyLoss()

        # Biến lưu kết quả tốt nhất của Fold này
        best_metric = {
            'acc': 0.0, 'pre': 0.0, 'rec': 0.0, 'f1': 0.0, 'auc': 0.0, 'epoch': 0
        }

        # Training Loop
        for epoch in range(1, 101): # Demo 100 epochs
            use_dropout = epoch > 5
            
            train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, use_dropout)
            val_loss, val_acc, pre, rec, f1, auc = test(model, val_loader, criterion, device)
            
            # Lưu kết quả tốt nhất dựa trên Accuracy (hoặc F1)
            if val_acc > best_metric['acc']:
                best_metric = {'acc': val_acc, 'pre': pre, 'rec': rec, 'f1': f1, 'auc': auc, 'epoch': epoch}
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
                scheduler.step()

        print(f"--> Kết quả tốt nhất Fold {fold+1} (tại Epoch {best_metric['epoch']}): Acc {best_metric['acc']:.4f}")
        fold_results.append(best_metric)

    # Tổng kết
    print('\n================================')
    print('TỔNG KẾT 5 FOLDS (Best Accuracy Epoch)')
    print('================================')
    if len(fold_results) > 0:
        avg_acc = np.mean([res['acc'] for res in fold_results])
        avg_f1 = np.mean([res['f1'] for res in fold_results])
        
        for i, res in enumerate(fold_results):
            print(f"Fold {i+1}: Acc: {res['acc']:.4f}, F1: {res['f1']:.4f}, AUC: {res['auc']:.4f}")
        
        print('--------------------------------')
        print(f"Average Accuracy: {avg_acc:.4f}")
        print(f"Average F1-Score: {avg_f1:.4f}")
    else:
        print("Không có kết quả nào được ghi nhận.")

if __name__ == '__main__':
    main()