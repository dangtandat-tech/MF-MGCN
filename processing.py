import os
import glob
import numpy as np
import pandas as pd
import mne
import warnings

# Tắt cảnh báo không cần thiết để output gọn gàng hơn
warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CẤU HÌNH THAM SỐ (THEO ĐÚNG BÀI BÁO)
# ==============================================================================

# Các băng tần (Frequency Bands) được định nghĩa chính xác theo mục 2.1 [cite: 252]
BANDS = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 25),
    'Gamma': (25, 45)
}

# Thông số cửa sổ trượt (Sliding Window):
# - Độ dài T = 10 giây [cite: 317]
# - Độ chồng lấp (Overlap) = 90% 
WINDOW_SIZE = 10  
OVERLAP = 0.9     

# Thời lượng tín hiệu: Chọn 5 phút (300 giây) từ mỗi người tham gia [cite: 307]
DURATION_LIMIT = 300  

# Danh sách 19 kênh chuẩn theo hệ thống 10-20 [cite: 286, 304]
STANDARD_CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 
    'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 
    'Pz', 'P4', 'T6', 'O1', 'O2'
]

# Định nghĩa vùng não để xây dựng Ma trận Kề Cấu trúc (Structural Connectivity)
# Các kênh được nhóm vào các vùng Frontal, Central, Parietal, Temporal, Occipital [cite: 287, 288]
BRAIN_REGIONS = {
    'Frontal': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8'],
    'Central': ['C3', 'Cz', 'C4'],
    'Parietal': ['P3', 'Pz', 'P4'],
    'Temporal': ['T3', 'T4', 'T5', 'T6'],
    'Occipital': ['O1', 'O2']
}

# ==============================================================================
# 2. CÁC HÀM TÍNH TOÁN CỐT LÕI
# ==============================================================================

def calculate_de(signal_data):
    """
    Tính đặc trưng Differential Entropy (DE).
    Công thức (3): DE = 0.5 * log2(2 * pi * e * sigma^2). [cite: 245]
    """
    # Tính phương sai (variance - sigma^2) dọc theo trục thời gian (axis=-1)
    variance = np.var(signal_data, axis=-1)
    
    # Thêm 1e-10 để tránh lỗi chia cho 0 hoặc log(0)
    de_value = 0.5 * np.log2(2 * np.pi * np.e * variance + 1e-10)
    return de_value

def create_structural_adjacency():
    """
    Tạo ma trận kề cấu trúc (Spatial Connectivity - A_struct).
    Theo công thức (7): Nếu hai điện cực thuộc cùng một vùng não thì A[m,n]=1, ngược lại = 0[cite: 290, 291].
    """
    n_nodes = len(STANDARD_CHANNELS)
    adj = np.zeros((n_nodes, n_nodes))
    
    # Mapping từng kênh vào vùng não tương ứng
    chan_to_region = {}
    for region, chans in BRAIN_REGIONS.items():
        for ch in chans:
            chan_to_region[ch] = region
            
    # Duyệt từng cặp kênh để xác định kết nối
    for i, ch1 in enumerate(STANDARD_CHANNELS):
        for j, ch2 in enumerate(STANDARD_CHANNELS):
            region1 = chan_to_region.get(ch1)
            region2 = chan_to_region.get(ch2)
            
            # Logic: Cùng vùng não => kết nối (1)
            if region1 is not None and region2 is not None and region1 == region2:
                adj[i, j] = 1
            else:
                adj[i, j] = 0
    return adj

# ==============================================================================
# 3. QUY TRÌNH XỬ LÝ CHO MỘT BỆNH NHÂN (SUBJECT)
# ==============================================================================

def process_one_subject(file_path, save_dir, sub_id):
    try:
        # --- BƯỚC 1: ĐỌC DỮ LIỆU & CHỌN KÊNH ---
        # Load dữ liệu thô (đã qua tiền xử lý cơ bản của tác giả dataset gốc) [cite: 310]
        raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
        
        # Lọc chỉ lấy 19 kênh chuẩn
        available_chans = raw.info['ch_names']
        picks = [ch for ch in STANDARD_CHANNELS if ch in available_chans]
        
        if len(picks) < 19:
            print(f"Bỏ qua {sub_id}: Thiếu kênh (Chỉ tìm thấy {len(picks)}/19).")
            return

        raw.pick_channels(picks, ordered=True) # Sắp xếp đúng thứ tự
        
        # --- BƯỚC 2: CẮT DỮ LIỆU ĐỒNG NHẤT (5 PHÚT) [cite: 307] ---
        if raw.times[-1] > DURATION_LIMIT:
            raw.crop(tmin=0, tmax=DURATION_LIMIT)
        else:
            print(f"Lưu ý: {sub_id} ngắn hơn 5 phút ({raw.times[-1]:.1f}s).")

        # --- BƯỚC 3: PHÂN RÃ TẦN SỐ (FILTERING) [cite: 116] ---
        # Quan trọng: Lọc trên toàn bộ tín hiệu TRƯỚC khi cắt nhỏ (Sliding Window)
        # để tránh méo tín hiệu ở biên (edge artifacts).
        filtered_signals = {}
        for band_name, (l_freq, h_freq) in BANDS.items():
            raw_copy = raw.copy()
            # Dùng bộ lọc FIR thiết kế firwin (chuẩn MNE)
            raw_copy.filter(l_freq, h_freq, fir_design='firwin', verbose=False)
            filtered_signals[band_name] = raw_copy.get_data() # Shape: (19, n_total_samples)

        # --- BƯỚC 4: CẮT CỬA SỔ TRƯỢT & TRÍCH XUẤT ĐẶC TRƯNG DE [cite: 312, 313] ---
        sfreq = raw.info['sfreq']
        window_samples = int(WINDOW_SIZE * sfreq)
        step_samples = int(window_samples * (1 - OVERLAP)) # Bước nhảy (10s * 10% = 1s)
        total_samples = raw.n_times
        
        features_list = [] 
        
        # Vòng lặp sliding window
        for start in range(0, total_samples - window_samples + 1, step_samples):
            end = start + window_samples
            
            segment_band_features = []
            for band_name in BANDS.keys():
                # Lấy dữ liệu của băng tần tương ứng trong cửa sổ thời gian
                data_segment = filtered_signals[band_name][:, start:end]
                
                # Tính DE [cite: 235]
                de_val = calculate_de(data_segment) 
                segment_band_features.append(de_val)
            
            # Xếp chồng 5 băng tần: Shape (19 channels, 5 bands)
            # Đây là input features cho mỗi node trong đồ thị
            segment_features_matrix = np.stack(segment_band_features, axis=1) 
            features_list.append(segment_features_matrix)
            
        if not features_list:
            print(f"Lỗi: Không tạo được đoạn nào cho {sub_id} (Dữ liệu quá ngắn).")
            return

        # Chuyển list thành mảng numpy: (Số đoạn, 19, 5)
        all_features = np.array(features_list)
        
        # --- BƯỚC 5: TÍNH MA TRẬN KỀ CHỨC NĂNG (FUNCTIONAL CONNECTIVITY) [cite: 204] ---
        # Tính Pearson correlation dựa trên đặc trưng trung bình của các đoạn.
        # Điều này tạo ra một "Functional Graph" đại diện cho subject đó.
        mean_features = np.mean(all_features, axis=0) # Trung bình dọc theo các đoạn -> Shape (19, 5)
        functional_adj = np.corrcoef(mean_features)   # Shape (19, 19)

        # --- BƯỚC 6: LƯU DỮ LIỆU ĐÃ XỬ LÝ ---
        # 1. Lưu Features (Flatten thành 2D để dễ lưu CSV: mỗi dòng là 1 segment)
        n_segs, n_nodes, n_bands = all_features.shape
        features_flat = all_features.reshape(n_segs, -1)
        
        pd.DataFrame(features_flat).to_csv(os.path.join(save_dir, f"{sub_id}_features.csv"), header=False, index=False)
        
        # 2. Lưu Functional Adjacency Matrix
        pd.DataFrame(functional_adj).to_csv(os.path.join(save_dir, f"{sub_id}_adj_pearson.csv"), header=False, index=False)
        
        print(f"Đã xử lý xong: {sub_id} | Tạo được {n_segs} segments.")

    except Exception as e:
        print(f"LỖI khi xử lý {sub_id}: {str(e)}")

# ==============================================================================
# 4. HÀM MAIN (CHẠY TOÀN BỘ)
# ==============================================================================

def main_processing():
    # --- CẤU HÌNH ĐƯỜNG DẪN ---
    # SỬA DÒNG DƯỚI ĐÂY: Trỏ đến thư mục chứa dataset ds004504 của bạn
    RAW_DATA_ROOT = r"D:/Downloads/Projects/MF-MGCN-main/ds004504"
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
    
    # Tạo thư mục lưu kết quả nếu chưa có
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        
    # --- TẠO 1 LẦN: MA TRẬN CẤU TRÚC (STRUCTURAL ADJACENCY) [cite: 105, 290] ---
    # Ma trận này cố định dựa trên giải phẫu não, không phụ thuộc vào dữ liệu subject
    struct_adj = create_structural_adjacency()
    pd.DataFrame(struct_adj).to_csv(os.path.join(PROCESSED_DIR, "structural_adjacency.csv"), header=False, index=False)
    print(">>> Đã khởi tạo: Structural Adjacency Matrix (kết nối không gian).")

    # --- DUYỆT QUA CÁC THƯ MỤC BỆNH NHÂN (sub-xxx) ---
    if not os.path.exists(RAW_DATA_ROOT):
        print(f"LỖI: Không tìm thấy thư mục dữ liệu gốc tại {RAW_DATA_ROOT}")
        return

    sub_folders = [f for f in os.listdir(RAW_DATA_ROOT) if f.startswith('sub-')]
    sub_folders.sort()
    
    print(f">>> Tìm thấy {len(sub_folders)} đối tượng. Bắt đầu xử lý...")
    
    count = 0
    for sub_id in sub_folders:
        eeg_path = os.path.join(RAW_DATA_ROOT, sub_id, 'eeg')
        # Tìm file .set trong thư mục eeg của mỗi subject
        set_files = glob.glob(os.path.join(eeg_path, "*.set"))
        
        if set_files:
            # Xử lý file .set đầu tiên tìm thấy
            process_one_subject(set_files[0], PROCESSED_DIR, sub_id)
            count += 1
        else:
            print(f"Cảnh báo: Không tìm thấy file .set cho {sub_id}")
            
    print(f"\n>>> TỔNG KẾT: Đã xử lý thành công {count}/{len(sub_folders)} đối tượng.")
    print(f">>> Dữ liệu đã được lưu tại: {PROCESSED_DIR}")

if __name__ == "__main__":
    main_processing()