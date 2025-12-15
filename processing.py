import os
import glob
import numpy as np
import pandas as pd
import mne
import warnings

# Tắt cảnh báo MNE không cần thiết để output gọn gàng
warnings.filterwarnings("ignore")
mne.set_log_level('ERROR')

# ==============================================================================
# 1. CẤU HÌNH THAM SỐ (THEO ĐÚNG BÀI BÁO)
# ==============================================================================

# [cite_start]Các băng tần (Frequency Bands) [cite: 252]
BANDS = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 25),
    'Gamma': (25, 45)
}

# Thông số cửa sổ trượt (Sliding Window):
# [cite_start]- Độ dài T = 10 giây [cite: 317]
# [cite_start]- Độ chồng lấp (Overlap) = 90% [cite: 110]
WINDOW_SIZE = 10  
OVERLAP = 0.9     

# [cite_start]Thời lượng tín hiệu: Chọn 5 phút (300 giây) [cite: 307]
DURATION_LIMIT = 300  

# [cite_start]Danh sách 19 kênh chuẩn theo hệ thống 10-20 [cite: 304]
STANDARD_CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 
    'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 
    'Pz', 'P4', 'T6', 'O1', 'O2'
]

# [cite_start]Định nghĩa vùng não [cite: 287, 288]
BRAIN_REGIONS = {
    'Frontal': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8'],
    'Central': ['C3', 'Cz', 'C4'],
    'Parietal': ['P3', 'Pz', 'P4'],
    'Temporal': ['T3', 'T4', 'T5', 'T6'],
    'Occipital': ['O1', 'O2']
}

# ==============================================================================
# 2. CÁC HÀM TÍNH TOÁN CỐT LÕI (ĐÃ GIA CỐ)
# ==============================================================================

def calculate_de(signal_data):
    """
    Tính đặc trưng Differential Entropy (DE) với cơ chế bảo vệ artifacts.
    [cite_start]Công thức (3): DE = 0.5 * log2(2 * pi * e * sigma^2). [cite: 245]
    """
    # [QUAN TRỌNG] Clip biên độ tín hiệu để loại bỏ nhiễu mắt/cơ bắp quá lớn
    # MNE lưu trữ dưới dạng Volts. 100 microVolts = 1e-4 V.
    # Bất cứ tín hiệu nào vượt quá +/- 100uV thường là nhiễu trong resting-state EEG.
    signal_data = np.clip(signal_data, -100e-6, 100e-6)
    
    # Tính phương sai (variance)
    variance = np.var(signal_data, axis=-1)
    
    # [QUAN TRỌNG] Tránh log(0) bằng cách đặt sàn cho phương sai
    variance = np.clip(variance, 1e-18, None)
    
    # Tính DE
    factor = 2 * np.pi * np.e
    de_value = 0.5 * np.log2(factor * variance)
    
    # [QUAN TRỌNG] Xử lý NaN/Inf nếu có
    # Giá trị DE của EEG thường nằm trong khoảng -20 đến 10 (tùy đơn vị log).
    # Nếu ra -Inf (do variance cực nhỏ), ta gán về giá trị thấp nhất hợp lý.
    de_value = np.nan_to_num(de_value, nan=0.0, posinf=0.0, neginf=-50.0)
    
    return de_value

def create_structural_adjacency():
    """
    Tạo ma trận kề cấu trúc (Spatial Connectivity).
    [cite_start]Công thức (7): A[m,n]=1 nếu cùng vùng não. [cite: 290, 291]
    """
    n_nodes = len(STANDARD_CHANNELS)
    adj = np.zeros((n_nodes, n_nodes))
    
    chan_to_region = {}
    for region, chans in BRAIN_REGIONS.items():
        for ch in chans:
            chan_to_region[ch] = region
            
    for i, ch1 in enumerate(STANDARD_CHANNELS):
        for j, ch2 in enumerate(STANDARD_CHANNELS):
            region1 = chan_to_region.get(ch1)
            region2 = chan_to_region.get(ch2)
            
            if region1 and region2 and region1 == region2:
                adj[i, j] = 1
            else:
                adj[i, j] = 0
    return adj

# ==============================================================================
# 3. QUY TRÌNH XỬ LÝ (SUBJECT PROCESSING)
# ==============================================================================

def process_one_subject(file_path, save_dir, sub_id):
    try:
        # --- BƯỚC 1: ĐỌC DỮ LIỆU ---
        try:
            raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
        except Exception as e:
            print(f"Lỗi đọc file {sub_id}: {e}")
            return

        # Chọn 19 kênh chuẩn
        available_chans = raw.info['ch_names']
        picks = [ch for ch in STANDARD_CHANNELS if ch in available_chans]
        
        if len(picks) < 19:
            print(f"BỎ QUA {sub_id}: Thiếu kênh (Tìm thấy {len(picks)}/19).")
            return

        raw.pick_channels(picks, ordered=True)
        
        # Cắt thời lượng
        if raw.times[-1] > DURATION_LIMIT:
            raw.crop(tmin=0, tmax=DURATION_LIMIT)

        # [QUAN TRỌNG] Xử lý NaN trong raw data ngay từ đầu
        data = raw.get_data()
        if np.isnan(data).any():
            data = np.nan_to_num(data, nan=0.0)
            raw._data = data

        # --- BƯỚC 2: PHÂN RÃ TẦN SỐ (Bandpass Filter) ---
        filtered_signals = {}
        for band_name, (l_freq, h_freq) in BANDS.items():
            raw_copy = raw.copy()
            # Dùng iir filter thay vì firwin để nhanh hơn và ổn định cho đoạn ngắn
            # skip_by_annotation='edge' để tránh lỗi biên
            try:
                raw_copy.filter(l_freq, h_freq, method='iir', verbose=False)
            except:
                # Fallback nếu IIR lỗi
                raw_copy.filter(l_freq, h_freq, verbose=False)
            filtered_signals[band_name] = raw_copy.get_data()

        # --- BƯỚC 3: SLIDING WINDOW & FEATURE EXTRACTION ---
        sfreq = raw.info['sfreq']
        window_samples = int(WINDOW_SIZE * sfreq)
        step_samples = int(window_samples * (1 - OVERLAP))
        total_samples = raw.n_times
        
        features_list = []
        
        for start in range(0, total_samples - window_samples + 1, step_samples):
            end = start + window_samples
            
            segment_band_features = []
            valid_segment = True
            
            for band_name in BANDS.keys():
                data_segment = filtered_signals[band_name][:, start:end]
                
                # Kiểm tra độ dài segment
                if data_segment.shape[1] < window_samples:
                    valid_segment = False
                    break
                
                # Tính DE
                de_val = calculate_de(data_segment)
                segment_band_features.append(de_val)
            
            if valid_segment:
                # Stack: (19 channels, 5 bands)
                segment_features_matrix = np.stack(segment_band_features, axis=1)
                features_list.append(segment_features_matrix)
        
        if not features_list:
            print(f"Cảnh báo: Không tạo được đoạn nào cho {sub_id} (Dữ liệu quá nhiễu hoặc ngắn).")
            return

        # Chuyển thành numpy array: (n_segs, 19, 5)
        all_features = np.array(features_list)
        n_segs = all_features.shape[0]

        # --- BƯỚC 4: TÍNH MULTI-GRAPH ADJACENCY (PEARSON) ---
        adj_matrices = []
        for band_idx in range(5):
            # Lấy features của band thứ band_idx: (n_segs, 19)
            band_data = all_features[:, :, band_idx]
            
            # Tính correlation giữa các kênh theo thời gian (biến thiên của DE)
            # Transpose -> (19, n_segs)
            channel_series = band_data.T
            
            # [QUAN TRỌNG] Xử lý trường hợp chuỗi hằng số (variance=0) gây ra NaN khi chia
            try:
                with np.errstate(divide='ignore', invalid='ignore'):
                    corr = np.corrcoef(channel_series)
            except:
                corr = np.zeros((19, 19))
            
            # Thay thế NaN bằng 0 (không tương quan)
            corr = np.nan_to_num(corr, nan=0.0)
            
            adj_matrices.append(corr)
            
        # Kết quả: (5, 19, 19)
        adj_multiband = np.array(adj_matrices)

        # --- BƯỚC 5: LƯU DỮ LIỆU ---
        # 1. Features
        features_flat = all_features.reshape(n_segs, -1)
        # Clean lần cuối trước khi lưu
        features_flat = np.nan_to_num(features_flat, nan=0.0)
        
        pd.DataFrame(features_flat).to_csv(os.path.join(save_dir, f"{sub_id}_features.csv"), header=False, index=False)
        
        # 2. Adjacency
        np.save(os.path.join(save_dir, f"{sub_id}_adj_multiband.npy"), adj_multiband)
        
        print(f"Processed: {sub_id} | Segments: {n_segs}")

    except Exception as e:
        print(f"FAILED {sub_id}: {str(e)}")

# ==============================================================================
# 4. HÀM MAIN
# ==============================================================================

def main_processing():
    # --- CẤU HÌNH ĐƯỜNG DẪN ---
    # !!! Hãy thay đổi đường dẫn này trỏ đến thư mục dataset ds004504 trên máy bạn !!!
    RAW_DATA_ROOT = r"D:/Downloads/Projects/MF-MGCN-main/ds004504"
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
    
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        
    # Tạo ma trận cấu trúc (1 lần)
    struct_adj = create_structural_adjacency()
    pd.DataFrame(struct_adj).to_csv(os.path.join(PROCESSED_DIR, "structural_adjacency.csv"), header=False, index=False)
    print(">>> Initialized: Structural Adjacency Matrix.")

    if not os.path.exists(RAW_DATA_ROOT):
        print(f"ERROR: Không tìm thấy thư mục {RAW_DATA_ROOT}")
        return

    sub_folders = [f for f in os.listdir(RAW_DATA_ROOT) if f.startswith('sub-')]
    sub_folders.sort()
    
    print(f">>> Tìm thấy {len(sub_folders)} subjects. Bắt đầu xử lý...")
    
    count = 0
    for sub_id in sub_folders:
        eeg_path = os.path.join(RAW_DATA_ROOT, sub_id, 'eeg')
        set_files = glob.glob(os.path.join(eeg_path, "*.set"))
        
        if set_files:
            process_one_subject(set_files[0], PROCESSED_DIR, sub_id)
            count += 1
        else:
            print(f"Warning: Không có file .set cho {sub_id}")
            
    print(f"\n>>> HOÀN TẤT: Đã xử lý {count}/{len(sub_folders)} subjects.")
    print(f">>> Dữ liệu lưu tại: {PROCESSED_DIR}")

if __name__ == "__main__":
    main_processing()