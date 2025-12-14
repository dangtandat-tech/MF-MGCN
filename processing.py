import os
import glob
import numpy as np
import pandas as pd
import mne
import scipy.stats
import warnings
import pymatreader # <--- QUAN TRỌNG: Import để MNE tự động dùng

# Tắt cảnh báo không cần thiết
warnings.filterwarnings("ignore")

# Cấu hình theo bài báo
BANDS = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 25),
    'Gamma': (25, 45)
}

WINDOW_SIZE = 10  # giây
OVERLAP = 0.9     # 90%

STANDARD_CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 
    'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 
    'Pz', 'P4', 'T6', 'O1', 'O2'
]

def calculate_de(signal_data):
    # Thêm epsilon nhỏ để tránh log(0)
    variance = np.var(signal_data, axis=-1)
    de = 0.5 * np.log2(2 * np.pi * np.e * variance + 1e-10)
    return de

def process_one_subject(file_path, save_dir, sub_id):
    try:
        # Đọc file .set với MNE
        # MNE sẽ tự động dùng pymatreader nếu file là MATLAB v7.3
        raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
        
        # Kiểm tra kênh
        available_chans = raw.info['ch_names']
        # Chuẩn hóa tên kênh (chữ hoa/thường) nếu cần, ở đây giả định tên đúng chuẩn
        picks = [ch for ch in STANDARD_CHANNELS if ch in available_chans]
        
        if len(picks) < 19:
            print(f"Bỏ qua {sub_id}: Chỉ tìm thấy {len(picks)}/19 kênh chuẩn.")
            return

        raw.pick_channels(picks, ordered=True)
        
        sfreq = raw.info['sfreq']
        window_samples = int(WINDOW_SIZE * sfreq)
        step_samples = int(window_samples * (1 - OVERLAP))
        total_samples = raw.n_times
        
        all_segments_features = [] 
        
        # Cắt đoạn (Sliding Window)
        for start in range(0, total_samples - window_samples + 1, step_samples):
            end = start + window_samples
            segment_features = []
            
            # Lấy dữ liệu đoạn
            data_segment = raw.get_data(start=start, stop=end)
            
            for band_name, (l_freq, h_freq) in BANDS.items():
                # Lọc tần số
                filtered_data = mne.filter.filter_data(data_segment, sfreq, l_freq, h_freq, verbose=False)
                # Tính DE
                de_values = calculate_de(filtered_data) 
                segment_features.append(de_values)
            
            # Stack lại: [19 kênh, 5 bands]
            segment_features = np.stack(segment_features, axis=1)
            all_segments_features.append(segment_features)

        if not all_segments_features:
            print(f"Cảnh báo: {sub_id} quá ngắn.")
            return

        all_segments_features = np.array(all_segments_features)
        
        # Lưu Features (Flatten để dễ lưu CSV)
        num_segments, num_nodes, num_bands = all_segments_features.shape
        features_flat = all_segments_features.reshape(num_segments, -1)
        
        feat_save_path = os.path.join(save_dir, f"{sub_id}_features.csv")
        pd.DataFrame(features_flat).to_csv(feat_save_path, header=False, index=False)
        
        # Lưu Adjacency (Correlation matrix trung bình)
        mean_features = np.mean(all_segments_features, axis=0) # [19, 5]
        adj_matrix = np.corrcoef(mean_features) # [19, 19]
        
        adj_save_path = os.path.join(save_dir, f"{sub_id}_adjacency.csv")
        pd.DataFrame(adj_matrix).to_csv(adj_save_path, header=False, index=False)
        
        print(f"Đã xử lý: {sub_id} ({num_segments} đoạn)")

    except Exception as e:
        print(f"Lỗi xử lý {sub_id}: {e}")

def main_processing():
    # --- CẤU HÌNH ĐƯỜNG DẪN ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # SỬA: Đường dẫn tới thư mục gốc chứa 'sub-xxx'
    RAW_DATA_ROOT = r"D:/Downloads/Projects/MF-MGCN-main/MF-MGCN-main/ds004504"
    
    PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
    
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
    
    print(f"Đọc dữ liệu từ: {RAW_DATA_ROOT}")
    
    sub_folders = [f for f in os.listdir(RAW_DATA_ROOT) if f.startswith('sub-')]
    sub_folders.sort()
    
    if not sub_folders:
        print("LỖI: Không tìm thấy thư mục sub- nào.")
        return
    
    count = 0
    for sub_id in sub_folders:
        eeg_path = os.path.join(RAW_DATA_ROOT, sub_id, 'eeg')
        set_files = glob.glob(os.path.join(eeg_path, "*.set"))
        
        if set_files:
            process_one_subject(set_files[0], PROCESSED_DIR, sub_id)
            count += 1
            
    if count == 0:
        print("CẢNH BÁO: Không xử lý được file nào cả!")
    else:
        print(f"Hoàn tất! Đã xử lý thành công {count} subjects.")

if __name__ == "__main__":
    main_processing()