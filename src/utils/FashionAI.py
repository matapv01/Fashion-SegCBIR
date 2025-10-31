import os
import numpy as np
import pandas as pd



# Trỏ đến thư mục gốc (ví dụ: fashion-SegCBIR)
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

def load_metadata(pkl_path="vector_database/df_with_vectors.pkl"):
    """
    Load DataFrame chứa vector embeddings, filename và label.
    Sau đó trả về 3 biến:
        - all_vectors: np.ndarray (N x D)
        - all_files: list[str]
        - all_labels: list[str]
    """
    # ---- Kiểm tra và load file .pkl ----
    full_path = os.path.join(root_dir, pkl_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file: {full_path}")

    df = pd.read_pickle(full_path)
    print(f"✅ Đã load DataFrame với {len(df)} bản ghi từ {full_path}")

    # ---- Kiểm tra cột cần thiết ----
    required_cols = ["vector"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Thiếu cột '{col}' trong DataFrame!")

    # ---- Lấy vector, filenames và labels ----
    all_vectors = np.vstack(df["vector"].values).astype(np.float32)
    all_vectors /= np.linalg.norm(all_vectors, axis=1, keepdims=True)

    # Xử lý cột link
    if "link" in df.columns:
        all_files = df["link"].tolist()
    else:
        raise KeyError("Không tìm thấy cột chứa tên file (link) trong DataFrame!")

    # Xử lý cột subCategory
    if "subCategory" in df.columns:
        all_labels = df["subCategory"].tolist()
    else:
        all_labels = ["unknown"] * len(df)
        print("⚠️ Không tìm thấy cột 'subCategory', gán mặc định là 'unknown'.")

    print(f"✅ all_vectors: {all_vectors.shape}, all_files: {len(all_files)}, all_labels: {len(all_labels)}")
    return df, all_vectors, all_files, all_labels


class FashionAI:
    def __init__(self):
        self.all_vectors = None
        self.all_files = None
        self.all_labels = None
        self.load_metadata()

    def load_metadata(self, pkl_path="vector_database/df_with_vectors.pkl"):
        _, self.all_vectors, self.all_files, self.all_labels = load_metadata(pkl_path)
