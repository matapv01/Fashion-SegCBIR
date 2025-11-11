import torch


FASHION_LABELS = {
    1: "hat",
    3: "sunglass", 
    4: "upper-clothes",
    5: "skirt",
    6: "pants",
    7: "dress",
    8: "belt",
    9: "left-shoe",
    10: "right-shoe",
    16: "bag",
    17: "scarf"
}

FASHION_LABELS_VI = {
    1: "Mũ",
    3: "Kính râm", 
    4: "Áo",
    5: "Chân váy",
    6: "Quần",
    7: "Váy liền",
    8: "Thắt lưng",
    9: "Giày trái",
    10: "Giày phải",
    16: "Túi xách",
    17: "Khăn"
}

FASHION_COLORS = {
    1: [255, 0, 0],      # hat - red
    3: [0, 255, 0],      # sunglass - green
    4: [0, 255, 255],    # upper-clothes - cyan
    5: [255, 0, 255],    # skirt - magenta
    6: [128, 0, 128],    # pants - purple
    7: [255, 192, 203],  # dress - pink
    8: [165, 42, 42],    # belt - brown
    9: [255, 165, 0],    # left-shoe - orange
    10: [255, 20, 147],  # right-shoe - deep pink
    16: [75, 0, 130],    # bag - indigo
    17: [255, 105, 180]  # scarf - hot pink
}

topk = 50 # số lượng kết quả truy xuất ban đầu
topn = 5 # số lượng kết quả sau khi rerank
threshold = -15  # threshold for rerank score

# trọng số cho embedding
text_weight = 0.5
image_weight = 0.5

device = "cuda" if torch.cuda.is_available() else "cpu"
data_root = "data"
