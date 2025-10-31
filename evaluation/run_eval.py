import json
from evaluation.metrics import CBIRMetrics

def run_eval():
    """Chạy đánh giá CBIR dựa trên file predictions_all.json đã lưu"""

    # Load kết quả predictions
    with open("logs/predictions_all.json", "r", encoding="utf-8") as f:
        all_predictions = json.load(f)

    ground_truths = []
    predictions = []

    for subcategory, info in all_predictions.items():
        # Ground truth: nhãn đúng là chính subcategory
        gt = [subcategory]  # Hoặc list nhiều nhãn nếu có
        ground_truths.append(gt)

        # Predictions: lấy nhãn top-k
        pred = [res["label"] for res in info["results"]]
        predictions.append(pred)


    metrics = CBIRMetrics(
        ground_truths=ground_truths,
        predictions=predictions,
    )

    results = metrics.evaluate_all(k_list=[1, 5])
    print("========== CBIR Metrics ==========")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    with open("logs/cbir_metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("✅ Metrics saved to logs/cbir_metrics_summary.json")

