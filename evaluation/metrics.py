import numpy as np
from sklearn.metrics import average_precision_score
from scipy.stats import rankdata

class CBIRMetrics:
    """
    Class tính các metric phổ biến cho hệ thống CBIR.
    """

    def __init__(self, ground_truths, predictions, similarities=None):
        """
        Args:
            ground_truths (List[List[int]]): danh sách ID thật (hoặc nhãn đúng) cho mỗi query
            predictions (List[List[int]]): danh sách ID dự đoán (top-k kết quả) cho mỗi query
            similarities (List[List[float]], optional): điểm similarity tương ứng (nếu có)
        """
        self.ground_truths = ground_truths
        self.predictions = predictions
        self.similarities = similarities

    def precision_at_k(self, k=5):
        precisions = []
        for gt, pred in zip(self.ground_truths, self.predictions):
            relevant = sum(p in gt for p in pred[:k])
            precisions.append(relevant / k)
        return np.mean(precisions)

    def recall_at_k(self, k=5):
        recalls = []
        for gt, pred in zip(self.ground_truths, self.predictions):
            relevant = sum(p in gt for p in pred[:k])
            recalls.append(min(relevant / len(gt), 1.0) if len(gt) > 0 else 0)
        return np.mean(recalls)

    def mean_average_precision(self):
        """
        mAP = mean of Average Precision over all queries
        """
        ap_scores = []
        for gt, pred in zip(self.ground_truths, self.predictions):
            y_true = [1 if p in gt else 0 for p in pred]
            y_score = np.linspace(1, 0, len(pred))  # giảm dần theo rank
            ap = average_precision_score(y_true, y_score) if any(y_true) else 0
            ap_scores.append(ap)
        return np.mean(ap_scores)

    def ndcg_at_k(self, k=5):
        """
        nDCG (Normalized Discounted Cumulative Gain)
        """
        def dcg(relevances):
            return np.sum([
                rel / np.log2(idx + 2) for idx, rel in enumerate(relevances)
            ])

        ndcgs = []
        for gt, pred in zip(self.ground_truths, self.predictions):
            relevances = [1 if p in gt else 0 for p in pred[:k]]
            ideal = sorted(relevances, reverse=True)
            ndcg = dcg(relevances) / dcg(ideal) if dcg(ideal) > 0 else 0
            ndcgs.append(ndcg)
        return np.mean(ndcgs)

    def avg_cosine_similarity(self, k=5):
        """
        Nếu similarities được cung cấp, tính trung bình cosine similarity top-k
        """
        if self.similarities is None:
            raise ValueError("Bạn cần truyền similarities để tính metric này.")
        avg_sims = [np.mean(sim[:k]) for sim in self.similarities]
        return np.mean(avg_sims)

    def evaluate_all(self, k_list=[1, 5, 10]):
        """
        Trả về dict gồm tất cả metric cho các giá trị K
        """
        results = {}
        for k in k_list:
            results[f"Precision@{k}"] = self.precision_at_k(k)
            results[f"Recall@{k}"] = self.recall_at_k(k)
            results[f"nDCG@{k}"] = self.ndcg_at_k(k)
        results["mAP"] = self.mean_average_precision()
        if self.similarities is not None:
            results["AvgCosineSim@5"] = self.avg_cosine_similarity(5)
        return results
