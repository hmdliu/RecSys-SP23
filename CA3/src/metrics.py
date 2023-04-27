
from math import log2
from functools import partial

def compute_metrics(pred_list, round_digits=None):
    metrics = {}
    for m, f in METRIC_DICT.items():
        if round_digits is None:
            metrics[m] = f(pred_list)
        else:
            metrics[m] = round(f(pred_list), round_digits)
    return metrics

def recall(pred_list, k=10):
    val = count = 0
    for pred, label in pred_list:
        if len(label) > 0:          # for male/female group eval
            pred = set(map(lambda p: p.item(), pred[:k]))
            val += len(pred & label) / len(label)
            count += 1
    return val / count

def ndcg(pred_list, k=10):
    val = count = 0
    for pred, label in pred_list:
        if len(label) > 0:          # for male/female group eval
            dcg = sum([(pred[i].item() in label)/log2(i+2) for i in range(k)])
            idcg = sum([1/log2(i+2) for i in range(min(len(label), k))])
            val += dcg / idcg
            count += 1
    return val / count

METRIC_DICT = {
    'Recall@10': partial(recall, k=10),
    'Recall@50': partial(recall, k=50),
    'NDCG@10': partial(ndcg, k=10),
    'NDCG@50': partial(ndcg, k=50),
}
