import numpy as np


def ndcg_metric(topN_dict, test_dict):
    ndcg = 0
    for key, topn_set in topN_dict.items():
        test_set = test_dict.get(key)
        dsct_list = [1 / np.log2(i + 1) for i in range(1, len(topn_set) + 1)]
        z_k = sum(dsct_list)
        if test_set is not None:
            mask = [0 if i not in test_set else 1 for i in topn_set]
            ndcg += sum(np.multiply(dsct_list, mask)) / z_k
    ndcg = ndcg / len(topN_dict.items())
    return ndcg
