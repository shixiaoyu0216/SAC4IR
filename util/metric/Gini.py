import numpy as np


def MaxMinNormalization(x_list):
    x_max = max(x_list)
    x_min = min(x_list)
    for i in range(len(x_list)):
        x_list[i] = (x_list[i] - x_min) / (x_max - x_min)
    return x_list


def Gini(p_dic):
    p_list = list(p_dic.values())
    cum = np.cumsum(sorted(np.append(p_list, 0)))
    sum = cum[-1]
    x = np.array(range(len(cum))) / len(p_list)
    y = cum / sum
    B = np.trapz(y, x=x)
    A = 0.5 - B
    G = A / (A + B)
    return G
