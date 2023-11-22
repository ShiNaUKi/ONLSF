#-*- encoding:UTF-8 -*-
import warnings
warnings.filterwarnings("ignore")
import copy
import numpy as np
import random as rd
import time
import math
import copy
from noisy_tool import  noise_main_multiprocess
import pandas as pd

from FOBOS import  FOBOS
from FOBOS import LR

import multiprocessing
from multiprocessing import Pool

import numba as nb

# metrics
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score

decay_choices = {"ionosphere": 4, "wbc": 2, "wdbc": 0, "german": 3, "diabetes": 4,"credit":4,"australian":4}
contribute_error_rates = {"ionosphere": 0.02, "wbc": 0.02, "wdbc": 0.02, "german": 0.005, "diabetes": 0.02,"credit":0.01,"australian":0.01}



def Gmean_(tn, tp, fn, fp):
    if (tp + fn) * (tn + fp) == 0:
        return 0
    else:
        return np.sqrt((tp / (tp + fn)) * (tn / (tn + fp)))

def F1_measure_(tn, tp, fn, fp):
    if (2*tp + fp + fn) == 0:
        return 0
    else:
        return (2.0*tp) / (2.0*tp + fp + fn)

def Precision_(tn, tp, fn, fp):
    if (tp + fp) == 0:
        return 0
    else:
        return 1.0 * tp / (tp + fp)

def Recall_(tn, tp, fn, fp):
    if (tp + fn) == 0:
        return 0
    else:
        return 1.0 * tp / (tp + fn)


# @nb.jit()
def get_CAR(X, Y_with_noise, Y0, args):
    n_samples = len(X)          # 特征个数
    n_features = len(X[0])      # 特征维度
    all_Y_NF = []

    # 自适应算法所需要的两个超参数decay_choice和contribute_error_rate
    if args.dataset in decay_choices.keys():
        print(f"{args.dataset} is in.")
        decay_choice = decay_choices[args.dataset]
        contribute_error_rate = contribute_error_rates[args.dataset]
    else:
        # 默认设置
        decay_choice = 2
        contribute_error_rate = 0.02

    # 3.训练
    # 三类算法对比, 无噪声标签、带噪声标签、标签经过NF处理

    if args.agorithm == "FOBOS":
        stream_classifier_noise,  \
             stream_classifier_NF = \
            FOBOS(K=5, alpha=0.01, lambda_=1., n_features=n_features, decisionFunc=LR), FOBOS(K=5, alpha=0.01, lambda_=1., n_features=n_features, decisionFunc=LR)


    # CAR and cost_time
    stream_CAR_noise, stream_CAR_NF = [], []
    # stream_Cost_times, stream_Cost_times_noise,  stream_Cost_times_NF = [], [], []


    num_buffer = math.ceil( n_samples / args.buffer_size)


    # F1-measure, G-mean, Precision, Recall
    # stream_classifier_F1s, stream_classifier_noise_F1s, stream_classifier_NF_F1s = [], [], []
    # stream_classifier_Gs, stream_classifier_noise_Gs, stream_classifier_NF_Gs = [], [], []
    # stream_classifier_Precisions, stream_classifier_noise_Precisions, stream_classifier_NF_Precisions = [], [], []
    # stream_classifier_Recalls,  stream_classifier_noise_Recalls, stream_classifier_NF_Recalls = [], [], []



    # label Refurishment
    # args.multiprocess
    manager = multiprocessing.Manager()
    res_queue = manager.Queue(num_buffer + 1)
    lock = manager.Lock()
    process_pool = Pool(args.multiprocess)

    for i in range(num_buffer):

        # 1.确定缓冲池
        args.logger.info(f"buffer {i}")
        buffer_start = i * args.buffer_size
        buffer_end = min((i + 1) *  args.buffer_size, n_samples)

        if i == num_buffer - 1:  # 获取最后一个缓冲区所有样本
            buffer_end = n_samples
        stream_X = X[buffer_start:buffer_end]            # 特征流
        # buffer_Y0 = Y0[buffer_start:buffer_end]          # 原始标签
        noise_Y = Y_with_noise[buffer_start:buffer_end]  # 带噪声标签
        # X, Y_with_noisy, index_labeled_samples, buffer_id = 0, args = None, lock = None, res_queue = None
        process_pool.apply_async(noise_main_multiprocess, args=(stream_X, copy.deepcopy(noise_Y), np.arange(len(stream_X)),
                                     i,args, lock, res_queue))
    process_pool.close()
    process_pool.join()

    tmp_data = []
    while not res_queue.empty():
        t = res_queue.get()
        tmp_data.append(t)
    tmp_data.sort(key=lambda x: x[0])
    # tmp_data = np.array(tmp_data)
    # tmp_data = tmp_data[np.argsort(tmp_data[:, 0])]


    for tt in tmp_data:
        all_Y_NF += tt[1].tolist()


    denoise_per_sample = 0.0
    # four style of noise processing function
    n_samples = len(all_Y_NF)

    for row in range(n_samples):
        # stream_classifier_predict_labels.append(predict_label)
        # tn, fp, fn, tp = confusion_matrix(true_labels, stream_classifier_predict_labels, labels=[-1, 1]).ravel()
        # stream_classifier_Gs.append(Gmean_(tn, tp, fn, fp))
        # stream_classifier_F1s.append(F1_measure_(tn, tp, fn, fp))
        # stream_classifier_Precisions.append(Precision_(tn, tp, fn, fp))
        # stream_classifier_Recalls.append(Recall_(tn, tp, fn, fp))

        # stream_classifier_Gs.append(0.0)
        # stream_classifier_F1s.append(0.0)
        # stream_classifier_Precisions.append(0.0)
        # stream_classifier_Recalls.append(0.0)

        # 2.带噪声标签的训练结果
        indices = [i for i in range(n_features)]
        x = X[row]
        y = Y_with_noise[row]
        y0 = Y0[row]
        if y0 == -1:    # 标签形式更换为{0, 1}
            y0 = 0
        if y == -1:
            y = 0
        # begin_time = time.time()  # 计算stream的所需时间
        p, decay, loss, w = stream_classifier_noise.fit(indices, x, y, decay_choice, contribute_error_rate)
        # stream_Cost_times_noise.append(time.time() - begin_time)
        # stream_Cost_times_noise.append(0.0)
        accuracy = [int(np.abs(y0 - p) < 0.5)]
        stream_CAR_noise.append(accuracy)

        # # F1-measure, G-mean, Precision, Recall指标计算
        # if accuracy[0] == 0:
        #     if y0 == 0:
        #         predict_label = 1
        #     else:
        #         predict_label = -1
        # else:
        #     if y0 == 0:
        #         predict_label = -1
        #     else:
        #         predict_label = 1
        #
        # stream_classifier_noise_predict_labels.append(predict_label)
        # tn, fp, fn, tp = confusion_matrix(true_labels, stream_classifier_noise_predict_labels, labels=[-1, 1]).ravel()
        # stream_classifier_noise_Gs.append(Gmean_(tn, tp, fn, fp))
        # stream_classifier_noise_F1s.append(F1_measure_(tn, tp, fn, fp))
        # stream_classifier_noise_Precisions.append(Precision_(tn, tp, fn, fp))
        # stream_classifier_noise_Recalls.append(Recall_(tn, tp, fn, fp))
        # stream_classifier_noise_Gs.append(0.0)
        # stream_classifier_noise_F1s.append(0.0)
        # stream_classifier_noise_Precisions.append(0.0)
        # stream_classifier_noise_Recalls.append(0.0)


        # 3.噪声经过处理
        if args.agorithm == "FOBOS":
            indices = [i for i in range(n_features)]
            x = X[row]
            y = all_Y_NF[row]
            y0 = Y0[row]
            if y0 == -1:
                y0 = 0
            if y == -1:
                y = 0
            # begin_time = time.time()
            p, decay, loss, w = stream_classifier_NF.fit(indices, x, y, decay_choice, contribute_error_rate)
            # c_t = time.time() - begin_time
            # stream_Cost_times_NF.append(0.0)
            accuracy = [int(np.abs(y0 - p) < 0.5)]
            stream_CAR_NF.append(accuracy)





    stream_CAR_noise = np.cumsum(stream_CAR_noise) / (np.arange(len(stream_CAR_noise)) + 1.0)
    stream_CAR_NF = np.cumsum(stream_CAR_NF) / (np.arange(len(stream_CAR_NF)) + 1.0)



    return stream_CAR_noise.tolist(), stream_CAR_NF.tolist()