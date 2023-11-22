#-*- encoding:UTF-8 -*-
import numpy as np
import copy
from collections import Counter
import math
from scipy.spatial.distance import cdist
import pandas as pd
import random as rd

def EuclideanDist(X1, X2):
    """欧式距离"""
    # np.set_printoptions(precision=5)
    m1, n1 = X1.shape
    m2, n2 = X2.shape
    if n1 != n2:
        print("Error from EuclideanDist, columns of X1 and columns of X2 isn't aligned")
        exit(-1)
    else:
        dist = cdist(X1,X2,metric='euclidean')

    return dist

def get_dc(X, index_labeled_samples, args):
    """类似于DPC, 求截止距离"""
    # u = 0.01
    u = args.dc_ratio
    v = 1.5
    m = X.shape[0]
    N = len(index_labeled_samples)

    # 计算距离
    # u相当于dc的比例，v为每个点的影响范围系数，v*dc
    # dcRatio = 0.01;
    # dc = dcRatio * maxDist

    # 所有数据的欧式距离
    dist = EuclideanDist(X, X)
    dist[np.where(dist == 0)] = 0.01

    # 带标签数据的欧式距离
    dist_L = EuclideanDist(X[index_labeled_samples,:], X[index_labeled_samples,:])  # 各带标签样本之间的距离
    dist_L[np.where(dist_L == 0)] = 0.01

    # maxDist = np.max(dist)
    # dc = dcRatio * maxDist
    # dcRation = 0.01
    dis = np.zeros((math.ceil((m - 1) * m / 2), 1))      # 这是所有距离
    num = 0                                 # num为所有距离个数

    for i in range(m):
        for j in range(i+1, m):
            dis[num] = dist[i, j]
            num = num + 1

    # 降维
    dis = dis.reshape(-1)


    position = round(u * num) - 1     # 取第u % 的点为截止距离, 类似于DPC

    sda = np.sort(dis)
    dc = sda[position]                      # 截止距离
    # dc = 0.4 * np.max(sda)
    doubt = []
    rest = list(range(N))                   # 剩余未处理的样本

    return dc,  rest, dist_L, sda, v

def ConfidenceCenter(Y, labels, N, dc, dist_L, args):
    wrong = []
    init = []
    block = dict()
    major = np.zeros((N, 3))     # label types, num, ratio
    # rule1, rule2, rule3 = [], [], []
    # wrong1, wrong2, wrong3 = [], [], []
    # 统计节点的邻节点情况
    for i in range(N):
        current_dist = dist_L[i, :]                              # 当前访问的带标签样本与其他样本之间的距离
        block[i] = np.where(current_dist < dc)[0].tolist()       # 距离小于dc的样本为当前样本的邻居（含自身）
        indexs_from_block = block[i]
        table = Counter(Y[labels[indexs_from_block]])           # 统计各标签出现的频率

        if len(table) == 0:     # 无结果
            major[i, 2] = 0
            major[i, 1] = -1
            major[i, 0] = -2
            continue

        idx = list(table.keys())
        vals = list(table.values())
        idMax = vals.index(max(vals))
        major[i, 2] = vals[idMax] * 1.0 / np.sum(vals) * 100  # 记录多数相同标签的比例
        major[i, 1] = vals[idMax]                            # 记录多数相同标签的数量
        major[i, 0] = idx[idMax]                             # 记录多数相同标签类别



    for i in range(N):
        # 参数beta1, 控制邻居最多标签的种类
        if major[i, 1] >= args.p1_low:
            # 一定的次数, 与一定的比例, 才可被选为init节点
            if (major[i, 1] <= args.p1_high and major[i, 2] >= args.q1_low):   #标签占一定比例，出现一定次数, 才能加入Init
                init.append(i)
                if Y[labels[i]] != major[i, 0]:         # 根据样本的邻居标签, 修改标签
                    Y[labels[i]] = major[i, 0]   #节点标签, 与邻居标签不一致,修改Label
                    wrong.append(i)

            elif (major[i, 1] > args.p2_low and major[i,2] >= args.q2_low):
                init.append(i)
                if Y[labels[i]] != major[i, 0]:  # 根据样本的邻居标签, 修改标签
                    Y[labels[i]] = major[i, 0]  # 节点标签, 与邻居标签不一致,修改Label
                    wrong.append(i)
        # 某一类特殊情况
        elif major[i, 2] == 100 and major[i, 1] >= args.p3_low:
            init.append(i)
    return init, Y, wrong, #rule1, rule2, rule3, wrong1, wrong2, wrong3

def Label_Furbishment(dist_L, v, dc, sda, init, rest, wrong, doubt, record, labels, Y, args):
    # v, dc:  These parameters control the limitation of KNN trusted set.
    # sda : distance matrix of all nodes.
    # init : Trusted set
    # label
    if type(init) != list: # 单个初始样本,
        R = [init]
    else:
        R = init           # 多个初始样本
    num = len(sda)      # sort dist, 无重复的线性排序

    # 求出每个与init节点, 距离小于2*dc的节点, 作为备选节点
    block = []
    init_dist = dist_L[R, :]
    for i in init_dist:
        block += np.where(i < 2*dc)[0].tolist()
    block = list(set(block) & set(rest))        # 求block中未信任节点

    while len(rest) > 0:
        dist1 = dist_L[R][:, rest]              #
        dist2 = np.sum(dist1, axis=0)

        q = np.where(dist2 == np.min(dist2))[0]    #寻找距离与已预测样本之和最小的样本
        p = rest[q[0]]                             #这是待预测样本
        dist3 = dist1[:, q[0]]


        idx = np.argsort(dist3)              # 被选中节点与其他init节点的距离
        row = np.where(dist3 < v * dc)[0]    # 提取能影响到待预测样本的已处理样本
        num_train = len(row)

        #  Top K nodes
        train = []
        if args.K == -1:
            top_K = num_train
        else:
            top_K = args.K
        if num_train > top_K:
            for w in range(top_K):
                train.append(R[idx[w]])
        elif num_train > 0 and num_train <= top_K:
            for w in range(num_train):
                train.append(R[idx[w]])

        # 不存在距离小于v*dc的点, 但存在小于2*dc的点, 从1.5*dc -> 2.0 * dc
        if len(train) == 0  and len(block) > 0:
            dist1 = dist_L[R][:, block]
            dist2 = dist1.sum(axis=0)
            q = np.where(dist2 == min(dist2))[0]
            p = block[q[0]]

            dist3 = dist1[:, q[0]]
            idx = np.argsort(dist3)
            row =np.where(dist3 < v * dc)[0] # 提取能影响到待预测样本的已处理样本

            num_train = len(row)

            # 最多只取3个样本
            train = []
            if args.K == -1:
                top_K = num_train
            else:
                top_K = args.K
            if num_train > top_K:
                for w in range(top_K):
                    train.append(R[idx[w]])
            elif num_train > 0 and num_train <= top_K:
                for w in range(num_train):
                    train.append(R[idx[w]])
            else:
                train = []


        if len(train) == 0: # 不存在与init节点, 1.5dc 和 2dc节点距离的节点
            break



            # 初始化概率
        p_pos = 0
        p_neg = 0

        # 根据距离加入权重
        r = dist_L[p][train]    # 提取训练样本与当前样本的距离
        weight = np.exp(1. / r)
        weight = weight / sum(weight)

        # 保留一定位数小数, 能找到对应点的下标
        sda = np.around(sda, 9)
        r = np.around(r, 9)

        # 预测附近样本的准确性（对于单个样本）
        for j in range(len(train)):
            if Y[labels[train[j]]] == 1:
                p_pos = p_pos + weight[j]
            else:
                p_neg = p_neg + weight[j]



        # 取预测概率较大的那个
        esti = np.sign(p_pos - p_neg)

        if Y[labels[p]] == esti:
            R.append(p)
        else:
            if np.abs(p_pos - p_neg) > args.r_threshold:
                wrong.append(p)
                R.append(p)
                Y[labels[p]] = esti
            else:
                doubt.append(p)

        # remove processed nodes from block and rest
        if p in block:
            block.remove(p)

        if p in rest:
            rest.remove(p)

    record += R
    record = list(set(record))

    return record, rest, wrong, doubt, Y

def noise_main_multiprocess(X, Y_with_noisy, index_labeled_samples, buffer_id=0, args=None, lock=None, res_queue=None):
    # 1. neighbor distance
    dc,  rest, dist_L, sda, v = get_dc(X, index_labeled_samples, args)

    # 2. Center
    init, Y_with_noisy, wrong = ConfidenceCenter(Y_with_noisy, index_labeled_samples, len(index_labeled_samples), dc, dist_L, args)

    # 3. labels refurbishment
    record = init
    rest = list(set(rest) - set(init))  # 从rest中删除掉init节点
    doubt = []
    record, rest, wrong, doubt, Y = Label_Furbishment(dist_L, v, dc, sda, init, rest, wrong, doubt, record, index_labeled_samples, Y_with_noisy, args)


    record += rest
    record += doubt

    lock.acquire()
    res_queue.put([buffer_id, Y_with_noisy])
    lock.release()
    return
