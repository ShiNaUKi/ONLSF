#-*- encoding:UTF-8 -*-
import copy
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
# from load_data import load_data
from parser_OLWN import get_parse
import random
from online_learning import get_CAR
import matplotlib.pyplot as plt
import logging
import torch

def logger_config(log_path,logging_name):
    # 获取logger对象,取名
    logger = logging.getLogger(logging_name)
    # 输出DEBUG及以上级别的信息，针对所有输出的第一层过滤
    logger.setLevel(level=logging.DEBUG)
    # 获取文件日志句柄并设置日志级别，第二层过滤
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    # 生成并设置文件日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # console相当于控制台输出，handler文件输出。获取流句柄并设置日志级别，第二层过滤
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # 为logger对象添加句柄
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

def get_header(title, count):
    tmp  = []
    for i in range(count):
        tmp.append(title +str(i))
    return tmp


type_to_path = {
    "sym":"01_Sym",
    "asym":"02_ASym",
    "flip":"03_Flip"
}

buffer_size_table = {
    "breast":150,
    "hapt":1000,
    "wdbc":300,
    "wpbc":100,
    "dermatology":200,
}


if __name__ == "__main__":
    # 1.reading noisy-labeled data stream
    args = get_parse().parse_args()
    args.buffer_size = buffer_size_table[args.dataset]
    # reading data
    data = pd.read_csv(f"../dataset/{type_to_path[args.noise_type]}/{args.dataset}.csv", header=None)
    data = data.values

    log_path = f"{args.log_save_path}{args.dataset}_{args.noise_type}_.log"
    args.logger = logger_config(log_path, args.logging_name)


    X = data[:, :-2]
    Y_with_noise = data[:, -2]      # reading noisy labels from dataset
    Y0 = data[:, -1]
    Y0[Y0 != 1] = -1  # 标签形式替换为{-1, 1}
    Y_with_noise[Y_with_noise != 1] = -1


    # 运行多次求平均值
    for i in range(args.process_times):
        # args.logger.info(f"{args.dataset} round {i}")
        fobos_CAR, CAR_NF = get_CAR(X, Y_with_noise, Y0, args)

    print(f"Noise type = {args.noise_type}, dataset = {args.dataset},  CAR of ONLSF = {CAR_NF[-1]}")

    plt.plot(np.arange(len(fobos_CAR)), np.array(fobos_CAR), label='FOBOS')
    plt.plot(np.arange(len(CAR_NF)), np.array(CAR_NF), label='ONLSF')
    plt.xlabel(f"{args.noise_type}_{args.dataset}")
    plt.ylabel(f"CAR")
    plt.legend()
    plt.plot()
    plt.show()








