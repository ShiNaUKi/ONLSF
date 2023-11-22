#-*- encoding:UTF-8 -*-
import argparse

def get_parse():
    parser = argparse.ArgumentParser(description="Run ONLSF")

    # 1.数据集的设定
    parser.add_argument('--dataset', nargs='?', default='wdbc',
                        help='Choose a dataset from {'
                             'Sym : [breast, hapt, wdbc, wpbc],'
                             'ASym: [dermatology, hapt, wdbc]'
                             'Flip: [breast, hapt, wdbc]}')
    # 2.
    # type of noise, sym of asym
    parser.add_argument("--noise_type", type=str, default="sym",
                        help="type of noise, (sym, asym, flip)")
    # sym
    parser.add_argument('--sym_noise_rate', type=float, default=0.4,
                        help='the rate of label corruption')
    # asym
    parser.add_argument('--asym_noise_rate1', type=float, default=0.3,
                        help='the rate of label corruption')
    parser.add_argument('--asym_noise_rate2', type=float, default=0.5,
                        help='the rate of label corruption')
    parser.add_argument('--Flip_noise_rate', type=float, default=0.4,
                        help='the rate of label corruption')

    # 3.online learning
    parser.add_argument('--process_times', type=int, default=1,
                        help='num of runing times')



    # 邻域阈值p1, p2
    parser.add_argument('--p1_low', type=int, default=7,
                        help='p1_low')
    parser.add_argument('--p1_high', type=int, default=15,
                        help='num of buffer')
    parser.add_argument('--p2_low', type=int, default=30,
                        help='q1_low')
    parser.add_argument('--p3_low', type=int, default=4,
                        help='p3_low')


    # density
    parser.add_argument('--q1_low', type=int, default=90,
                        help='q1_low')
    parser.add_argument('--q2_low', type=int, default=85,
                        help='q2_low')

    # dc
    parser.add_argument('--dc_ratio', type=float, default=0.01,
                        help='dc_ratio')

    # flag of refurishment
    parser.add_argument('--r_threshold', type=float, default=0.1,
                        help='r_threshold')

    # 特征比例和在线学习缓冲区大小
    # parser.add_argument('--feature_rate', type=float, default=0.1,
    #                     help='rate of rest features, [0.1, 0.25, 0.5]')

    # 算法种类:
    parser.add_argument('--agorithm', type=str, default="FOBOS",
                        help='online learning agorithm, such like FOBOS,  FTRL,  OGD, RDA, TG, FTRL_ADP')

    # 结果保存路径
    parser.add_argument('--log_save_path', type=str, default="../log/",
                        help='path to save res')
    # KNN
    parser.add_argument('--K', type=int, default=3,
                        help='num of node to filter')
    # log 参数
    parser.add_argument("--logging_name", type=str, default="log")

    # shuffle
    parser.add_argument("--shuffle", type=int, default=0)
    parser.add_argument("--norm", type=int, default=0)
    parser.add_argument("--multiprocess", type=int, default=2)
    return parser
