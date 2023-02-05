import os
import sys
import math
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn import preprocessing
from scipy.signal import butter, lfilter


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def read_file(file):
    data = sio.loadmat(file)
    data = data['data']
    # print(data.shape)
    return data


def compute_DE(signal):
    variance = np.var(signal, ddof=1)  # 计算方差
    return math.log(2 * math.pi * math.e * variance) / 2  # 这就是论文里那个计算de的公式，这里好像用的是样本方差而不是标准差
# 关于np.var():https://blog.csdn.net/shiyuzuxiaqianli/article/details/105555055?spm=1001.2101.3001.6650.7&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Edefault-7-105555055-blog-81077663.pc_relevant_multi_platform_featuressortv2dupreplace&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Edefault-7-105555055-blog-81077663.pc_relevant_multi_platform_featuressortv2dupreplace&utm_relevant_index=10

def decompose(file):
    # trial*channel*sample
    start_index = 384  # 3s pre-trial signals
    data = read_file(file)
    shape = data.shape
    frequency = 128

    decomposed_de = np.empty([0, 4, 120])  # 关于np.empty：https://blog.csdn.net/SaintTsy/article/details/122596417
    # 其实就是用来初始化的
    base_DE = np.empty([0, 128])

    for trial in range(40):
        temp_base_DE = np.empty([0])
        temp_base_theta_DE = np.empty([0])
        temp_base_alpha_DE = np.empty([0])
        temp_base_beta_DE = np.empty([0])
        temp_base_gamma_DE = np.empty([0])

        temp_de = np.empty([0, 120])

        for channel in range(32):
            trial_signal = data[trial, channel, 384:]
            base_signal = data[trial, channel, :384]
            # ****************compute base DE****************
            # 先经过每个频段的带通滤波器，然后算相应频段内的de，
            # 准备期间的3s，0.5s为一段分为了6段，每个频段求了平均
            # 而真正任务期间的60s  将8064-384=7680个采样点以0.5s为一段(64个为一段)分为了120个，就是下面的120个循环，不求平均
            base_theta = butter_bandpass_filter(base_signal, 4, 8, frequency, order=3)
            base_alpha = butter_bandpass_filter(base_signal, 8, 14, frequency, order=3)
            base_beta = butter_bandpass_filter(base_signal, 14, 31, frequency, order=3)
            base_gamma = butter_bandpass_filter(base_signal, 31, 45, frequency, order=3)

            base_theta_DE = (compute_DE(base_theta[:64]) + compute_DE(base_theta[64:128]) + compute_DE(
                base_theta[128:192]) + compute_DE(base_theta[192:256]) + compute_DE(base_theta[256:320]) + compute_DE(
                base_theta[320:])) / 6  # 6段每段都计算一个de，然后取平均
            base_alpha_DE = (compute_DE(base_alpha[:64]) + compute_DE(base_alpha[64:128]) + compute_DE(
                base_alpha[128:192]) + compute_DE(base_theta[192:256]) + compute_DE(base_theta[256:320]) + compute_DE(
                base_theta[320:])) / 6
            base_beta_DE = (compute_DE(base_beta[:64]) + compute_DE(base_beta[64:128]) + compute_DE(
                base_beta[128:192]) + compute_DE(base_theta[192:256]) + compute_DE(base_theta[256:320]) + compute_DE(
                base_theta[320:])) / 6
            base_gamma_DE = (compute_DE(base_gamma[:64]) + compute_DE(base_gamma[64:128]) + compute_DE(
                base_gamma[128:192]) + compute_DE(base_theta[192:256]) + compute_DE(base_theta[256:320]) + compute_DE(
                base_theta[320:])) / 6

            temp_base_theta_DE = np.append(temp_base_theta_DE, base_theta_DE)  # 关于np.append函数：https://blog.csdn.net/weixin_42216109/article/details/93889047
            temp_base_gamma_DE = np.append(temp_base_gamma_DE, base_gamma_DE)
            temp_base_beta_DE = np.append(temp_base_beta_DE, base_beta_DE)
            temp_base_alpha_DE = np.append(temp_base_alpha_DE, base_alpha_DE)

            # ****************compute task DE****************
            theta = butter_bandpass_filter(trial_signal, 4, 8, frequency, order=3)
            alpha = butter_bandpass_filter(trial_signal, 8, 14, frequency, order=3)
            beta = butter_bandpass_filter(trial_signal, 14, 31, frequency, order=3)
            gamma = butter_bandpass_filter(trial_signal, 31, 45, frequency, order=3)

            DE_theta = np.zeros(shape=[0], dtype=float)
            DE_alpha = np.zeros(shape=[0], dtype=float)
            DE_beta = np.zeros(shape=[0], dtype=float)
            DE_gamma = np.zeros(shape=[0], dtype=float)

            for index in range(120):
                DE_theta = np.append(DE_theta, compute_DE(theta[index * 64:(index + 1) * 64]))
                DE_alpha = np.append(DE_alpha, compute_DE(alpha[index * 64:(index + 1) * 64]))
                DE_beta = np.append(DE_beta, compute_DE(beta[index * 64:(index + 1) * 64]))
                DE_gamma = np.append(DE_gamma, compute_DE(gamma[index * 64:(index + 1) * 64]))
            temp_de = np.vstack([temp_de, DE_theta])  # 纵向堆叠
            temp_de = np.vstack([temp_de, DE_alpha])
            temp_de = np.vstack([temp_de, DE_beta])
            temp_de = np.vstack([temp_de, DE_gamma])
        temp_trial_de = temp_de.reshape(-1, 4, 120)
        decomposed_de = np.vstack([decomposed_de, temp_trial_de])
        print(decomposed_de.shape)
        temp_base_DE = np.append(temp_base_theta_DE, temp_base_alpha_DE)
        temp_base_DE = np.append(temp_base_DE, temp_base_beta_DE)
        temp_base_DE = np.append(temp_base_DE, temp_base_gamma_DE)
        base_DE = np.vstack([base_DE, temp_base_DE])
    decomposed_de = decomposed_de.reshape(-1, 32, 4, 120).transpose([0, 3, 2, 1]).reshape(-1, 4, 32).reshape(-1, 128)
    print("base_DE shape:", base_DE.shape)  # 40*128
    print("trial_DE shape:", decomposed_de.shape)  # 4800*128,4800=40次trials*120段，64个点为一段
    return base_DE, decomposed_de


def get_labels(file):
    # 0 valence, 1 arousal, 2 dominance, 3 liking
    valence_labels = sio.loadmat(file)["labels"][:, 0] > 5  # valence labels
    arousal_labels = sio.loadmat(file)["labels"][:, 1] > 5  # arousal labels
    final_valence_labels = np.empty([0])
    final_arousal_labels = np.empty([0])
    # print("labels.shape:",len(valence_labels)) 就是40，没错的
    for i in range(len(valence_labels)):
        # 这里的valence_labels的长度应该是40，下面的循环的意思是构造的标签shape为40*120，并且每个120个点标签值是一样的，因为40次trials每次的trial的标签值相同
        for j in range(0, 120):
            final_valence_labels = np.append(final_valence_labels, valence_labels[i])
            final_arousal_labels = np.append(final_arousal_labels, arousal_labels[i])
    print("labels:", final_arousal_labels.shape)
    return final_arousal_labels, final_valence_labels


def wgn(x, snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)


def feature_normalize(data):
    mean = data[data.nonzero()].mean()
    sigma = data[data.nonzero()].std()
    data_normalized = data
    data_normalized[data_normalized.nonzero()] = (data_normalized[data_normalized.nonzero()] - mean) / sigma
    return data_normalized


if __name__ == '__main__':
    dataset_dir = "G:/For_EEG/DEAP数据集/data_preprocessed_matlab/"

    result_dir = "G:/For_EEG/EEG_code/4 （论文加代码）基于CNN和LSTM的脑电情绪识别（数据集为DEAP和seed）4D-CRNN/4D-CRNN-master/DEAP1/all_0.5/"
    if os.path.isdir(result_dir) == False:
        os.makedirs(result_dir)

    for file in os.listdir(dataset_dir):  # 文件夹里的每一个文件 进行decompose处理
        print("processing: ", file, "......")
        file_path = os.path.join(dataset_dir, file)
        base_DE, trial_DE = decompose(file_path)
        arousal_labels, valence_labels = get_labels(file_path)
        sio.savemat(result_dir + "DE_" + file,
                    {"base_data": base_DE, "data": trial_DE, "valence_labels": valence_labels,
                     "arousal_labels": arousal_labels})
