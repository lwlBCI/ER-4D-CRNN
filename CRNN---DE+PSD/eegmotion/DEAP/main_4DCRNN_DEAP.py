import numpy as np
import scipy.io as sio
import os
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from keras.layers import Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import Flatten, Dense, Concatenate, Reshape, LSTM
from keras.models import Sequential, Model

import keras
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from keras import backend as K
import time
from sklearn.model_selection import StratifiedKFold


num_classes = 2  # 每个情绪类别是一个二分类，是或不是
batch_size = 128
img_rows, img_cols, num_chan = 8, 9, 4
flag = 'a'
t = 6

acc_list = []
std_list = []
all_acc = []

short_name = ['01', '02', '03', '04', '05', '06', '07', '08', '09',
              '10', '11', '12', '13', '14', '15', '16', '17', '18',
              '19', '20', '21', '22', '23', '24', '25', '26', '27',
              '28', '29', '30', '31', '32']

# 45次实验分别进行10倍交叉验证，取平均
dataset_dir = "G:/For_EEG/EEG_code/4 （论文加代码）基于CNN和LSTM的脑电情绪识别（数据集为DEAP和seed）4D-CRNN/4D-CRNN-master/DEAP1/with_base_0.5/"
for i in range(len(short_name)):
    K.clear_session()
    start = time.time()
    # print("\nprocessing: ", short_name[i], "......")
    file_path = os.path.join(dataset_dir, 'DE_s'+short_name[i])
    file = sio.loadmat(file_path)
    data = file['data']  # shape为4800 *4 * 8 * 9
    y_v = file['valence_labels'][0]
    y_a = file['arousal_labels'][0]
    y_v = to_categorical(y_v, num_classes)   # num_classes是类别个数，独热编码的结果是几个类别就是几
    y_a = to_categorical(y_a, num_classes)
    one_falx = data.transpose([0, 2, 3, 1])
    # one_falx = one_falx[:,:,:,2:4]
    one_falx = one_falx.reshape((-1, t, img_rows, img_cols, num_chan))  # 这里这样划分的目的是为了做交叉验证，分成了6份
    one_y_v = np.empty([0,2])
    one_y_a = np.empty([0,2])
    for j in range(int(len(y_a)//t)):
        one_y_v = np.vstack((one_y_v, y_v[j*t]))
        one_y_a = np.vstack((one_y_a, y_a[j*t]))
    # print(one_y_v.shape)
    # print(one_y_a.shape)
    # print(one_falx.shape)

    if flag=='v':
        one_y = one_y_v
    else:
        one_y = one_y_a

    seed = 7
    np.random.seed(seed)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)   # 交叉验证为5
    # StratifiedKFold这个函数好像是分层抽样，比如有1,2,3,4 4个数的时候，第一次抽1，2，下次就会抽3，4
    # 所以这里分为6则交叉验证，label也要按照6的分层来设计，也就是每隔6个，这6个相同，然后下一组6个也要相同
    # x_train, x_test, y_train, y_test = train_test_split(one_falx, one_y, test_size=0.1)
    cvscores = []

    # create model
    for train, test in kfold.split(one_falx, one_y.argmax(1)):
        img_size = (img_rows, img_cols, num_chan)
        # print("train.shape:",train.shape) 这里是640
        # print("train.shape:", test.shape) 这里是160,我明白了是在每一个800的组里面进行5则交叉验证
        def create_base_network(input_dim):
            seq = Sequential()
            seq.add(Conv2D(64, 5, activation='relu', padding='same', name='conv1', input_shape=input_dim))
            seq.add(Conv2D(128, 4, activation='relu', padding='same', name='conv2'))
            seq.add(Conv2D(256, 4, activation='relu', padding='same', name='conv3'))
            seq.add(Conv2D(64, 1, activation='relu', padding='same', name='conv4'))
            seq.add(MaxPooling2D(2, 2, name='pool1'))
            seq.add(Flatten(name='fla1'))
            seq.add(Dense(512, activation='relu', name='dense1'))
            seq.add(Reshape((1, 512), name='reshape'))
            return seq


        base_network = create_base_network(img_size)
        input_1 = Input(shape=img_size)
        input_2 = Input(shape=img_size)
        input_3 = Input(shape=img_size)
        input_4 = Input(shape=img_size)
        input_5 = Input(shape=img_size)
        input_6 = Input(shape=img_size)

        out_all = Concatenate(axis=1)([base_network(input_1), base_network(input_2), base_network(input_3), base_network(input_4), base_network(input_5), base_network(input_6)])
        lstm_layer = LSTM(128, name='lstm')(out_all)
        out_layer = Dense(2, activation='softmax', name='out')(lstm_layer)
        model = Model([input_1, input_2, input_3, input_4, input_5, input_6], out_layer)
        # model.summary()

        # Compile model
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.adam_v2.Adam(),
                      metrics=['accuracy'])
        # Fit the model
        # one_falx本身的shape是(800,6,8,9,4),而这里x_train的shape变成了(640,6,8,9,4)
        # 这个意思就是索引只在第一个维度进行索引，其他维度不变
        x_train = one_falx[train]
        # print(x_train.shape) 640,6,8,9,4
        # print(x_train[:, 0].shape,x_train[:, 1].shape,x_train[:, 2].shape,x_train[:, 3].shape,x_train[:, 4].shape,x_train[:, 5].shape)
        # 都是640,8,9,4

        y_train = one_y[train]
        #print(y_train.shape) 640,2

        model.fit([x_train[:, 0], x_train[:, 1], x_train[:, 2], x_train[:, 3], x_train[:, 4], x_train[:, 5]], y_train, epochs=100, batch_size=64, verbose=1)
        # 这里的x_train[:,0/1/2/3/4/5]对应的ytrain是同一个值，因为每隔120个点才会变化一次标签，而6是120的公因子，所以这6个训练集的标签是一样的
        # 下面的evaluate同理
        # 训练过程中，5则交叉验证，每一则100个epochs，所以会输出五个准确率的值


        # evaluate the model
        x_test = one_falx[test]
        y_test = one_y[test]
        scores = model.evaluate([x_test[:, 0], x_test[:, 1], x_test[:, 2], x_test[:, 3], x_test[:, 4], x_test[:, 5]], y_test, verbose=0)

        print("%.2f%%" % (scores[1] * 100))
        all_acc.append(scores[1] * 100)

    # print("all acc: {}".format(all_acc))
    print("mean acc: {}".format(np.mean(all_acc)))
    print("std acc: {}".format(np.std(all_acc)))
    acc_list.append(np.mean(all_acc))
    std_list.append(np.std(all_acc))
    print("进度： {}".format(short_name[i]))
    all_acc = []
    end = time.time()
    print("%.2f" % (end - start))
print('Acc_all: {}'.format(acc_list))
print('Std_all: {}'.format(std_list))
print('Acc_avg: {}'.format(np.mean(acc_list)))
print('Std_avg: {}'.format(np.mean(std_list)))