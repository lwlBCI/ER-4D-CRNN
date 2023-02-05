## 写在前面的话

:kissing_heart:从今天开始，我会陆陆续续更新关于EEG-Recognize+Deep Learning的文章与模型。从我个人的理解角度寻找到现有的结合这两方面的、有效地、结构清晰的文章，并附录清晰的解释与数据代码

:kissing_closed_eyes:情绪识别是脑电信号解码一个较为简单、但同时包容性最强的一个解码领域，之前的或现有的研究工作有很多，**公开的数据集**也有很多，如上述所言，在这里我只更新一些比较先进的、具有可解释性的模型。

:satisfied:值得注意的是，我并不会将复现文章所用到的数据全部上传，因为数据量实在太大且这些数据都是开源的可以去网站自行下载，提示：所有的数据都请不要商业化，仅供科研支持！如果您确实确实不想去网站下载(开源网站下载需要填写申请)，可以联系我提供！

## 4D-CRNN模型

:notes:这篇文章提出了一种新的方法，称为四维卷积递归神经网络，它将多通道EEG信号的频率、空间和时间信息明确地集成在一起，以提高基于EEG的情绪识别精度。 首先，为了保持脑电信号的这三种信息，我们将不同通道的**差分熵特征**转换为4D结构来训练深层模型。 然后，我们介绍了由**卷积神经网络（CNN）和带有长短期记忆（LSTM）单元的递归神经网络相结合的CRNN模型**。 CNN用于从4D输入的每个时间片中学习频率和空间信息，LSTM用于从CNN输出中提取时间相关性。LSTM最后一个节点的输出执行分类。 我们的模型在同一受试者的分割下，在SEED和DEAP数据集上都达到了最先进的性能。**实验结果表明，将脑电信号的频率、空间和时间信息整合起来进行情感识别是有效的。**

:star:**个人理解：**说白了就是这篇文章在当年应该是算情绪分类里面非常好的模型结构，从时域、频域和空间域三个角度来进行脑电信号信息的提取，这种模型结构的构建无论是从逻辑上还是最后的识别准确率上都显得非常高级，并且在Seed数据集和Deap数据集中都取得了非常好的效果。其中，在频域和空间域中使用卷积神经网络CNN来提取信息，使用LSTM从CNN输出中提取不同脑电信号切片中的时间相关性

<img src="https://cdn.jsdelivr.net/gh/lwlBCI/ER-4D-CRNN/images/image-20230203205555377.png" alt="image-20230203205555377" style="zoom:67%;" />

:point_right:为了同时整合EEG信号的频率、空间和时间特征，我们构建了一个包含这三种信息的4D特征结构，如上图所示。为了增加训练数据量，我们将原始EEG试验分为无重叠的Ts长片段，并给每个片段分配原始试验的标签。然后，对于每个片段，我们使用**巴特沃斯滤波器将其分解为四个频带**，包括α、β、γ和θ。其次，我们用0.5秒的窗口从每个频带提取**差分熵（DE）特征，这已被证明是情感识别的最稳定特征。**第三，我们将每个频带的DE特征组织为2-D图进行叠加，因此，每个段都可以表示为4D结构Xn∈R h x w x d x 2T，n∈1,,,N,其中h x w分别代表2D图的高度和宽度，N代表的是样本总数,d代表的是频带数(本文是4)，2T表示分段长度的两倍。**我个人对这一段话的理解是：**所谓的4D模型，就是首先**将数据从时域变换到频域，以0.5s的时域数据为切片，使用滤波器转换到4个频域带，进行DE特征的提取，然后将这些数据添加到了8 x 9的电极位置排列2D图中，这样得到的数据shape为：8 x 9 x 4，至于为什么是8 x 9的2D图，之后会有解释，2T的原因是因为以0.5s做时域数据切片，那么总时间为T的话总得时间段就是2T。**(注意本文档中的x均代表乘号)

:point_right:假设原始EEG片段表示为Sn∈R m x r x T，其中m和r分别表示电极数和原始脑电图信号的采样率。对于每个脑电片段，我们用0.5秒的窗口计算每个频带的DE特征，并用Z分数归一化对每个DE向量进行归一化。因此，原始EEG段被转换为DE段Pn∈R m x d x 2T，其中d表示频带数，在本文中我们将其设置为4。我个人对这一段话的理解是：从总的数据量来看，应该是 通道数 x 实验时间，假设我们现在的通道数为m，实验时间为T，当以0.5s作为时间切片，并且以四个频段来作为频域特征的提取时，总的数据量就是m x d x 2T，但是我们仔细想想其实和上面那一段是一模一样的，上面的h*w就是将通道信息转化为了一个2D图，以deap数据集举例：32个电极以一定的规律平铺在8 x 9的图中而已。下面这个图是一个例子，把62个电极平铺到8 x 9的2D图中

![image-20230203211533770](https://cdn.jsdelivr.net/gh/lwlBCI/ER-4D-CRNN/images/image-20230203211533770.png)

:pray:接下来需要说明的是：要想读懂这篇文章所要做的事情，需要结合代码来进行讲解，文章的大部分篇幅都放在了如何处理Deap与Seed数据集以及如何搭建网络的模型中，因此后文中我们将针对不同的数据集来进行分别解释。

### Deap数据集

#### 1D文件预处理

:smile:Deap数据集介绍：Deap数据集是一个用于分析人类情感状态的多模态数据集。记录了32名参与者的脑电图(EEG)和外周生理信号，每人观看40段一分钟长的音乐视频片段。参与者根据唤醒水平(arousal)、效价(valence)、喜欢/不喜欢(like/dislike)、主导(dominance)对每个视频进行1-9级评分。在与每个参与者对应的32.dat文件中，有两个数组：数据和标签。每个视频中有40个通道，依次有8064个EEG信号数据，总共形成322560个。标签的形状为40×4，其中4表示效价、唤醒、支配和喜爱(表2)。使用Python NumPy数组，并使用cPickle库和编码latin1加载.dat文件。

:smile:Deap数据集官方开源链接：http://www.eecs.qmul.ac.uk/mmv/datasets/deap/

| 数据Data   |           40 x 40 x 8064(Video x channel x data)            |
| ---------- | :---------------------------------------------------------: |
| 标签Labels | 40 x 4 x video x label(Valence、Arousal、Liking、Dominance) |
| 参与者编号 |                             32                              |
| 采样率     |                             128                             |

```python
"""
32位受试者：
    每位受试者40次trials：
        每位受试者32channels：
            每一个channel：
                Base_de:也就是数据集的基准部分，一共是3s的数据，384个点，0.5s为一段分为6段，每一段的处理：先从时域
                        数据中滤波4个频段，分别计算每个频段的DE值，然后取6段数据的平均值作为一个基准值，因此对于每
                        一位受试者来讲最终得到的数据格式为40*128，其中40指的是40次trials，128=32*4，指的是32个通道，
                        每个通道提取4个频段对应的4个de值
                Temp_de:这个部分指的是任务期间的数据，并且时间长度为60s，因为数据点数为60*128=7680，同样的，将60s数据
                        以0.5s为切片分为了120个循环，对于每个0.5s切片我们提取4个频段的信息并计算4个DE，这样对于每一位
                        受试者来说，40次trials，每次32个通道，每个通道是120次循环，每次循环得到4个DE，因此数据格式为
                        4800*128，4800=40*120,128=32*4.   
"""

```

:cherry_blossom:关于代码框里的这部分内容需要结合“DEAP_1D.py”来进行理解，代码中注释+代码框中的解释足够读者了解数据的生成过程。

#### 3D文件预处理

:full_moon_with_face:这部分对应的文件为："DEAP_1D_3D.py"

```python
# 由1D文件中我们得到的是temp_de=(4800,128),Base_de=40*128
# 在3D文件中，其实最主要的核心就是包含下面这两步：
# 1.将任务期间的temp_de-Base_de(基准阶段),相当于是刺激阶段减去平静期间
# 具体的循环为：
def get_dataset_deviation(trial_data, base_data):
    new_dataset = np.empty([0, 128])
    for i in range(0, 4800):
        base_index = i // 120
        # print(base_index)
        base_index = 39 if base_index == 40 else base_index  # 最后一个值4800//120=40,这句代码意思是把4800这个点base_index的值也归到39
        new_record = get_vector_deviation(trial_data[i], base_data[base_index]).reshape(1, 128)
        # print(new_record.shape)
        new_dataset = np.vstack([new_dataset, new_record])
    # print("new shape:",new_dataset.shape)
    return new_dataset
def get_vector_deviation(vector1, vector2):
    return vector1 - vector2
# 代码的逻辑是这样的，首先将temp_de中的4800个数据，每120个分配到base_index中，相当于就是Base_de中的40个trials的数据每一个 对应128个
# 0对应0-128,1对应1-128,...,39对应0-128（仔细理解10和11行代码）
2.将一维的数据转换为2D的矩阵形式
def pre_process(path, y_n):
    # DE feature vector dimension of each band
    data_3D = np.empty([0, 8, 9])
    sub_vector_len = 32
    trial_data, base_data, arousal_labels, valence_labels = read_file(path)
    if y_n == "yes":# 这里yes的判断是根据我们是否想用 任务-基准 的数据
        data = get_dataset_deviation(trial_data, base_data)
        data = preprocessing.scale(data, axis=1, with_mean=True, with_std=True, copy=True)
    else:
        data = preprocessing.scale(trial_data, axis=1, with_mean=True, with_std=True, copy=True)
    # convert 128 vector ---> 4*9*9 cube
    print(data.shape) # 4800*128
    for vector in data: #其实这里的vector就是(1,128)的格式,每一个vector都是4800里的一个数据
        for band in range(0, 4):
            data_2D_temp = data_1Dto2D(vector[band * sub_vector_len:(band + 1) * sub_vector_len])# 间隔为32
            data_2D_temp = data_2D_temp.reshape(1, 8, 9)
            # print("data_2d_temp shape:",data_2D_temp.shape)
            data_3D = np.vstack([data_3D, data_2D_temp])
    data_3D = data_3D.reshape(-1, 4, 8, 9)
    print("final data shape:", data_3D.shape)  # 4800,4,8,9
    return data_3D, arousal_labels, valence_labels
"""
35行代码是灵魂，，它将每一个vector(1*128)分为4份，θ-0：32，α-32：64，β-64：96，γ-96：128
    这里要抛出一个问题：为什么32个数据点是同一个频段范围，而不是4个点内四个频段都有？
    这是因为在前面的1D程序中，我们使用的是np.vstack纵向堆叠，也就是先堆叠32个电极的θ信息，再堆叠α信息，再堆叠β信息，再堆叠γ信息
    电极数是32，造成了堆叠的切片间隔是32
    然后我们将使用data_1Dto2D函数将每一个1*128的数据，以32个数据点为一个频段的方式铺到8*9的2D图中，128/32=4也就是说最终铺完的结构为4*8*9
    这也就是最终数据结构为4800*4*8*9的原因，nb！
"""
    
```

:bell:而关于训练文件“main_4DCRNN_DEAP.py”在此处不再进行细致的阐述，在我个人上传的代码文件中已经进行了细致的注释，主要是里面牵扯了一个交叉验证，总体思想就是先把(4800,4,8,9)的数据变成(4800,8,9,4),然后进行切片分割为(800,6,8,9,4),并且将以800进行5则交叉验证，分割为640:160，也就是4:1的比例，因此分割后的x_train.shape=(640,8,9,4),x_train[:,0/1/2/3/4/5].shape=(640,8,9,4)

### Seed数据集

#### 1D预处理文件

:fu:对应的文件为："SEED_1D_X89.py"

:point_right:官方开源链接：https://bcmi.sjtu.edu.cn/~seed/

```python
"""
首先要说明的是，seed数据集和deap数据集是不同的，这体现在：seed数据集的采样率是200，且包含15名受试者，每名受试者在三个不同的日期进行实验
每次试验进行15次trials，也就是说一共是：15*3=45次实验，每次实验15个trials，一共是45*15个trials
并且要特别说明的是：每个trials的数据长度也是不同的，因为截取的是电影片段，并不能保证单次实验的15个trials的每一个trial时间长度都一样
但是，每个人的三次实验，15次trials的顺序和长度一样，不同受试者的实验顺序和trials也一样，也就是说每个人的每次实验从数据量来讲都相同
并且，seed数据集没有基准部分不需要base文件，因此不管每次的trial多久，都是任务期间的数据
对于每次实验的15个trials来讲，数据长度分别是：47001,46601,41201,47601,37001,39001,47401,43201,53001,47401,47001,46601,47001,47601,41201
因此对于每位受试者的单次实验来说，包含15次trials，每个trial有62个通道，且数据长度为第6行的15个值，加起来是678800,
取0.5s为一个时间切片，采样率为200的前提下，就是100个点作为一个数据切片，那么678800/100=6788，(seed数据集选了5个频段的，但是最终只用了四个)也就是说，
每一个受试者的每次试验最终的数据片段总数为6788,62个通道，每个通道将产生5个频段的de值，关于频段滤波和de值计算和deap数据的处理过程相同
因此，单位受试者的单次实验数据shape为(6788,62,5)，总的数据量为(45*6788,62,5),15位受试者，每人三次实验
特别强调的是，如前面所说的，单位受试者的单次实验数据总量都是(6788,62,5)，因为每个人的实验都是相同的
对于labels来讲，seed数据集是一个三分类，高兴，中立，悲伤，所以labels的处理很简单，就是每次trial的label是相同的，所以数据切片的话也是单次trial里的所有数据片段的label都相同
labels的最终shape为(45*6788,3)
"""
# 在1D文件中，多了一个部分就是将62通道数据排列到8*9的电极中，代码如下：
X = np.load('G:/For_EEG/EEG_code/4 （论文加代码）基于CNN和LSTM的脑电情绪识别（数据集为DEAP和seed）4D-CRNN/4D-CRNN-master/SEED/DE0.5s/X_1D.npy')
y = np.load('G:/For_EEG/EEG_code/4 （论文加代码）基于CNN和LSTM的脑电情绪识别（数据集为DEAP和seed）4D-CRNN/4D-CRNN-master/SEED/DE0.5s/y.npy')
# 这两行代码的路径要换成读者自己的 X_1D.npy和y.npy文件是运行代码时自动生成的，可能这里有些读者会有误解，依据代码查看就可以了

# 生成8*9的矩阵形式，标准的10-20电极系统
X89 = np.zeros((len(y), 8, 9, 5))  # 305460,8,9,5，当然只有存数据的地方有值，其余的地方都是0

# 第0行的5个电极(看论文)
X89[:, 0, 2, :] = X[:, 3, :]  # 第0行、第2列是第3通道
X89[:, 0, 3:6, :] = X[:, 0:3, :] # 第0行、第3:6列的3个通道
X89[:, 0, 6, :] = X[:, 4, :] # 第0行、第62列是第4通道
for i in range(5):
    X89[:, i + 1, :, :] = X[:, 5 + i * 9:5 + (i + 1) * 9, :]  # 中间五行按照循环来
X89[:, 6, 1:8, :] = X[:, 50:57, :]  # 第6行的7个电极
X89[:, 7, 2:7, :] = X[:, 57:62, :]  # 第7行的5个电极
np.save("G:/For_EEG/EEG_code/4 （论文加代码）基于CNN和LSTM的脑电情绪识别（数据集为DEAP和seed）4D-CRNN/4D-CRNN-master/SEED/DE0.5s/X89.npy", X89)
# 因此，最终存成的格式为：305460,8,9,5
```

#### 3D预处理文件

:muscle:对应的文件为："SEED_X89_3D.py"

```python
"""
SEED数据集的3D文件与deap数据集的不同，主要是来进行训练前的数据结构清洗，通过for循环将45个实验样本的数据进行清理，每个
包含15次trials，每次trial的数据长度不同，因此通过for循环将单个trial数据长度判断的形式分割为78+77+68+79+61+65+79+72+
88+79+78+77+78+79+68=1126个片段，当然这里要解释一下为什么是这些数字，因为在1D文件中我们已经强调了，以0.5s的时间切片进行处理
那么15个trials的数据分别为470,466,412,476,370,390,474,432,530,474,470,466,470,476,412，因为在代码的循环中，当t=6时，我们以这些数据整除6的值作为单次trials的又一次片段分割，说白了这部分好像是为了后面模型训练的Concatenate，所以以6作为分割
所有的轮次加起来为1126，所以明白为啥是1126了吧？因为while循环条件里面的数字并不能整除6，只能取到整除位，所以加起来就是1126

"""

```

```python
# 训练模型的main_4DCRNN_SEED.py文件中也基本和deap是一样，不过要注意的是，seed数据集只有三分类，因此就是需要把每个人的三次实验样本的label
# 进行扩充，也就是1126*3，然后进行独热编码成3378*3的形式
```

### 模型结构

![image-20230203220350054](https://cdn.jsdelivr.net/gh/lwlBCI/ER-4D-CRNN/images/image-20230203220350054.png)

```python
"""

对于一个样本Xn（一个4D结构），通过CNN从它的每个时间切片中提取频率和空间信息。与传统的卷积层后面通常是池层的CNN不同，
我们只在最后一个卷积层之后添加池层。池操作用于以信息丢失为代价减少参数量。然而，样本Xn的2D贴图大小非常小，因此最好
保留所有信息，而不是合并信息以减少参数的数量。因此，我们只在最后一个卷积层之后使用一个池层。
CNN模块类似于Yang等人（2018a）中的CNN结构，但不同之处在于我们在最后一个卷积层之后添加了一个maxpooling层，原因
如上所述。如图4所示，它包含四个卷积层、一个最大池层和一个完全连接层。具体来说，第一卷积层（Conv1）有64个特征映射，
过滤器大小为5*5.
接下来的两个卷积层（Conv2、Conv3）分别有128和256个特征图，滤波器大小为4*4.第四卷积层（Conv4）包括
64个特征图，滤波器大小为1*1，用于融合前一卷积层的特征图。对于所有卷积层，应用零填充和校正线性单位（ReLU）激活函数。
卷积运算后，最大池层（Pool）的大小为2*2，步幅为2，以减少过拟合并增强网络的鲁棒性。最后，池层的输出被展平并馈送到具有
512个单元的完全连接层（FC）。
"""
# 代码结构如下：
# CNN结构：
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
# LSTM结构：
lstm_layer = LSTM(128, name='lstm')(out_all)
out_layer = Dense(2, activation='softmax', name='out')(lstm_layer)
model = Model([input_1, input_2, input_3, input_4, input_5, input_6], out_layer)
# model.summary()

# Compile model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.adam_v2.Adam(),
              metrics=['accuracy'])
# Fit the model
x_train = one_falx[train]
y_train = one_y[train]
model.fit([x_train[:, 0], x_train[:, 1], x_train[:, 2], x_train[:, 3], x_train[:, 4], x_train[:, 5]], y_train, epochs=100, batch_size=64, verbose=1)
# 这里的x_train[:,0/1/2/3/4/5]对应的ytrain是同一个值，因为每隔120个点才会变化一次标签，而6是120的公因子，所以这6个训练集的标签是一样的
# 下面的evaluate同理
# 训练过程中，5则交叉验证，每一则100个epochs，所以会输出五个准确率的值


```

### 如何使用

:sweat_drops:关于代码运行顺序这里简单做一下说明:

:droplet:"DEAP"文件夹中为使用Deap数据集的情绪识别，首先运行"DEAP_1D.py"然后运行"DEAP_1D_3D.py",最终执行"main_4DCRNN_DEAP.py"训练

:droplet:"DEAP_1D.py"中要注意152行的"dataset_dir"需要换成使用者自己的Deap数据集路径，154行result_dir需要换成使用者自己的文件夹路径，方便后续数据保存

:droplet:"DEAP_1D_3D.py"中73行的dataset_dir要换成"DEAP_1D.py"文件中的result_dir的路径,而76行与80行的result_dir需要使用者换成自己的路径，方便后续数据保存

:droplet:"main_4DCRNN_DEAP.py"中35行代码dataset_dir要换成"DEAP_1D_3D.py"中76行的result_dir

:droplet:同样的SEED文件夹中保存的是基于Seed数据集来进行情绪识别解码的运行文件，大致运行原理与Deap相同，但要**注意路径的设置要换成使用者自己的**，**代码均可运行，我个人都已经做了修改和注释，请读者放心食用！**

### Cite
Shen F, Dai G, Lin G, et al. EEG-based emotion recognition using 4D convolutional recurrent neural network[J]. Cognitive Neurodynamics, 2020, 14: 815-828.

最后，感谢[原作者](https://github.com/aug08)提供的思路与代码
