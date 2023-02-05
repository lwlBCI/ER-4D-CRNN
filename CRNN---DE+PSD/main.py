from eegemotion.data_load import load_data
from eegemotion.train import train
from eegemotion.utils import print_results

# specify dataset and model dirs
dataset_dir = 'G:/For_EEG/multi/multi-task-cnn-eeg-emotion-main/eegemotion/DEAP_data/with_base_0.5/'
#model_dir = 'G:/For_EEG/multi/multi-task-cnn-eeg-emotion-main/eegemotion/model_save'
#metrics_dir = 'G:/For_EEG/multi/multi-task-cnn-eeg-emotion-main/eegemotion/metrics_save'
model_dir='.'
metrics_dir='.'

img_size = img_rows, img_cols, num_chan = 8, 9, 8  # matrix shape of input data
number_of_inputs = 1  # how many frames is taken into account during one pass

features_type = 'multi'  # 'PSD', 'DE' or 'multi' be carefull with num_chan
num_classes = 2  # number of classes of input data
frames_per_subject = 4800  # how many frames per one subject
seed = 7  # random seed

dropout_rate = .2
model_name = 'MT_CNN'  # will be a filename part
lr_decay_factor = 0.5  # multiplixity factor of lr, where it's stucked in th plateu
lr_decay_patience = 5  # how many epochs without prgress before lr_decay，这可能是为了模型不持续提高准确率就停止训练的代码
epochs_n = 200  # maximum number of epochs
verbose = 0  # 0, 1 or 2

subject_n = 32

task = 'multi'  # 'valence', 'arousal' or 'multi'
fine_tuning = True  # fine tune to all subjects specifically

y_a_all_subject, y_v_all_subject, x_all_subject, all_subject_id = \
    load_data(dataset_dir, subject_n, img_size, number_of_inputs, features_type, num_classes, frames_per_subject, seed)

train(x_all_subject, y_a_all_subject, y_v_all_subject, all_subject_id, subject_n,
      dropout_rate, number_of_inputs, model_dir, metrics_dir, model_name, img_size,
      lr_decay_factor, lr_decay_patience, epochs_n, seed, verbose, task, fine_tuning)
# (153600,1,8,9,8),(153600,2),(1523600,2),(153600,), 32, 0.2, 1, model路径, metrics路径,
# “MT_CNN”, [8,9,8], 0.5, 5, 200, 7, 0, 'multi',True
# 要注意的是，train函数的最终模型输出中是有两个输出流，valence和arousal都有，所以这里的标签数据单独存在，并且都是153600*2
# 而且模型得到的loss是两个维度的加和，并且也可以单独输出每个情绪维度的loss


if fine_tuning:
    scores_dict = \
        {
            'before_fine_tuning': f'{metrics_dir}/{model_name}_scores_SD_before.pkl',
            'valence_task_fine_tuning': f'{metrics_dir}/{model_name}_valence_scores_SD.pkl',
            'arousal_fine_tuning': f'{metrics_dir}/{model_name}_arousal_scores_SD.pkl',
            'multi_fine_tuning': f'{metrics_dir}/{model_name}_multi_scores_SD.pkl'
        }  # dict with paths of scores files

    print_results(scores_dict, fine_tuning)
else:
    scores_dict = {'results': f'{metrics_dir}/{model_name}_scores_SI.pkl'}  # dict with path of scores file
    print_results(scores_dict, fine_tuning)
