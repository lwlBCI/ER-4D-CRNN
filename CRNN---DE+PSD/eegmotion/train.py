import os
import pickle
from shutil import copyfile

import numpy as np
from sklearn.model_selection import StratifiedKFold

import tensorflow as tf
import keras
from keras import backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

from eegemotion.utils import ReduceLRBacktrack, get_sample_weights
from eegemotion.model import create_MT_CNN


def train(x_all_subject, y_a_all_subject, y_v_all_subject, all_subject_id,
          subject_n=32, dropout_rate=.2, number_of_inputs=1, model_dir='.', metrics_dir='.',
          model_name='MT_CNN', img_size=(8, 9, 8), lr_decay_factor=0.5,
          lr_decay_patience=5, epochs_n=200, seed=7, verbose=0, task='multi',
          fine_tuning=True):
    # (153600,1,8,9,8),(153600,2),(1523600,2),(153600,), 32, 0.2, 1, model路径, metrics路径,
    # “MT_CNN”, [8,9,8], 0.5, 5, 200, 7, 0, 'multi',True
    if task == 'multi':
        loss_weights = [1, 1]
    elif task == 'valence':
        loss_weights = [1, 0]
    elif task == 'arousal':
        loss_weights = [0, 1]
    else:
        raise NotImplementedError

    model_checkpoint_path_SD = f'{model_dir}/{model_name}-for-test.hdf5' # 第一种文件类型

    lrate_silent = lambda: ReduceLRBacktrack(best_path=model_checkpoint_path_SD,
                                             monitor='val_loss', patience=lr_decay_patience,
                                             factor=lr_decay_factor, verbose=verbose)

    lrate = lambda model_checkpoint_path: ReduceLROnPlateau(best_path=model_checkpoint_path,
                                                            monitor="val_loss", patience=5,
                                                            factor=0.5, verbose=1)
    # 关于ModelCheckpoint：https://blog.csdn.net/qq_28949847/article/details/109067395
    # 关于ReduceLROnPlateau和EarlyStopping配合使用：https://blog.csdn.net/weiyunna/article/details/113876631
    # 这里的lrate路径和save_model的路径(model_checkpoint_path)只是一种替代作用，下面使用的时候会用真实的路径来替代
    es = lambda: EarlyStopping(monitor='val_loss', mode='min',
                               verbose=1, patience=16)

    save_model_for_test = lambda: ModelCheckpoint(
        model_checkpoint_path_SD,
        monitor='val_loss', save_best_only=True,
        save_weights_only=False,  # 注意
        mode='min', verbose=verbose)

    save_model = lambda model_checkpoint_path: ModelCheckpoint(model_checkpoint_path,
                                                               monitor='val_loss', save_best_only=True,
                                                               save_weights_only=False,  # 注意
                                                               mode='min')

    scores_subject_independent_list = []

    np.random.seed(seed)
    n_splits = 5
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train, test) in enumerate(kfold.split(x_all_subject, y_a_all_subject.argmax(1))):
        # 所有的数据153600拿来做五则交叉
        print(f'\n\nFold {fold + 1}/{n_splits}\n\n')
        # if fold > 1:
        #    continue
        K.clear_session()

        model_checkpoint_path_SI_unique = f'{model_dir}/{model_name}-weight_AV-fold{fold + 1:02d}' + \
                                          '-epoch{epoch:02d}-loss{val_loss:.2f}-A_accuracy{val_out_a_accuracy:.4f}-V_accuracy{val_out_v_accuracy:.4f}.hdf5'
        model_checkpoint_path_SI_for_load = f'{model_dir}/{model_name}-weight_AV-fold{fold + 1:02d}.hdf5'
        # 第二种+第三种文件类型
        # 第二种是用来显示5则交叉验证中其中一次内所有epoch的损、准确率等信息
        # 第三种是标志着5则交叉验证某一则的开始
        model = create_MT_CNN(img_size, dropout_rate, number_of_inputs)

        model.compile(loss=[keras.losses.categorical_crossentropy, keras.losses.categorical_crossentropy],
                      loss_weights=loss_weights,
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      metrics=['accuracy'])  # 对应于模型的每个输出的损失的权重。loss_weights是一个列表，对应于每个输出的权重，默认1
        # 这里需要解释的是loss_weight这个参数，它对应的是每个损失的权重，在这里我们是两个输出流，一个arousal一个valence
        # 因此对应的列表里为[1,1],代表二者的权重比例相同，其实后面的loss我们也可以看到，总loss=arousal_loss+valence_loss

        # Fit the model
        x_train = x_all_subject[train]
        y_train_a = y_a_all_subject[train]
        y_train_v = y_v_all_subject[train]
        # print("x_train.shape):",x_train.shape)(122880, 1, 8, 9, 8)
        subject_id_train = all_subject_id[train]

        x_test = x_all_subject[test]
        y_test_a = y_a_all_subject[test]
        y_test_v = y_v_all_subject[test]

        subject_id_test = all_subject_id[test]

        sample_weights = get_sample_weights([y_train_v, y_train_a])  # 解决样本不均衡的问题，但这里好像没有不均衡吧？不懂

        hist = model.fit([x_train[:, i] for i in range(x_train.shape[1])],
                         [y_train_v, y_train_a], epochs=epochs_n,
                         batch_size=64, verbose=1,
                         sample_weight=sample_weights,
                         callbacks=[save_model(model_checkpoint_path_SI_unique),  # 如前文，给定路径
                                    save_model(model_checkpoint_path_SI_for_load),  # 如前文，给定路径
                                    lrate(model_checkpoint_path_SI_for_load),   # 如前文，给定路径
                                    es()],
                         validation_data=([x_test[:, i] for i in range(x_test.shape[1])],
                                          [y_test_v, y_test_a]))

        scores = model.evaluate([x_test[:, i] for i in range(x_test.shape[1])],
                                [y_test_v, y_test_a], verbose=1)

        scores_subject_independent_list.append(scores)

        if fine_tuning:
            multi_scores_subject_dependent_list = []
            valence_scores_subject_dependent_list = []
            arousal_scores_subject_dependent_list = []
            scores_subject_dependent_list_before = []

            multi_scores_subject_dependent_per_fold = []
            valence_scores_subject_dependent_per_fold = []
            arousal_scores_subject_dependent_per_fold = []
            scores_subject_dependent_per_fold_before = []

            metrics = ['loss', 'valence loss', 'arousal loss', 'valence acc', 'arousal acc']

            for i in range(subject_n):
                short_name = f'{i + 1:02}'
                K.clear_session()
                print("\nprocessing: ", short_name, "......")

                model = create_MT_CNN(img_size, dropout_rate, number_of_inputs)

                model.compile(loss=[keras.losses.categorical_crossentropy, keras.losses.categorical_crossentropy],
                              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001 / 8),
                              metrics=['accuracy'])

                model.load_weights(model_checkpoint_path_SI_for_load)  # 加载权重

                # Fit the model
                x_train_for_subject = x_train[subject_id_train == i]
                y_train_v_for_subject = y_train_v[subject_id_train == i]
                y_train_a_for_subject = y_train_a[subject_id_train == i]

                x_test_for_subject = x_test[subject_id_test == i]
                y_test_v_for_subject = y_test_v[subject_id_test == i]
                y_test_a_for_subject = y_test_a[subject_id_test == i]

                scores_for_subject = model.evaluate(
                    [x_test_for_subject[:, i] for i in range(x_test_for_subject.shape[1])],
                    [y_test_v_for_subject, y_test_a_for_subject], verbose=verbose)

                best_loss_SI = scores_for_subject[:3] # 这里的loss应该是包含了三个维度，一个是总loss一个是avalence_loss,另一个是arousal_loss
                print("best_loss:",best_loss_SI.shape)
                scores_subject_dependent_per_fold_before.append(scores_for_subject[-2:])

                print('Before fine-tuning:', list(zip(metrics, [round(score, 6) for score in scores_for_subject])))

                # AV Multi-task fine-tuning
                copyfile(model_checkpoint_path_SI_for_load, model_checkpoint_path_SD)
                callbacks = [lrate_silent(),
                             save_model_for_test(),
                             es()]  # 这里的callback用的是第一种文件类型的路径，也就是test路径！

                for callback in callbacks:
                    callback.best = best_loss_SI[0]

                sample_weights = get_sample_weights([y_train_v_for_subject, y_train_a_for_subject])
                hist_for_subject = model.fit([x_train_for_subject[:, i] for i in range(x_train_for_subject.shape[1])],
                                             [y_train_v_for_subject, y_train_a_for_subject], epochs=epochs_n,
                                             sample_weight=sample_weights,
                                             batch_size=64, verbose=verbose,
                                             callbacks=callbacks,
                                             validation_data=(
                                             [x_test_for_subject[:, i] for i in range(x_test_for_subject.shape[1])],
                                             [y_test_v_for_subject, y_test_a_for_subject]))

                model.load_weights(model_checkpoint_path_SD)  # 这里加载的权重是针对每个受试者而言的，并不是上面把153600条数据全部用于训练的那个权重

                # 用这个模型(单个受试者)的参数进行测试
                scores_for_subject = model.evaluate(
                    [x_test_for_subject[:, i] for i in range(x_test_for_subject.shape[1])],
                    [y_test_v_for_subject, y_test_a_for_subject], verbose=verbose)

                multi_scores_subject_dependent_per_fold.append(scores_for_subject[-2:])
                print('After fine-tuning on Multi-Task',
                      list(zip(metrics, [round(score, 6) for score in scores_for_subject])))

                # For Valence
                K.clear_session()
                model = create_MT_CNN(img_size, dropout_rate, number_of_inputs)

                model.compile(loss=keras.losses.categorical_crossentropy, loss_weights=[1, 0],  # 这里只看valence
                              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001 / 4),
                              metrics=['accuracy'])

                model.load_weights(model_checkpoint_path_SI_for_load)

                copyfile(model_checkpoint_path_SI_for_load, model_checkpoint_path_SD)
                callbacks = [lrate_silent(),
                             save_model_for_test(),
                             es()]
                for callback in callbacks:
                    callback.best = best_loss_SI[1]

                sample_weights = get_sample_weights([y_train_v_for_subject])
                hist_for_subject = model.fit([x_train_for_subject[:, i] for i in range(x_train_for_subject.shape[1])],
                                             [y_train_v_for_subject, y_train_a_for_subject], epochs=epochs_n,
                                             sample_weight=sample_weights,
                                             batch_size=64, verbose=verbose,
                                             callbacks=callbacks,
                                             validation_data=(
                                             [x_test_for_subject[:, i] for i in range(x_test_for_subject.shape[1])],
                                             [y_test_v_for_subject, y_test_a_for_subject]))

                model.load_weights(model_checkpoint_path_SD)

                scores_for_subject = model.evaluate(
                    [x_test_for_subject[:, i] for i in range(x_test_for_subject.shape[1])],
                    [y_test_v_for_subject, y_test_a_for_subject], verbose=verbose)

                valence_scores_subject_dependent_per_fold.append(scores_for_subject[-2:])
                print('After fine-tuning on VALENCE',
                      list(zip(metrics, [round(score, 6) for score in scores_for_subject])))

                # Arousal
                K.clear_session()
                model = create_MT_CNN(img_size, dropout_rate, number_of_inputs)

                model.compile(loss=keras.losses.categorical_crossentropy, loss_weights=[0, 1],
                              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001 / 4),
                              metrics=['accuracy'])

                model.load_weights(model_checkpoint_path_SI_for_load)

                copyfile(model_checkpoint_path_SI_for_load, model_checkpoint_path_SD)
                callbacks = [lrate_silent(),
                             save_model_for_test(),
                             es()]
                for callback in callbacks:
                    callback.best = best_loss_SI[2]

                sample_weights = get_sample_weights([y_train_a_for_subject])
                hist_for_subject = model.fit([x_train_for_subject[:, i] for i in range(x_train_for_subject.shape[1])],
                                             [y_train_v_for_subject, y_train_a_for_subject], epochs=epochs_n,
                                             sample_weight=sample_weights,
                                             batch_size=64, verbose=verbose,
                                             callbacks=callbacks,
                                             validation_data=(
                                             [x_test_for_subject[:, i] for i in range(x_test_for_subject.shape[1])],
                                             [y_test_v_for_subject, y_test_a_for_subject]))

                model.load_weights(model_checkpoint_path_SD)

                scores_for_subject = model.evaluate(
                    [x_test_for_subject[:, i] for i in range(x_test_for_subject.shape[1])],
                    [y_test_v_for_subject, y_test_a_for_subject], verbose=verbose)

                arousal_scores_subject_dependent_per_fold.append(scores_for_subject[-2:])
                print('After fine-tuning on AROUSAL',
                      list(zip(metrics, [round(score, 6) for score in scores_for_subject])))

            valence_scores_subject_dependent_list.append(valence_scores_subject_dependent_per_fold)
            arousal_scores_subject_dependent_list.append(arousal_scores_subject_dependent_per_fold)
            multi_scores_subject_dependent_list.append(multi_scores_subject_dependent_per_fold)

            scores_subject_dependent_list_before.append(scores_subject_dependent_per_fold_before)

            with open(os.path.join(metrics_dir, f'{model_name}_scores_SD_before.pkl'), 'wb') as fl:
                pickle.dump(scores_subject_dependent_list_before, fl)

            with open(os.path.join(metrics_dir, f'{model_name}_valence_scores_SD.pkl'), 'wb') as fl:
                pickle.dump(valence_scores_subject_dependent_list, fl)

            with open(os.path.join(metrics_dir, f'{model_name}_arousal_scores_SD.pkl'), 'wb') as fl:
                pickle.dump(arousal_scores_subject_dependent_list, fl)

            with open(os.path.join(metrics_dir, f'{model_name}_multi_scores_SD.pkl'), 'wb') as fl:
                pickle.dump(multi_scores_subject_dependent_list, fl)

        with open(os.path.join(metrics_dir, f'{model_name}_scores_SI.pkl'), 'wb') as fl:
            pickle.dump(scores_subject_independent_list, fl)
