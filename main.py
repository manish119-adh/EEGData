from collections import namedtuple
from EEGModels import EEGNet
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import numpy as np
from eeg_data import EEGData
from importlib import import_module
import glob
import os
import keras
import pandas as pd

import eeg_data

performance = []

def run_eegnet(X, y, classes, file_name="usenix_inexp", kernelLen=64, dropout=0.5, F1=8, D=2, F2=16, norm_rate=0.25 ):
    (n, c, t) = X.shape
    task_y = y
    X_train, X_test, y_train, y_test = train_test_split(X, task_y, test_size=0.3)
    # validation data
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)
    eegnet = EEGNet(classes, Chans=c, Samples=t, kernLength=kernelLen, dropoutRate= dropout, F1=F1, D=D, F2=F2, norm_rate=norm_rate)
    eegnet.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True  # Important: This restores the model to the best weights found during training
    )

# The ModelCheckpoint callback saves the model only when a new best 'val_loss' is achieved.
# 'save_best_only=True' is the key parameter here.
    model_file = f"model/{file_name}.keras.weights.h5"
    model_checkpoint_callback = ModelCheckpoint(
        filepath=model_file,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )
    eegnet.fit(x =X_train, y= y_train, batch_size=10, epochs=100, validation_data=(X_valid, y_valid), callbacks=[early_stopping_callback, model_checkpoint_callback])

    eegnet = EEGNet(classes, Chans=c, Samples=t)
    eegnet.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy',
    keras.metrics.Precision(average='macro', name='macro_precision'),
        keras.metrics.Recall(average='macro', name='macro_recall'),
        keras.metrics.F1Score(average='macro', name='macro_f1_score')])
    eegnet.load_weights(model_file)
    loss, acc, macro_precision, macro_recall, macro_f1 = eegnet.evaluate(X_test, y_test, verbose=0)
    print(f'Accuracy of the best model in test data: {acc:.4f}')
    return loss, acc, macro_precision, macro_recall, macro_f1


def run_task_identification(X, y, file_name="usenix_inexp", kernelLen=64, dropout=0.5, F1=8, D=2, F2=16, norm_rate=0.25, per_user=False, parameters = {}):
    all_tasks = list(set([d[1] for d in y]))
    all_subjects = list(set(d[0] for d in y))
    all_subjects.sort()
    results = []
    parameters = dict(parameters)
    parameters["kernel_len"] = kernelLen
    parameters["dropout"] = dropout
    parameters["F1"] = F1
    parameters["D"] = D
    parameters["F2"] = F2
    parameters["norm_rate"] = norm_rate
    

    task_y = np.array([all_tasks.index(d[1]) for d in y])
    # person_y = [d[0] for d in y]
    # training data
    print("Task identification cross user")

    def add_result(user, acc, macro_precision, macro_recall, macro_f1):
        val_ = dict(parameters)
        val_["user"] = user
        val_["accuracy"] = acc
        val_["macro_precision"] = macro_precision
        val_["macro_recall"] = macro_recall
        val_["macro_f1"] = macro_f1
        results.append(val_)


    loss, acc, macro_precision, macro_recall, macro_f1 = run_eegnet(X, task_y, len(all_tasks), file_name=f"{file_name}_task_crossuser", kernelLen=kernelLen, dropout=dropout, F1=F1, D=D, F2=F2, norm_rate=norm_rate)
    add_result("All", acc, macro_precision, macro_recall, macro_f1)


    # individual task classification for each user separately EO EC excluded
    if not per_user:
        return
    for u in all_subjects:
        retainidx = ([i for i in range(len(X)) if y[i][0] == u ])
        X_ret = X[retainidx]
        y_ret = task_y[retainidx]

        print(f"Task identification for User {u}")
        loss, acc, macro_precision, macro_recall, macro_f1 = run_eegnet(X_ret, y_ret, len(all_tasks), file_name = f"{file_name}_task_u{u}", kernelLen=kernelLen, dropout=dropout, F1=F1, D=D, F2=F2, norm_rate=norm_rate)
        add_result(f"{u}", acc, macro_precision, macro_recall, macro_f1)
    return




if __name__ == "__main__":
    kernel_lens = [16,32, 48,64, 80, 96, 112,128]
    dropout_rates = [0.1, 0.2, 0.25, 0.4, 0.5]
    F1s = [4, 8, 16, 32]
    Ds=[2, 4, 6]
    norm_rates = [0.0, 0.2, 0.25, 0.5, 0.75, 1]
    
    strides = [0.2, 0.4, 0.5, 0.8]
    overlaps = [0.3, 0.4, 0.5, 1.0]
    lows = [-1.0, -0.8, -0.5, -0.4, -0.3, 0, 0.2]
    highs = [0.5, 0.8, 1.0, 1.3, 1.5, 2.0]

    results = []
    edfdata = EEGData.load_from_EDF_dir("data/USENIX_Inexpensive")
    event_list_g = import_module("usenix_inexp")
    event_list_generator = event_list_g.event_list_generator
    event_lists = []
    for fil in edfdata._files:
        fil = f"{fil[:-4]}-events.txt"
        event_list = event_list_generator(fil)
        event_lists.append(event_list)
    
    edfdata_original = edfdata.copy()
    # select main 14 electrodes
    edfdata.select_channels(["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"])
    edfdata.fir_filter(low=1, high=50)

    def run_task_with_tuning(X, y, file_name="usenix_inexp", parameters = {}):
        for kernelLen in kernel_lens:
                for dropoutrate in dropout_rates:
                    for F1 in F1s:
                        for D in Ds:
                            for norm_rate in norm_rates:

                                run_task_identification(X, y, file_name="usenix_inexp", kernelLen=kernelLen, dropout=dropoutrate, F1=F1, D=D, F2=F1*D, norm_rate=norm_rate, parameters=parameters)



    results = []
    for hi in highs:
        for low in lows:
            X, y = edfdata.epoch_data(list(range(len(edfdata._data))), event_lists, low=low, hi=hi)
            retainidx = [i for i in range(len(y)) if y[i][1][:2] != "EO" and y[i][1][:2] != "EC"] # ignore EO and EC events
            X = X[retainidx]
            y = y[retainidx]
            res_ = run_task_with_tuning(X, y, file_name="usenix_inexp", parameters = {"hi":hi, "low":low})
            results.extend(res_)

    df = pd.DataFrame(results)
    df.to_csv("usenix_inexp.csv")



    

    # print("Cross task user identification")

    # user_y = np.array([d[0] for d in y])
    # run_eegnet(X, user_y, len(edfdata._data), file_name=f"usenix_inexp_user_crosstask")

    # for task in all_tasks:
    #     if task[:2] == "EO" or task[:2] == "EC":
    #         continue
    #     retainidx = ([i for i in range(len(X)) if y[i][1] == task])
    #     X_ret = X[retainidx]
    #     y_ret = user_y[retainidx]
    #     print(f"Identifying user for task {task}")
    #     run_eegnet(X_ret, y_ret, len(edfdata._data), file_name=f"usenix_inexp_user_task_{task}")
    results = []
    channels = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
    bands = ["THETA", "ALPHA", "LOW_BETA", "HIGH_BETA", "GAMMA"]
    bshms_channels = [f"{channel}_{band}" for band in bands for channel in channels]
    bshms_freq = 128

    files = [fil for fil in glob.glob("data/bshms/Emotiv-based-Data/*/*/*/Emotiv/*.bp.csv")]
    file_split = [fil.split(os.sep) for fil in files]
    tasks = [fil[-3] for fil in file_split]
    subjects = [int(fil[3][4:]) for fil in file_split]

    bshms_data = EEGData.load_from_CSV_files(files, frequency=128, channels=bshms_channels, sep=",")
    results = []

    for stride in strides:
        for overlap in overlaps:
            X, y = bshms_data.create_sliding_task(tasks, subjects, stride=stride, window=stride+overlap) 
            #edfdata.epoch_data(list(range(len(edfdata._data))), event_lists, low=low, hi=hi
            res_ = run_task_with_tuning(X, y, file_name="bshms", parameters = {"stride":stride, "overlap":overlap})
            results.extend(res_)

    df = pd.DataFrame(results)
    df.to_csv("usenix_inexp.csv")













    



    





