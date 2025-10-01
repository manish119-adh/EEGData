from collections import namedtuple
import random
from EEGModels import EEGNet
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
from eeg_data import EEGData
from importlib import import_module
import glob
import os
import tensorflow as tf
import pandas as pd
# import intel_extension_for_tensorflow as itex

import eeg_data

performance = []

def run_eegnet(X, y, classes, file_name="usenix_inexp", kernelLen=64, dropout=0.5, F1=8, D=2, F2=16, norm_rate=0.25, train_test_val_split = (0.3, 0.1) ):
    """
    X = X, y= label,
    file_name = File name to store 
    kernelLen = length of temporal kernel in first later in EEgNet
    dropout: Rate of dropout layers def 0.5
    F1 = Number of Convolution channel kernels in first CNN layer in EEGNet
    D = Number of spatial kernels in first block of EEGNet
    F2 = Number of pointwise convolution kernels in second block of EEgNet
    train_test_split:
       If it is a tuple of  floats there must be (test_ratio, validation_ratio) with test_ratio+validation_ratio < 1
       It is split randomly with the ratio into train test and valiodation set
    Otherwise, 
       It is split as 0=train, 1 = test, 2=validation
    Otherwise if it is list of integers, it must have have [0,1,2] which separates each into train, test and validation sets
    """
    (n, c, t) = X.shape
    task_y = y
    if type(train_test_val_split) is tuple:
        # tuple of floats (test ratio, validation ratio)
        (test_rat, val_rat) = train_test_split
        assert test_rat + val_rat < 1
        X_train, X_test, y_train, y_test = train_test_split(X, task_y, test_size=test_rat)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=val_rat/(1 - test_rat))
    elif type(train_test_val_split) is list or type(train_test_val_split) is np.ndarray:
        train_test_val_split = np.array(train_test_val_split)
        X_train = X[train_test_val_split == 0]
        y_train = task_y[train_test_val_split == 0]
        X_test = X[train_test_val_split == 1]
        y_test = task_y[train_test_val_split == 1]
        X_valid = X[train_test_val_split == 2]
        y_valid = task_y[train_test_val_split == 2]
    else:
        print(type(train_test_val_split))
        raise Exception("Invalid train test split")
      
    # validation date
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

    eegnet = EEGNet(classes, Chans=c, Samples=t,  kernLength=kernelLen, dropoutRate= dropout, F1=F1, D=D, F2=F2, norm_rate=norm_rate)
    eegnet.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy',
    # tf.keras.metrics.Precision(name='macro_precision'),
    #     tf.keras.metrics.Recall(name='macro_recall'),
    #     tf.keras.metrics.F1Score(name='macro_f1_score')
    ])
    eegnet.load_weights(model_file)
    y_predicted = eegnet.predict(X_test)
    y_predicted = np.argmax(y_predicted, axis=1)
    acc = accuracy_score(y_test, y_predicted)
    macro_precision = precision_score(y_test, y_predicted, average="macro")
    macro_recall = recall_score(y_test, y_predicted, average="macro")
    macro_f1 = f1_score(y_test, y_predicted, average="macro")
    # loss, acc, macro_precision, macro_recall, macro_f1 = eegnet.evaluate(X_test, y_test, verbose=0)
    print(f'Accuracy of the best model in test data: {acc:.4f}')
    return acc, macro_precision, macro_recall, macro_f1


def run_task_identification(X, y, file_name="usenix_inexp", kernelLen=64, dropout=0.5, F1=8, D=2, F2=16, norm_rate=0.25, per_user=False, parameters = {}, train_test_val_split=(0.3, 0.1)):
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
    print(f"Tuning for parameters in {file_name}: parameters")
    

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


    acc, macro_precision, macro_recall, macro_f1 = run_eegnet(X, task_y, len(all_tasks), file_name=f"{file_name}_task_crossuser", kernelLen=kernelLen, dropout=dropout, F1=F1, D=D, F2=F2, norm_rate=norm_rate, train_test_val_split=test_train_val_split)
    add_result("All", acc, macro_precision, macro_recall, macro_f1)



    # individual task classification for each user separately EO EC excluded
    if not per_user:
        return results
    for u in all_subjects:
        retainidx = ([i for i in range(len(X)) if y[i][0] == u ])
        X_ret = X[retainidx]
        y_ret = task_y[retainidx]
        if type(train_test_val_split) is list or type(train_test_val_split) is np.ndarray:
            test_val_split_ret = np.array(train_test_val_split)[retainidx]
        else:
            test_val_split_ret = train_test_val_split
        print(f"Task identification for User {u}")
        acc, macro_precision, macro_recall, macro_f1 = run_eegnet(X_ret, y_ret, len(all_tasks), file_name = f"{file_name}_task_u{u}", kernelLen=kernelLen, dropout=dropout, F1=F1, D=D, F2=F2, norm_rate=norm_rate, train_test_val_split=test_val_split_ret)
        add_result(f"{u}", acc, macro_precision, macro_recall, macro_f1)
    return results




if __name__ == "__main__":
    print(tf.config.list_physical_devices('GPU'))
    kernel_lens = [16,32, 48,64, 80, 96, 112,128]
    dropout_rates = [0.1, 0.2, 0.25, 0.4, 0.5]
    F1s = [4, 8, 16, 32]
    Ds=[2, 4, 6]
    norm_rates = [0.0, 0.2, 0.25, 0.5, 0.75, 1]
    
    strides = [0.2, 0.4, 0.5, 0.8]
    overlaps = [0.3, 0.4, 0.5, 1.0]
    lows = [-1.0, -0.8, -0.5, -0.4, -0.3, 0, 0.2]
    highs = [0.5, 0.8, 1.0, 1.3, 1.5, 2.0]

    # results = []
    # edfdata = EEGData.load_from_EDF_dir("data/USENIX_Inexpensive")
    # event_list_g = import_module("usenix_inexp")
    # event_list_generator = event_list_g.event_list_generator
    # event_lists = []
    # for fil in edfdata._files:
    #     fil = f"{fil[:-4]}-events.txt"
    #     event_list = event_list_generator(fil)
    #     event_lists.append(event_list)
    
    # edfdata_original = edfdata.copy()
    # # select main 14 electrodes
    # edfdata.select_channels(["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"])
    # edfdata.fir_filter(low=1, high=50)

    def run_task_with_tuning(X, y, file_name="usenix_inexp", parameters = {}, completed_runs = set(), datatype=None, train_test_val_split=(0.3,0.1)):
        """
         datatype should not be None if completed_runs is to be used at all.
         It contains a set of parameters which have been completed so that there won't be
         a rerun 
        """
        results = []
        for kernelLen in kernel_lens:
                for dropoutrate in dropout_rates:
                    for F1 in F1s:
                        for D in Ds:
                            for norm_rate in norm_rates:
                                if datatype is not None:
                                    params = datatype(kernel_len=kernelLen, dropout=dropoutrate, F1=F1, D=D, F2=F1*D, norm_rate=norm_rate, **parameters)
                                    # do not run if they have already completed
                                    if params in completed_runs:
                                        continue
                                    
                                res_ = run_task_identification(X, y, file_name=file_name, kernelLen=kernelLen, dropout=dropoutrate, F1=F1, D=D, F2=F1*D, norm_rate=norm_rate, parameters=parameters, train_test_val_split=train_test_val_split)
                                df = pd.DataFrame(res_)
                                # write to file in append mode assuming the file is deleted at the start
                                if os.path.exists(f"{file_name}_results.csv"):
                                    df.to_csv(f"{file_name}_results.csv", mode="a", header=False)
                                else:
                                    df.to_csv(f"{file_name}_results.csv")
                                results.extend(res_)
                                if datatype is not None:
                                    completed_runs.add(params)
        return results



    # results = []
    # for hi in highs:
    #     for low in lows:
    #         X, y = edfdata.epoch_data(list(range(len(edfdata._data))), event_lists, low=low, hi=hi)
    #         retainidx = [i for i in range(len(y)) if y[i][1][:2] != "EO" and y[i][1][:2] != "EC"] # ignore EO and EC events
    #         X = X[retainidx]
    #         y = y[retainidx]
    #         res_ = run_task_with_tuning(X, y, file_name="usenix_inexp", parameters = {"hi":hi, "low":low})
    #         results.extend(res_)

    # df = pd.DataFrame(results)
    # df.to_csv("usenix_inexp_results.csv")



    

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
    test_train = [int(fil[4][:4] == "Test" ) for fil in file_split] # separete training and testing session
    # randomly make 40% of training into validation
    test_index = [i for i in range(len(test_train)) if test_train[i]]
    
    subjects = [int(fil[3][4:]) for fil in file_split]

    bshms_data = EEGData.load_from_CSV_files(files, frequency=128, channels=bshms_channels, sep=",")
    results = []

    BSHMSParams = namedtuple("BSHMSParams", ["stride", "overlap", "kernel_len", "dropout", "F1", "D", "F2", "norm_rate"])

    csv_file = "bshms_results.csv"
    completed_runs = set()
    if os.path.exists(csv_file):
        # get existing values so that we do not run them again
        data_f = pd.read_csv(csv_file)
        # data_f = data_f[["stride", "overlap", "kernel_len", "dropout", "F1", "D", "F2", "norm_rate"]]
        for tuple in (data_f.itertuples(index=False)):
            completed_runs.add(BSHMSParams(stride=tuple.stride, overlap=tuple.overlap, dropout=tuple.dropout, F1=tuple.F1, kernel_len=tuple.kernel_len, D=tuple.D, F2=tuple.F2, norm_rate=tuple.norm_rate))



    # We will defer validation split later

    

    for stride in strides:
        for overlap in overlaps:
            X, y, test_train_val_split = bshms_data.create_sliding_task(tasks, subjects, test_train_val_split = test_train, stride=stride, window=stride+overlap) 
            # edfdata.epoch_data(list(range(len(edfdata._data))), event_lists, low=low, hi=hi
            test_index = np.arange(len(X))[test_train_val_split == 1]
            val_index = np.random.choice(test_index,size=round(0.4 * len(test_index)))
            for v in val_index:
                test_train_val_split[v] = 2
            res_ = run_task_with_tuning(X, y, file_name="bshms", parameters = {"stride":stride, "overlap":overlap}, completed_runs=completed_runs, train_test_val_split=test_train_val_split, datatype=BSHMSParams)
            results.extend(res_)

    













    



    





