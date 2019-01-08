# -*- coding: utf-8 -*-
 
from __future__ import print_function

from collections import defaultdict
from sklearn.metrics import mean_squared_error
from keras.models import Model
from keras.layers import Input, Dense, Dropout
import numpy as np
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.callbacks import ModelCheckpoint
import pandas as pd
import matplotlib.pyplot as plt
from keras import backend as K
from sklearn.metrics import roc_curve, auc
import keras
from sklearn.model_selection import train_test_split
import util
import csv

np.random.seed(813306)

import time

def readucr(filename):
    data = np.loadtxt(filename, delimiter = ',')
    Y = data[:,0]
    X = data[:,1:]
    return X, Y

def inner_tanh(x):
    k = 3
    N = 4
    return 1/2 + 1/(2*(k-1)) * sum(K.tanh(x- (j/N)) for j in range(1, N-1))

def calculate_outlier_factor(X, Y, pred):
    outlier_factors = defaultdict(dict)
    for i in range(X.shape[0]):
        outlier_factors[i]["OF"] = mean_squared_error(X[i], pred[i])
        outlier_factors[i]["Y"] = Y[i]
    return outlier_factors

def loadClusterLabels(filePath):
    birth_data = []
    with open(filePath) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        # birth_header = next(csv_reader)  # 读取第一行每一列的标题
        for row in csv_reader:  # 将csv 文件中的数据保存到birth_data中
            birth_data.append(row)
    birth_data = np.array(birth_data)
    return birth_data[:,2]


direc = './UCR_TS_Archive_2015'
methodName = 'RNN-RS'

batch_size = 128
learnRate = 0.005
activation = 'relu'  # relu  tanh
patience = 50


def trainModel():
    nb_epochs = 1000
    all_result_file = 'all_results.txt'
    stationIDs = ['12066','13060']
    fname = 'canopy+k-means'
    allfilePath = "./data/allStationsFromOracl.csv"
    testfilePath = "./data/allStations-all.csv"
    negetiveLabels = ['0','1','3']
    allTracks, allDateList = util.loadTrackData(allfilePath)  # 返回的为字典
    testTracks, testDateList = util.loadTrackData(testfilePath)
    # 训练模型
    for stationID in stationIDs:
        print("stationID:", stationID)
        clusterOutFilePath = './kmeansOut/%s_6.csv' % stationID
        tracks = allTracks[stationID]
        dateList = allDateList[stationID]
        labels = loadClusterLabels(clusterOutFilePath)

        origin_x_train = tracks
        origin_y_train = labels
        origin_x_test = testTracks[stationID]
        origin_y_test = testDateList[stationID]

        for chosedLabel in negetiveLabels:
            idAOccur = np.where(origin_y_train == chosedLabel)
            origin_y_train = np.delete(origin_y_train, idAOccur)
            origin_x_train = np.delete(origin_x_train, idAOccur, axis=0)

        x_train, x_val, y_train, y_val = train_test_split(origin_x_test, origin_y_test, test_size=0.1, random_state=42)

        x_train_mean = x_train.mean()
        x_train_std = x_train.std()
        x_train = (x_train - x_train_mean) / (x_train_std)
        x_val_mean = x_val.mean()
        x_val_std = x_val.std()
        x_val = (x_val - x_val_mean) / (x_val_std)

        print(x_train.shape)

        inp = Input(x_train.shape[1:])
        input_dim = x_train.shape[1]

        # u1 = input_dim // 2
        # u2 = input_dim // 4
        # u3 = input_dim // 2
        u1 = 25
        u2 = 10
        u3 = 25

        y = Dropout(0.8)(inp)
        y = Dense(u1, activation=activation)(y)
        y = Dropout(0.8)(y)
        y = Dense(u2, activation=inner_tanh)(y)
        y = Dropout(0.8)(y)
        y = Dense(u3, activation=activation)(y)
        y = Dropout(0.8)(y)
        outp = Dense(input_dim, activation='sigmoid')(y)

        model = Model(inputs=inp, outputs=outp)
        model_checkpoint = ModelCheckpoint("./weightsOil/" + methodName + "_%s_weights.h5" % stationID, verbose=1,
                                           monitor='loss', save_best_only=True, save_weights_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                                      patience=50, min_lr=0.0001)
        early_stop = EarlyStopping(monitor='loss', min_delta=0, patience=patience, mode='min')
        optimizer = keras.optimizers.Adam(lr=learnRate)
        model.compile(loss="mean_squared_error", optimizer=optimizer)
        print(model.summary())

        start = time.time()
        # hist = model.fit(thisTRAIN, thisTRAIN, epochs=2000, callbacks=[early_stop], verbose=0)
        hist = model.fit(x_train, x_train, batch_size=batch_size, epochs=nb_epochs,
                         verbose=2,validation_data=(x_val, x_val),
                         # callbacks = [TestCallback((x_train, Y_train)), reduce_lr, keras.callbacks.TensorBoard(log_dir='./log'+fname, histogram_freq=1)])
                         callbacks=[reduce_lr,model_checkpoint,early_stop])
        print("Training Time : ", time.time() - start)

        # Print the testing results which has the lowest training loss.
        # log = pd.DataFrame(hist.history)
        # print(log.loc[log['loss'].idxmin]['loss'], log.loc[log['loss'].idxmin]['val_acc'])

        # Print the testing results which has the lowest training loss.
        log = pd.DataFrame(hist.history)
        log.to_csv('./historyOil/' + stationID + '_' +methodName+'_all_history.csv')

        # with open(all_result_file, "a") as f:
        #     f.write(fname + ", "+ methodName + ", " + str(log.loc[log['loss'].idxmin]['loss']) + "\n")

        # summarize history for accuracy
        # plt.plot(log['acc'])
        # plt.plot(log['val_acc'])
        # plt.title('model accuracy')
        # plt.ylabel('accuracy')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        # plt.savefig('./history/' + fname + '_'+methodName +'_model_accuracy.png')
        # plt.close()
        # summarize history for loss
        plt.plot(log['loss'])
        plt.plot(log['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('./historyOil/' + stationID + '_'+methodName+'_model_loss.png')


def evalModal():
    stationIDs = ['12066','13060']
    fname = 'canopy+k-means'
    testfilePath = "./data/allStations-all.csv"
    negetiveLabels = ['0', '1', '3']
    testTracks, testDateList = util.loadTrackData(testfilePath)
    # 训练模型
    for stationID in stationIDs:
        print("stationID:", stationID)
        clusterOutFilePath = './kmeansOut/%s_6.csv' % stationID
        tracks = testTracks[stationID]
        dateList = testDateList[stationID]

        x_test = tracks

        x_test_mean = x_test.mean()
        x_test_std = x_test.std()
        x_test = (x_test - x_test_mean) / (x_test_std)

        print(x_test.shape)
        input_dim = x_test.shape[1]
        inp = Input(x_test.shape[1:])
        # u1 = input_dim // 2
        # u2 = input_dim // 4
        # u3 = input_dim // 2
        u1 = 25
        u2 = 10
        u3 = 25

        y = Dropout(0.8)(inp)
        y = Dense(u1, activation=activation)(y)
        y = Dropout(0.8)(y)
        y = Dense(u2, activation=inner_tanh)(y)
        y = Dropout(0.8)(y)
        y = Dense(u3, activation=activation)(y)
        y = Dropout(0.8)(y)
        outp = Dense(input_dim, activation='sigmoid')(y)

        model = Model(inputs=inp, outputs=outp)
        optimizer = keras.optimizers.Adam(lr=learnRate)
        model.compile(loss="mean_squared_error", optimizer=optimizer)

        model.load_weights("./weightsOil/" + methodName + "_%s_weights.h5" % stationID)

        pred = model.predict(x_test)
        # for chosedLabel in negetiveLabels:
        #     thisTEST_labels = [0 if x == chosedLabel else 1 for x in y_test]
        # outLierF = mean_squared_error(X[i], pred[i])
        outLierF = []
        for i in range(x_test.shape[0]):
            outLierF.append(mean_squared_error(x_test[i], pred[i]))

        OF_max = max(outLierF)
        OF_min = min(outLierF)
        outLierF = (outLierF - OF_min) / OF_max

        with open("./outlierF/%s_%s.csv" %(stationID,methodName),"a") as f:
            for of in outLierF:
                f.write(str(of) + '\n')
        # with open("./outlierF/%s_%s_labels.csv" %(stationID,methodName),"a") as f:
        #     for of in thisTEST_labels:
        #         f.write(str(of) + '\n')



if __name__ == '__main__':
    trainModel()
    evalModal()

