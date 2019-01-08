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

direc = './UCR_TS_Archive_2015'
methodName = 'RNN-RS'

batch_size = 128
learnRate = 0.005
activation = 'relu'  # relu  tanh
patience = 50


def trainModel():
    nb_epochs = 1000
    all_result_file = 'all_results.txt'

    # flist = ['Two_Patterns',  'ElectricDevices', 'FaceAll', 'ECG5000','UWaveGestureLibraryAll']
    # flist = [ 'ElectricDevices','Two_Patterns','StarLightCurves','ECG5000']
    flist = ['StarLightCurves']
    # flist = ['StarLightCurves', 'ECG5000']
    # flist = [ 'Two_Patterns', 'StarLightCurves', 'ECG5000']
    for each in flist:
        # fname = each
        # x_train, y_train = readucr(fname+'/'+fname+'_TRAIN')
        # x_test, y_test = readucr(fname+'/'+fname+'_TEST')
        # nb_classes =len(np.unique(y_test))
        # y_train = (y_train - y_train.min())/(y_train.max()-y_train.min())*(nb_classes-1)
        # y_test = (y_test - y_test.min())/(y_test.max()-y_test.min())*(nb_classes-1)
        # batch_size = min(x_train.shape[0]/10, 16)
        #
        # Y_train = np_utils.to_categorical(y_train, nb_classes)
        # Y_test = np_utils.to_categorical(y_test, nb_classes)
        #
        # x_train_mean = x_train.mean()
        # x_train_std = x_train.std()
        # x_train = (x_train - x_train_mean)/(x_train_std)
        #
        # x_test = (x_test - x_train_mean)/(x_train_std)

        fname = each
        print(fname)
        # x_train, y_train = readucr(direc + '/' + fname + '/' + fname + '_TRAIN')
        # y_train = y_train - 1
        # x_test, y_test = readucr(direc + '/' + fname + '/' + fname + '_TEST')
        # y_test = y_test - 1
        # nb_classes = len(np.unique(y_test))
        # # batch_size = min(x_train.shape[0] / 10, 16)
        #
        # chosedLabel = nb_classes - 1
        #
        # idAOccur = np.where(y_train == chosedLabel)
        # thisTRAIN = np.delete(x_train, idAOccur, axis=0)
        #
        #
        # x_train_mean = thisTRAIN.mean()
        # x_train_std = thisTRAIN.std()
        # x_train = (thisTRAIN - x_train_mean) / (x_train_std)
        # x_test = (x_test - x_train_mean) / (x_train_std)

        origin_x_train, origin_y_train = readucr(direc + '/' + fname + '/' + fname + '_TRAIN')
        origin_y_train = origin_y_train - 1
        origin_x_test, origin_y_test = readucr(direc + '/' + fname + '/' + fname + '_TEST')
        origin_y_test = origin_y_test - 1

        nb_classes = len(np.unique(origin_y_test))
        # batch_size = min(x_train.shape[0] / 10, 16)
        chosedLabel = nb_classes - 1


        all_x = np.vstack((origin_x_train, origin_x_test))
        # y = np.vstack((y_train, y_test))
        all_y = np.r_[origin_y_train, origin_y_test]
        rediv1_x_train, x_test, rediv1_y_train, y_test = train_test_split(all_x, all_y, test_size=0.10, random_state=42)

        idAOccur = np.where(rediv1_y_train == chosedLabel)
        delChosLabel_y_train = np.delete(rediv1_y_train, idAOccur)
        delChosLabel_x_train = np.delete(rediv1_x_train, idAOccur, axis=0)

        rediv2_x_train, x_val, rediv2_y_train, y_val = train_test_split(delChosLabel_x_train, delChosLabel_y_train,
                                                                        test_size=0.10, random_state=42)
        # x_train_mean = thisTRAIN.mean()
        # x_train_std = thisTRAIN.std()
        # x_train = (thisTRAIN - x_train_mean) / (x_train_std)
        # x_test = (thisTEST - x_train_mean) / (x_train_std)

        x_train_mean = rediv2_x_train.mean()
        x_train_std = rediv2_x_train.std()
        x_train = (rediv2_x_train - x_train_mean) / (x_train_std)
        x_test_mean = x_test.mean()
        x_test_std = x_test.std()
        x_test = (x_test - x_test_mean) / (x_test_std)
        x_val_mean = x_val.mean()
        x_val_std = x_val.std()
        x_val = (x_val - x_val_mean) / (x_val_std)

        print(x_train.shape)
        # input_dim = x_train.shape[1]
        # inp = Input(shape=(input_dim,))
        # x = Dense(input_dim // 2, activation=activation)(inp)
        # x = Dense(input_dim // 4, activation=inner_tanh)(x)
        # x = Dense(input_dim // 2, activation=activation)(x)
        # outp = Dense(input_dim, activation="sigmoid")(x)

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
        model_checkpoint = ModelCheckpoint("./weights/" + methodName + "_%s_weights.h5" % fname, verbose=1,
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
        log.to_csv('./history/' + fname + '_' +methodName+'_all_history.csv')

        with open(all_result_file, "a") as f:
            f.write(fname + ", "+ methodName + ", " + str(log.loc[log['loss'].idxmin]['loss']) + "\n")

        with open("./trainTime.txt", "a") as f:
            f.write(fname + ", " + methodName + ", " + str(time.time() - start) + "\n")
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
        plt.savefig('./history/' + fname + '_'+methodName+'_model_loss.png')


def evalModal():
    # flist = ['ElectricDevices', 'Two_Patterns', 'StarLightCurves', 'ECG5000']
    # flist = ['Two_Patterns', 'ECG5000']
    # flist = ['ECG5000']
    # flist = ['StarLightCurves', 'ECG5000']
    flist = ['StarLightCurves']
    for each in flist:
        # fname = each
        # x_train, y_train = readucr(fname+'/'+fname+'_TRAIN')
        # x_test, y_test = readucr(fname+'/'+fname+'_TEST')
        # nb_classes =len(np.unique(y_test))
        # y_train = (y_train - y_train.min())/(y_train.max()-y_train.min())*(nb_classes-1)
        # y_test = (y_test - y_test.min())/(y_test.max()-y_test.min())*(nb_classes-1)
        # batch_size = min(x_train.shape[0]/10, 16)
        #
        # Y_train = np_utils.to_categorical(y_train, nb_classes)
        # Y_test = np_utils.to_categorical(y_test, nb_classes)
        #
        # x_train_mean = x_train.mean()
        # x_train_std = x_train.std()
        # x_train = (x_train - x_train_mean)/(x_train_std)
        #
        # x_test = (x_test - x_train_mean)/(x_train_std)

        fname = each
        print(fname)
        origin_x_train, origin_y_train = readucr(direc + '/' + fname + '/' + fname + '_TRAIN')
        origin_y_train = origin_y_train - 1
        origin_x_test, origin_y_test = readucr(direc + '/' + fname + '/' + fname + '_TEST')
        origin_y_test = origin_y_test - 1

        nb_classes = len(np.unique(origin_y_test))
        # batch_size = min(x_train.shape[0] / 10, 16)
        chosedLabel = nb_classes - 1

        this_nb_classes = nb_classes - 1

        all_x = np.vstack((origin_x_train, origin_x_test))
        # y = np.vstack((y_train, y_test))
        all_y = np.r_[origin_y_train, origin_y_test]
        rediv1_x_train, x_test, rediv1_y_train, y_test = train_test_split(all_x, all_y, test_size=0.10, random_state=42)

        # x_train_mean = thisTRAIN.mean()
        # x_train_std = thisTRAIN.std()
        # x_train = (thisTRAIN - x_train_mean) / (x_train_std)
        # x_test = (thisTEST - x_train_mean) / (x_train_std)

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

        model.load_weights("./weights/" + methodName + "_%s_weights.h5" % fname)

        pred = model.predict(x_test)
        thisTEST_labels = [0 if x == chosedLabel else 1 for x in y_test]
        # outLierF = mean_squared_error(X[i], pred[i])
        outLierF = []
        for i in range(x_test.shape[0]):
            outLierF.append(mean_squared_error(x_test[i], pred[i]))

        # OF_max = max(outLierF)
        # OF_min = min(outLierF)
        # outLierF = (outLierF - OF_min) / OF_max
        #
        # idAOccur = np.where(thisTEST_labels == 1)
        # for index in idAOccur:
        #     outLierF[index] = 1 - outLierF[index]

        fpr, tpr, threshold = roc_curve(thisTEST_labels, outLierF)
        roc_auc = auc(fpr, tpr)  ###计算auc的值
        print("AUC", roc_auc)
        with open("./outlierF/%s_%s.csv" %(fname,methodName),"a") as f:
            for of in outLierF:
                f.write(str(of) + '\n')
        with open("./outlierF/%s_%s_labels.csv" %(fname,methodName),"a") as f:
            for of in thisTEST_labels:
                f.write(str(of) + '\n')
        with open("./aucs.txt", "a") as f:
            f.write(
                fname + ',' + 'batch_size:' + str(batch_size) + ',unitNum:' + str(u1) + "," + str(
                    u2) + "," + str(u3) + ',learnRate:' + str(
                    learnRate) + ',activation:' + activation + ',patience:' + str(patience) + "\n")
            f.write(fname + ", " + methodName + ", " + str(roc_auc) + "\n")

        # outlierThreshold = np.max(outLierF) * 0.45
        # maxoutlierThreshold = np.max(outLierF)
        # outlierThresholdList = []
        # aucs = []
        #
        # while (outlierThreshold <= maxoutlierThreshold):
        #     outPrediction = [1 if x > outlierThreshold else 0 for x in outLierF]
        #     # Compute ROC curve and ROC area for each class
        #     fpr, tpr, threshold = roc_curve(thisTEST_labels, outPrediction)  ###计算真正率和假正率
        #     roc_auc = auc(fpr, tpr)  ###计算auc的值
        #     # auc_score = roc_auc_score(thisTEST_labels, outPrediction)
        #     outlierThresholdList.append(outlierThreshold);
        #     aucs.append(roc_auc)
        #     outlierThreshold += 0.001
        #
        # # plt.plot(fpr, tpr, color='darkorange',
        # #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
        # plt.figure(figsize=(10, 10))
        # lw = 2
        # plt.plot(outlierThresholdList, aucs, color='darkorange', lw=lw, linestyle='--')
        #
        # f = open("./aucs/" + methodName + "_%s_aucs.csv" % fname, "w")  # 打开文件以便写入
        # for index in range(len(outlierThresholdList)):
        #     f.write(str(outlierThresholdList[index]) + ',' + str(aucs[index]) + '\n')
        # f.close()

        # # plt.xlim([np.max(outLierF) * 0.40, maxoutlierThreshold * 1.1])
        # # plt.ylim([0.2, 1.05])
        # plt.xlabel('outlierThreshold')
        # plt.ylabel('AUC')
        # plt.title('AUC to outlierThreshold')
        # plt.tight_layout()
        # plt.savefig("./aucs/" + methodName + "_%s_aucs.png" % fname, dpi=100)
        # print("\nEvaluating : ")
        # loss, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)
        # print()
        # print("Final Accuracy : ", accuracy)

        # return accuracy



if __name__ == '__main__':
    trainModel()
    evalModal()

