from __future__ import print_function
import numpy as np

from keras.models import Sequential, Model
from keras.layers import LSTM, BatchNormalization, Dropout
from keras.utils import np_utils
import keras
import pandas as pd
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from sklearn.metrics import roc_curve, auc
import time
from sklearn.model_selection import train_test_split
# from keras.utils.training_utils import multi_gpu_model

def readucr(filename):
    data = np.loadtxt(filename, delimiter = ',')
    Y = data[:,0]
    X = data[:,1:]
    return X, Y

def getData1(fname):
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

    idAOccur = np.where(rediv1_y_train == chosedLabel)
    delChosLabel_y_train = np.delete(rediv1_y_train, idAOccur)
    delChosLabel_x_train = np.delete(rediv1_x_train, idAOccur, axis=0)
    # idBOccur = np.where(y_test == chosedLabel)
    # thisTEST_labels = np.delete(y_test, idBOccur)
    # thisTEST = np.delete(x_test, idBOccur, axis=0)

    rediv2_x_train, x_val, rediv2_y_train, y_val = train_test_split(delChosLabel_x_train, delChosLabel_y_train,
                                                                    test_size=0.10, random_state=42)

    y_train = (rediv2_y_train - rediv2_y_train.min()) / (
            rediv2_y_train.max() - rediv2_y_train.min()) * (this_nb_classes - 1)
    y_val = (y_val - y_val.min()) / (
            y_val.max() - y_val.min()) * (this_nb_classes - 1)
    # y_test = (y_test - y_test.min()) / (
    #         y_test.max() - y_test.min()) * (this_nb_classes - 1)

    Y_train = np_utils.to_categorical(y_train, this_nb_classes)
    Y_val = np_utils.to_categorical(y_val, this_nb_classes)
    # Y_test = np_utils.to_categorical(y_test, this_nb_classes)

    # x_train_mean = thisTRAIN.mean()
    # x_train_std = thisTRAIN.std()
    # x_train = (thisTRAIN - x_train_mean) / (x_train_std)
    # x_test = (thisTEST - x_train_mean) / (x_train_std)

    # x_train_mean = rediv2_x_train.mean()
    # x_train_std = rediv2_x_train.std()
    # x_train = (rediv2_x_train - x_train_mean) / (x_train_std)
    # x_test_mean = x_test.mean()
    # x_test_std = x_test.std()
    # x_test = (x_test - x_test_mean) / (x_test_std)
    # x_val_mean = x_val.mean()
    # x_val_std = x_val.std()
    # x_val = (x_val - x_val_mean) / (x_val_std)

    x_train_mean = rediv2_x_train.mean()
    x_train_std = rediv2_x_train.std()
    x_train = (rediv2_x_train - x_train_mean) / (x_train_std)

    x_test = (x_test - x_train_mean) / (x_train_std)
    x_val = (x_val - x_train_mean) / (x_train_std)

    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))
    x_val = x_val.reshape(x_val.shape + (1,))
    return x_train,x_val,x_test,Y_train,Y_val,y_test,this_nb_classes,chosedLabel

def getData2(fname):
    x_train, y_train = readucr(direc + '/' + fname + '/' + fname + '_TRAIN')
    x_test, y_test = readucr(direc + '/' + fname + '/' + fname + '_TEST')
    y_train = y_train - 1
    y_test = y_test - 1
    nb_classes = len(np.unique(y_test))
    # batch_size = min(x_train.shape[0] / 10, 16)
    batch_size = 30
    chosedLabel = nb_classes - 1

    idAOccur = np.where(y_train == chosedLabel)
    thisTRAIN_labels = np.delete(y_train, idAOccur)
    thisTRAIN = np.delete(x_train, idAOccur, axis=0)
    idBOccur = np.where(y_test == chosedLabel)
    y_val = np.delete(y_test, idBOccur)
    x_val = np.delete(x_test, idBOccur, axis=0)
    this_nb_classes = nb_classes - 1

    thisTRAIN_labels = (thisTRAIN_labels - thisTRAIN_labels.min()) / (
            thisTRAIN_labels.max() - thisTRAIN_labels.min()) * (this_nb_classes - 1)
    y_val = (y_val - y_val.min()) / (
            y_val.max() - y_val.min()) * (this_nb_classes - 1)

    Y_train = np_utils.to_categorical(thisTRAIN_labels, this_nb_classes)
    Y_val = np_utils.to_categorical(y_val, this_nb_classes)

    x_train_mean = thisTRAIN.mean()
    x_train_std = thisTRAIN.std()
    x_train = (thisTRAIN - x_train_mean) / (x_train_std)
    x_val_mean = x_val.mean()
    x_val_std = x_val.std()
    # x_val = (x_val - x_val_mean) / (x_val_std)
    x_val = (x_val - x_train_mean) / (x_train_std)
    x_test_mean = x_test.mean()
    x_test_std = x_test.std()
    # x_test = (x_test - x_test_mean) / (x_test_std)
    x_test = (x_test - x_train_mean) / (x_train_std)

    x_train = x_train.reshape(x_train.shape + (1,))
    x_val = x_val.reshape(x_val.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))

    return x_train,x_val,x_test,Y_train,Y_val,y_test,this_nb_classes,chosedLabel


direc = './UCR_TS_Archive_2015'

methodName = 'LSTM-VE'

batch_size = 32
unitNum = 120
learnRate = 0.001
activation = 'softsign'   # softsign tanh
patience = 10

def trainModel():
    nb_epochs = 70
    all_result_file = 'all_results.txt'

    # flist = ['Two_Patterns',  'ElectricDevices', 'FaceAll', 'ECG5000','UWaveGestureLibraryAll']
    # flist = [ 'ElectricDevices','Two_Patterns','StarLightCurves','ECG5000']
    flist = ['ElectricDevices']
    # flist = ['ECG5000','Two_Patterns','StarLightCurves']
    for each in flist:
        fname = each
        print(fname)
        # origin_x_train, origin_y_train = readucr(direc + '/' + fname + '/' + fname + '_TRAIN')
        # origin_y_train = origin_y_train -1
        # origin_x_test, origin_y_test = readucr(direc + '/' + fname + '/' + fname + '_TEST')
        # origin_y_test = origin_y_test - 1
        #
        # nb_classes = len(np.unique(origin_y_test))
        # # batch_size = min(x_train.shape[0] / 10, 16)
        # chosedLabel = nb_classes - 1
        #
        # this_nb_classes = nb_classes - 1
        #
        # all_x = np.vstack((origin_x_train, origin_x_test))
        # # y = np.vstack((y_train, y_test))
        # all_y = np.r_[origin_y_train, origin_y_test]
        # rediv1_x_train, x_test, rediv1_y_train, y_test = train_test_split(all_x, all_y, test_size=0.10, random_state=42)
        #
        # idAOccur = np.where(rediv1_y_train == chosedLabel)
        # delChosLabel_y_train = np.delete(rediv1_y_train, idAOccur)
        # delChosLabel_x_train = np.delete(rediv1_x_train, idAOccur, axis=0)
        # # idBOccur = np.where(y_test == chosedLabel)
        # # thisTEST_labels = np.delete(y_test, idBOccur)
        # # thisTEST = np.delete(x_test, idBOccur, axis=0)
        #
        # rediv2_x_train,x_val,rediv2_y_train, y_val = train_test_split(delChosLabel_x_train, delChosLabel_y_train, test_size=0.10, random_state=42)
        #
        # y_train = (rediv2_y_train - rediv2_y_train.min()) / (
        #         rediv2_y_train.max() - rediv2_y_train.min()) * (this_nb_classes - 1)
        # y_val = (y_val - y_val.min()) / (
        #         y_val.max() - y_val.min()) * (this_nb_classes - 1)
        # # y_test = (y_test - y_test.min()) / (
        # #         y_test.max() - y_test.min()) * (this_nb_classes - 1)
        #
        # Y_train = np_utils.to_categorical(y_train, this_nb_classes)
        # Y_val = np_utils.to_categorical(y_val, this_nb_classes)
        # # Y_test = np_utils.to_categorical(y_test, this_nb_classes)
        #
        # # x_train_mean = thisTRAIN.mean()
        # # x_train_std = thisTRAIN.std()
        # # x_train = (thisTRAIN - x_train_mean) / (x_train_std)
        # # x_test = (thisTEST - x_train_mean) / (x_train_std)
        #
        # # x_train_mean = rediv2_x_train.mean()
        # # x_train_std = rediv2_x_train.std()
        # # x_train = (rediv2_x_train - x_train_mean) / (x_train_std)
        # # x_test_mean = x_test.mean()
        # # x_test_std = x_test.std()
        # # x_test = (x_test - x_test_mean) / (x_test_std)
        # # x_val_mean = x_val.mean()
        # # x_val_std = x_val.std()
        # # x_val = (x_val - x_val_mean) / (x_val_std)
        #
        # x_train_mean = rediv2_x_train.mean()
        # x_train_std = rediv2_x_train.std()
        # x_train = (rediv2_x_train - x_train_mean) / (x_train_std)
        #
        # x_test = (x_test - x_train_mean) / (x_train_std)
        # x_val = (x_val - x_train_mean) / (x_train_std)
        #
        # x_train = x_train.reshape(x_train.shape + (1,))
        # x_test = x_test.reshape(x_test.shape + (1,))
        # x_val = x_val.reshape(x_val.shape + (1,))

        x_train, x_val, x_test, Y_train, Y_val, y_test, this_nb_classes,chosedLabel = getData1(fname)

        print(x_train.shape)
        model = Sequential()
        # ----------------1---------------------------
        # model.add(LSTM(48, input_shape=x_train.shape[1:], return_sequences=True,name="lstm_1"))
        # model.add(BatchNormalization())
        # model.add(LSTM(48,name="lstm_2"))
        # model.add(keras.layers.Dense(this_nb_classes, activation='softmax',name="last"))

        # ----------------2---------------------------
        # model.add(LSTM(128, input_shape=x_train.shape[1:], return_sequences=True,name="lstm_1"))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.2))
        # model.add(LSTM(128, return_sequences=False,name="lstm_2"))
        # model.add(Dropout(0.2))
        # model.add(keras.layers.Dense(this_nb_classes, activation='softmax', name="last"))

        # ----------------3---------------------------
        # model.add(LSTM(120, input_shape=x_train.shape[1:], return_sequences=True, name="lstm_1"))
        # # model.add(BatchNormalization())
        # model.add(Dropout(0.8))
        # model.add(LSTM(120, return_sequences=True,name="lstm_2"))
        # model.add(Dropout(0.8))
        # model.add(LSTM(120, return_sequences=False, name="lstm_3"))
        # model.add(Dropout(0.8))
        # model.add(keras.layers.Dense(this_nb_classes, activation='softmax', name="last"))

        # ----------------4---------------------------
        model.add(LSTM(unitNum, input_shape=x_train.shape[1:], return_sequences=True,name="lstm_1",activation=activation))
        # model.add(BatchNormalization())
        model.add(Dropout(0.8))
        model.add(LSTM(unitNum,return_sequences=True,name="lstm_2",activation=activation))
        # model.add(BatchNormalization())
        model.add(Dropout(0.8))
        model.add(LSTM(unitNum,return_sequences=False,name="lstm_3",activation=activation))
        model.add(Dropout(0.8))
        model.add(keras.layers.Dense(this_nb_classes, activation='softmax',name="last"))


        model_checkpoint = ModelCheckpoint("./weights/" + methodName + "_%s_weights.h5" % fname, verbose=1,
                                           monitor='loss', save_best_only=True, save_weights_only=True)

        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                                      patience=10, min_lr=0.0001)

        early_stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=patience, mode='max')
        callback_list = [model_checkpoint, reduce_lr, early_stop]
        # optimizer = keras.optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999,
        #                       epsilon=1e-08, decay=0.0)
        optimizer = keras.optimizers.Adam(lr=learnRate)

        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy', keras.metrics.categorical_accuracy])
        print(model.summary())

        # hist = model.fit(x_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs,
        #                  verbose=1, validation_data=(x_test, Y_test), callbacks=[reduce_lr, early_stop])
        start = time.time()
        hist = model.fit(x_train, Y_train, batch_size=batch_size, epochs=nb_epochs,
                         verbose=2, validation_data=(x_val, Y_val), callbacks=callback_list)
        print("Training Time : ", time.time() - start)

        with open("./trainTime.txt", "a") as f:
            f.write(fname + ", " + methodName + ", " + str(time.time() - start) + "\n")

        # Print the testing results which has the lowest training loss.
        log = pd.DataFrame(hist.history)
        log.to_csv('./history/' + fname + '_LSTM_all_history.csv')

        with open(all_result_file, "a") as f:
            f.write(fname + ", LSTM" + ", " + str(log.loc[log['loss'].idxmin]['loss']) + ", "
                    + str(log.loc[log['loss'].idxmin]['val_acc']) + "\n")

        # summarize history for accuracy
        plt.plot(log['acc'])
        plt.plot(log['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('./history/' + fname + '_LSTM_model_accuracy.png')
        plt.close()
        # summarize history for loss
        plt.plot(log['loss'])
        plt.plot(log['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('./history/' + fname + '_LSTM_model_loss.png')

#         -----------------------------------val--------------------------------------
#         # 取某一层的输出为输出新建为model，采用函数模型
#         dense1_layer_model = Model(inputs=model.input,
#                                    outputs=model.get_layer('last').output)
#         # 以这个model的预测值作为输出
#         last_output = dense1_layer_model.predict(x_test)
#
#         outLierF = np.var(last_output, axis=1)
#         thisTEST_labels = [0 if x == chosedLabel else 1 for x in y_test]
#         # print(np.min(thisTEST_labels))
#         fpr, tpr, threshold = roc_curve(thisTEST_labels, outLierF)
#         roc_auc = auc(fpr, tpr)  ###计算auc的值
#         print("AUC", roc_auc)
#         with open("./aucs.txt", "a") as f:
#             f.write(
#                 fname + ',' + 'batch_size:' + str(batch_size) + ',unitNum:' + str(unitNum) + ',learnRate:' + str(
#                     learnRate) + ',activation:' + activation + ',patience:' + str(patience) + "\n")
#             f.write(fname + ", " + methodName + ", " + str(roc_auc) + "\n")


def evalModal():
    # flist = ['ElectricDevices', 'Two_Patterns', 'StarLightCurves', 'ECG5000']
    flist = ['Two_Patterns']
    # flist = ['ECG5000', 'Two_Patterns', 'StarLightCurves']
    for each in flist:
        fname = each
        print(fname)
        # origin_x_train, origin_y_train = readucr(direc + '/' + fname + '/' + fname + '_TRAIN')
        # origin_y_train = origin_y_train - 1
        # origin_x_test, origin_y_test = readucr(direc + '/' + fname + '/' + fname + '_TEST')
        # origin_y_test = origin_y_test - 1
        #
        # nb_classes = len(np.unique(origin_y_test))
        # chosedLabel = nb_classes - 1
        #
        # this_nb_classes = nb_classes - 1
        #
        # all_x = np.vstack((origin_x_train, origin_x_test))
        # # y = np.vstack((y_train, y_test))
        # all_y = np.r_[origin_y_train, origin_y_test]
        # rediv1_x_train, x_test, rediv1_y_train, y_test = train_test_split(all_x, all_y, test_size=0.20, random_state=42)
        #
        # # y_test = (y_test - y_test.min()) / (
        # #         y_test.max() - y_test.min()) * (this_nb_classes - 1)
        #
        # # Y_test = np_utils.to_categorical(y_test, this_nb_classes)
        #
        # x_test_mean = x_test.mean()
        # x_test_std = x_test.std()
        # x_test = (x_test - x_test_mean) / (x_test_std)

        x_train, x_val, x_test, Y_train, Y_val, y_test, this_nb_classes,chosedLabel = getData2(fname)

        model = Sequential()

        # ----------------1---------------------------

        # model.add(LSTM(48, input_shape=x_test.shape[1:], return_sequences=True,name="lstm_1"))
        # model.add(BatchNormalization())
        # model.add(LSTM(48,name="lstm_2"))
        # model.add(keras.layers.Dense(this_nb_classes, activation='softmax',name="last"))

        # ----------------2---------------------------
        # model.add(LSTM(128, input_shape=x_test.shape[1:], return_sequences=True, name="lstm_1"))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.2))
        # model.add(LSTM(128, return_sequences=False, name="lstm_2"))
        # model.add(Dropout(0.2))
        # model.add(keras.layers.Dense(this_nb_classes, activation='softmax', name="last"))

        # ----------------4---------------------------
        model.add(LSTM(unitNum, input_shape=x_test.shape[1:], return_sequences=True, name="lstm_1",activation='tanh'))
        model.add(BatchNormalization())
        # model.add(Dropout(0.8))
        model.add(LSTM(unitNum, return_sequences=True, name="lstm_2",activation='tanh'))
        model.add(BatchNormalization())
        # model.add(Dropout(0.8))
        model.add(LSTM(unitNum, return_sequences=False, name="lstm_3",activation='tanh'))
        # model.add(Dropout(0.8))
        model.add(keras.layers.Dense(this_nb_classes, activation='softmax', name="last"))

        optm = Adam(lr=1e-3)
        model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])

        model.load_weights("./weights/" + methodName + "_%s_weights.h5" % fname,)

        # 取某一层的输出为输出新建为model，采用函数模型
        dense1_layer_model = Model(inputs=model.input,
                                   outputs=model.get_layer('last').output)
        # 以这个model的预测值作为输出
        last_output = dense1_layer_model.predict(x_test)

        outLierF  = np.var(last_output,axis=1)


        thisTEST_labels = [0 if x == chosedLabel else 1 for x in y_test]
        with open("./outlierF/%s_%s.csv" %(fname,methodName),"a") as f:
            for of in outLierF:
                f.write(str(of) + '\n')
        with open("./outlierF/%s_%s_labels.csv" %(fname,methodName),"a") as f:
            for of in thisTEST_labels:
                f.write(str(of) + '\n')
        # print(np.min(thisTEST_labels))
        fpr, tpr, threshold = roc_curve(thisTEST_labels, outLierF)
        roc_auc = auc(fpr, tpr)  ###计算auc的值
        print("AUC", roc_auc)
        # plt.figure(figsize=(10, 10))
        # lw = 2
        # plt.plot(fpr, tpr, color='darkorange',
        #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
        # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver operating characteristic example')
        # plt.legend(loc="lower right")
        # plt.tight_layout()
        # plt.savefig("./aucs/"+methodName+"_%s_aucs.png" % fname, dpi=100)
        with open("./aucs.txt", "a") as f:
            f.write(
                fname + ',' + 'batch_size:' + str(batch_size) + ',unitNum:' + str(unitNum) + ',learnRate:' + str(learnRate) + ',activation:' + activation + ',patience:' + str(patience) + "\n")
            f.write(fname + ", " + methodName + ", " + str(roc_auc) + "\n")



if __name__ == '__main__':
    # trainModel()
    evalModal()