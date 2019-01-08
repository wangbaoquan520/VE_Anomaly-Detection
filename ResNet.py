# -*- coding: utf-8 -*-


from __future__ import print_function
  
from keras.models import Model
from keras.layers import Input, Dense, merge, Activation
from keras.utils import np_utils
import numpy as np
import pandas as pd
import keras 
from keras.callbacks import ReduceLROnPlateau,EarlyStopping
from keras.callbacks import ModelCheckpoint
import time
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

np.random.seed(813306)
 
def build_resnet(input_shape, n_feature_maps, nb_classes):
    # print ('build conv_x')
    x = Input(shape=(input_shape))
    conv_x = keras.layers.normalization.BatchNormalization()(x)
    conv_x = keras.layers.Conv2D(n_feature_maps, (8, 1), padding='same')(conv_x)
    conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)
     
    # print ('build conv_y')
    conv_y = keras.layers.Conv2D(n_feature_maps, (5, 1), padding='same')(conv_x)
    conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)
     
    # print ('build conv_z')
    conv_z = keras.layers.Conv2D(n_feature_maps, (3, 1), padding='same')(conv_y)
    conv_z = keras.layers.normalization.BatchNormalization()(conv_z)
     
    is_expand_channels = not (input_shape[-1] == n_feature_maps)
    if is_expand_channels:
        shortcut_y = keras.layers.Conv2D(n_feature_maps, (1, 1),padding='same')(x)
        shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)
    else:
        shortcut_y = keras.layers.normalization.BatchNormalization()(x)
    # print ('Merging skip connection')
    y = merge([shortcut_y, conv_z], mode='sum')
    y = Activation('relu')(y)
     
    # print ('build conv_x')
    x1 = y
    conv_x = keras.layers.Conv2D(n_feature_maps*2, (8, 1), padding='same')(x1)
    conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)
     
    # print ('build conv_y')
    conv_y = keras.layers.Conv2D(n_feature_maps*2, (5, 1), padding='same')(conv_x)
    conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)
     
    # print ('build conv_z')
    conv_z = keras.layers.Conv2D(n_feature_maps*2, (3, 1), padding='same')(conv_y)
    conv_z = keras.layers.normalization.BatchNormalization()(conv_z)
     
    is_expand_channels = not (input_shape[-1] == n_feature_maps*2)
    if is_expand_channels:
        shortcut_y = keras.layers.Conv2D(n_feature_maps*2, (1, 1),padding='same')(x1)
        shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)
    else:
        shortcut_y = keras.layers.normalization.BatchNormalization()(x1)
    # print ('Merging skip connection')
    y = merge([shortcut_y, conv_z], mode='sum')
    y = Activation('relu')(y)
     
    # print ('build conv_x')
    x1 = y
    conv_x = keras.layers.Conv2D(n_feature_maps*2, (8, 1), padding='same')(x1)
    conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)
     
    # print ('build conv_y')
    conv_y = keras.layers.Conv2D(n_feature_maps*2, (5, 1), padding='same')(conv_x)
    conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)
     
    # print ('build conv_z')
    conv_z = keras.layers.Conv2D(n_feature_maps*2, (3, 1), padding='same')(conv_y)
    conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

    is_expand_channels = not (input_shape[-1] == n_feature_maps*2)
    if is_expand_channels:
        shortcut_y = keras.layers.Conv2D(n_feature_maps*2, (1, 1),padding='same')(x1)
        shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)
    else:
        shortcut_y = keras.layers.normalization.BatchNormalization()(x1)
    # print ('Merging skip connection')
    y = merge([shortcut_y, conv_z], mode='sum')
    y = Activation('relu')(y)
     
    full = keras.layers.pooling.GlobalAveragePooling2D()(y)   
    out = Dense(nb_classes, activation='softmax',name='last')(full)
    # print ('        -- model was built.')
    return x, out
 
       
def readucr(filename):
    data = np.loadtxt(filename, delimiter = ',')
    Y = data[:,0]
    X = data[:,1:]
    return X, Y

methodName = 'ResNet'
direc = './UCR_TS_Archive_2015'
def trainModel():
    nb_epochs = 200
    all_result_file = 'all_results.txt'
    # flist = ['Two_Patterns',  'ElectricDevices', 'FaceAll', 'ECG5000','UWaveGestureLibraryAll']
    flist = [ 'ElectricDevices','Two_Patterns','StarLightCurves','ECG5000']
    # flist = ['StarLightCurves', 'ECG5000']
    for each in flist:
        fname = each
        print(fname)
        x_train, y_train = readucr(direc + '/' + fname + '/' + fname + '_TRAIN')
        y_train = y_train - 1
        x_test, y_test = readucr(direc + '/' + fname + '/' + fname + '_TEST')
        y_test = y_test - 1
        nb_classes = len(np.unique(y_test))
        # batch_size = min(x_train.shape[0] / 10, 16)
        batch_size = 30
        chosedLabel = nb_classes - 1

        idAOccur = np.where(y_train == chosedLabel)
        thisTRAIN_labels = np.delete(y_train, idAOccur)
        thisTRAIN = np.delete(x_train, idAOccur, axis=0)
        idBOccur = np.where(y_test == chosedLabel)
        thisTEST_labels = np.delete(y_test, idBOccur)
        thisTEST = np.delete(x_test, idBOccur, axis=0)
        this_nb_classes = nb_classes - 1

        thisTRAIN_labels = (thisTRAIN_labels - thisTRAIN_labels.min()) / (
                thisTRAIN_labels.max() - thisTRAIN_labels.min()) * (this_nb_classes - 1)
        thisTEST_labels = (thisTEST_labels - thisTEST_labels.min()) / (
                thisTEST_labels.max() - thisTEST_labels.min()) * (this_nb_classes - 1)

        Y_train = np_utils.to_categorical(thisTRAIN_labels, this_nb_classes)
        Y_test = np_utils.to_categorical(thisTEST_labels, this_nb_classes)

        x_train_mean = thisTRAIN.mean()
        x_train_std = thisTRAIN.std()
        x_train = (thisTRAIN - x_train_mean) / (x_train_std)
        # x_test = (thisTEST - x_train_mean) / (x_train_std)

        x_test_mean = thisTEST.mean()
        x_test_std = thisTEST.std()
        x_test = (thisTEST - x_test_mean) / (x_test_std)

        x_train = x_train.reshape(x_train.shape + (1,1,))
        x_test = x_test.reshape(x_test.shape + (1,1,))
        print(x_train.shape)

        x, y = build_resnet(x_train.shape[1:], 64, this_nb_classes)

        model = Model(inputs=x, outputs=y)
        optimizer = keras.optimizers.Adam()
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        early_stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, mode='max')
        model_checkpoint = ModelCheckpoint("./weights/" + methodName + "_%s_weights.h5" % fname, verbose=1,
                                           monitor='loss', save_best_only=True, save_weights_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                                      patience=50, min_lr=0.0001)
        start = time.time()
        hist = model.fit(x_train, Y_train, batch_size=batch_size, epochs=nb_epochs,
                         verbose=2, validation_data=(x_test, Y_test), callbacks=[reduce_lr,model_checkpoint,early_stop])
        print("Training Time : ", time.time() - start)
        log = pd.DataFrame(hist.history)
        log.to_csv('./history/' + fname + '_' + methodName + '_all_history.csv')

        with open("./trainTime.txt", "a") as f:
            f.write(fname + ", " + methodName + ", " + str(time.time() - start) + "\n")

        with open(all_result_file, "a") as f:
            f.write(fname + ", " + methodName + ", " + str(log.loc[log['loss'].idxmin]['loss']) + ", "
                    + str(log.loc[log['loss'].idxmin]['val_acc']) + "\n")

        # summarize history for accuracy
        plt.plot(log['acc'])
        plt.plot(log['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('./history/' + fname + '_' + methodName + '_model_accuracy.png')
        plt.close()
        # summarize history for loss
        plt.plot(log['loss'])
        plt.plot(log['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('./history/' + fname + '_' + methodName + '_model_loss.png')


def evalModal():
    flist = ['ElectricDevices', 'Two_Patterns', 'StarLightCurves', 'ECG5000']
    for each in flist:
        fname = each
        print(fname)
        x_train, y_train = readucr(direc + '/' + fname + '/' + fname + '_TRAIN')
        y_train = y_train - 1
        x_test, y_test = readucr(direc + '/' + fname + '/' + fname + '_TEST')
        y_test = y_test - 1
        nb_classes = len(np.unique(y_test))
        chosedLabel = nb_classes - 1
        this_nb_classes = nb_classes - 1

        idAOccur = np.where(y_train == chosedLabel)
        thisTRAIN = np.delete(x_train, idAOccur, axis=0)
        # x_train_mean = thisTRAIN.mean()
        # x_train_std = thisTRAIN.std()
        #
        # x_test = (x_test - x_train_mean) / (x_train_std)

        x_test_mean = x_test.mean()
        x_test_std = x_test.std()

        x_test = (x_test - x_test_mean) / (x_test_std)

        x_test = x_test.reshape(x_test.shape + (1,1,))

        x, y = build_resnet(x_test.shape[1:], 64, this_nb_classes)

        model = Model(inputs=x, outputs=y)
        optimizer = keras.optimizers.Adam()
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        model.load_weights("./weights/" + methodName + "_%s_weights.h5" % fname)

        # 取某一层的输出为输出新建为model，采用函数模型
        dense1_layer_model = Model(inputs=model.input,
                                   outputs=model.get_layer('last').output)
        # 以这个model的预测值作为输出
        last_output = dense1_layer_model.predict(x_test)

        outLierF = np.var(last_output, axis=1)
        thisTEST_labels = [0 if x == chosedLabel else 1 for x in y_test]
        # print(np.min(thisTEST_labels))
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
            # f.write(
                # fname + ',' + 'batch_size:' + str(batch_size) + ',unitNum:' + str(input_dim // 2) + "," + str(
                #     input_dim // 4) + "," + str(input_dim // 2) + ',learnRate:' + str(
                #     learnRate) + ',activation:' + activation + ',patience:' + str(patience) + "\n")
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
        #
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
    # trainModel()
    evalModal()