import os
import sys

import keras.layers
import keras.losses as losses
import tensorflow as tf
from tensorflow import keras
from dataload import *
from matplotlib import pyplot as plt

savepath = './mymodel.ckpt'
bestpath='./bestmodel.ckpt'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
epochs = 20
batchsize = 100
dropoutrate = 0.3


class mymodel(keras.Model):
    def __init__(self):
        super(mymodel, self).__init__()
        self.mod = keras.models.Sequential([
            keras.layers.Conv2D(filters=64, kernel_size=3, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),  # 层1

            keras.layers.Conv2D(filters=64, kernel_size=3, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
            keras.layers.Dropout(dropoutrate),  # 层2

            keras.layers.Conv2D(filters=128, kernel_size=3, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),  # 层3

            keras.layers.Conv2D(filters=128, kernel_size=3, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
            keras.layers.Dropout(dropoutrate),  # 层4

            keras.layers.Conv2D(filters=256, kernel_size=3, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),  # 层5

            keras.layers.Conv2D(filters=256, kernel_size=3, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),  # 层6

            keras.layers.Conv2D(filters=256, kernel_size=3, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
            keras.layers.Dropout(dropoutrate),  # 层7

            keras.layers.Conv2D(filters=512, kernel_size=3, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),  # 层8

            keras.layers.Conv2D(filters=512, kernel_size=3, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),  # 层9

            keras.layers.Conv2D(filters=512, kernel_size=3, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
            keras.layers.Dropout(dropoutrate),  # 层10

            keras.layers.Conv2D(filters=512, kernel_size=3, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),  # 层11

            keras.layers.Conv2D(filters=512, kernel_size=3, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),  # 层12

            keras.layers.Conv2D(filters=512, kernel_size=3, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
            keras.layers.Dropout(dropoutrate),  # 层13

            keras.layers.Flatten(),

            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(6, activation='softmax'),

        ])

    def call(self, inputs):
        output = self.mod(inputs)
        return output


def main():
    print('start')
    print(len(tf.config.experimental.list_physical_devices('GPU')))
    Mymodel = mymodel()
    num = 0
    acc = 0
    history=[]

    if sys.argv[1] == 'train':
        print('train')
        cp_callback = keras.callbacks.ModelCheckpoint(filepath=savepath, save_weight_only=True, perioh=1)
        Mymodel.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                        loss=losses.SparseCategoricalCrossentropy(from_logits=False),
                        metrics=['sparse_categorical_accuracy'])
        if os.path.exists(savepath):
            print("load model")
            Mymodel = keras.models.load_model(savepath)

        for i in range(epochs):
            print(i)
            list = loadseq()
            for j in range(85):
                curlist = list[num:num + batchsize]
                num += batchsize
                x_trainlist, y_trainlist = get_batchlist(curlist)
                x_train, y_train = get_batch(x_trainlist, y_trainlist)
                print(x_train.shape)
                print(y_train.shape)
                curhistory=Mymodel.fit(x_train, y_train, batch_size=20, epochs=1,callbacks=[cp_callback])
                history.append(curhistory)
            if (i >= 10):
                list1 = load_testseq()
                print(len(list1))
                x_testlist, y_testlist = get_testbatchlist(list1)
                right = 0
                for k in range(len(x_testlist)):
                    x_test, y_test = get_testbatch(x_testlist[k:k + 1], y_testlist[k:k + 1])
                    result = Mymodel.predict(x_test)
                    result = np.argmax(result)
                    print(result, ':', y_test[0])
                    if result == y_test[0]:
                        right += 1
                curacc = right / len(x_testlist)
                if (curacc > acc):
                    acc = curacc
                    keras.models.save_model(Mymodel, bestpath)
                print('curacc=', curacc)

            num = 0
        acclist=[]
        losslist=[]
        for a in history:
            acclist.append(a.history['sparse_categorical_accuracy'][0])
            losslist.append(a.history['loss'][0])
        plt.subplot(1,2,1)
        plt.plot(acclist,label='acc')
        plt.title('acc')
        plt.legend()
        plt.xticks(alpha=0)
        plt.subplot(1, 2, 2)
        plt.plot(losslist, label='loss')
        plt.title('loss')
        plt.legend()
        plt.xticks(alpha=0)
        plt.savefig('./accloss.jpg',dpi=300)
        print("finish!")

    elif sys.argv[1] == 'test':
        print('test')
        if os.path.exists(bestpath):
            print("load model")
            right = 0
            acc = 0
            Mymodel = keras.models.load_model(bestpath)
            list = load_testseq()
            print(len(list))
            x_testlist, y_testlist = get_testbatchlist(list)
            map=[[0 for n in range(6)] for m in range(6)]

            for i in range(len(x_testlist)):
                x_test, y_test = get_testbatch(x_testlist[i:i + 1], y_testlist[i:i + 1])
                result = Mymodel.predict(x_test)
                result = np.argmax(result)
                print(result, ':', y_test[0])
                map[result][int(y_test[0])]+=1
                if result == y_test[0]:
                    right += 1
            acc = right / len(x_testlist)
            print('acc=', acc)
            for n in range(6):
                print(map[n])






    else:
        print('error!')
        return


main()
