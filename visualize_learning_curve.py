# モデルの学習曲線を可視化するプログラム
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import numpy as np
import keras
import matplotlib.pyplot as plt

classes = ['horse', 'zebra']
num_classes = len(classes)
image_size = 64
epochs = 56

#メインの関数を定義する
def main():
    X_train, X_test, y_train, y_test = np.load('./npy/animal.npy', allow_pickle=True)
    #画像ファイルの正規化
    X_train = X_train.astype('float') / 255
    X_test = X_test.astype('float') / 255
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    model = model_train(X_train, y_train, X_test, y_test)
    model_eval(model, X_test, y_test)

# さっぱり意味がわからないけど、ニューロンネットワークを構築し、fitで学習してるんだと思う
def model_train(X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
              input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    #最適化の手法
    opt = keras.optimizers.rmsprop(lr=0.00005, decay=1e-6)

    #モデルのコンパイル
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])

    #historyに['val_loss', 'val_acc', 'loss', 'acc']を保存
    history = model.fit(X_train, y_train, batch_size=32,
                        epochs=epochs, validation_data=(X_test, y_test))

    #モデルの保存
    model.save('./model/animal_cnn.h5')

    #学習曲線の可視化
    graph_general(history)

    return model


def model_eval(model, X_test, y_test):
    scores = model.evaluate(X_test, y_test, verbose=1)
    print('Test Loss: ', scores[0])
    print('Test Accuracy: ', scores[1])

# 多分ここでグラフを表示してる
def graph_general(history):
    # Plot training & validation accuracy values
    # plt.plot(history.history['acc'])
    plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_acc'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train_acc', 'Test_acc'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    print("train: ", history.history['loss'])
    print("val: ", history.history['val_loss'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train_loss', 'Test_loss'], loc='upper left')
    plt.show()

if __name__ == '__main__':
    main()
