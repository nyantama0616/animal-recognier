# ダウンロードした画像をnumpyのデータ型に変換し、一つのファイルにまとめるファイル

from PIL import Image
import os
import glob
import numpy as np
from sklearn import model_selection

classes = ['horse', 'zebra']
num_classes = len(classes)
image_size = 64
max_img_num = 1000

#画像の読み込み
X = [] #画像データ
Y = [] #ラベルデータ

for index, class_ in enumerate(classes):
    photos_dir = './img/train/' + class_ + "/" #事前に、./img/train/horseっみたいなフォルダ作る必要あり

    #jpg形式の画像データを取得
    files = glob.glob(photos_dir + '*.jpg')

    #フォルダ内の全ての画像を１つずつ渡す
    for i, file in enumerate(files):
        #画像データがmax_img_numを超えたらループを抜ける
        if i >= max_img_num:
            break
        image = Image.open(file)
        image = image.convert('RGB')
        #画像データを64 x 64に変換
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        X.append(data)
        Y.append(index)

X = np.array(X)
Y = np.array(Y)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.3)
xy = (X_train, X_test, y_train, y_test)
np.save('./npy/animal.npy', xy)
