#任意の画像を分類する。
from keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
import numpy as np
from sklearn import model_selection

img_size = 64
horse_num = 8
zebra_num = 8
model_param = "./model/animal_cnn.h5"

def load_image(img_path):
    img = Image.open(img_path)
    img = img.convert('RGB')
    #画像データを64 x 64に変換
    img = img.resize((img_size, img_size))
    # 画像データをnumpy配列に変換
    img = np.asarray(img)
    img = img / 255.0
    return img

def judge(img_path):
    model = load_model(model_param)
    img = load_image(img_path)
    pred = model.predict(np.array([img]))
    print(round(pred[0][0] / 1.0, 3) * 100, "%",
        round(pred[0][1] / 1.0, 3) * 100, "%")
    ans = np.argmax(pred, axis=1)
    if ans == 0:
        print(">>> 馬")
    elif ans == 1:
        print(">>> シマウマ")
    
    return ans
    
v = [0, 0]
for i in range(horse_num):
    ans = int(judge("./img/test/horse/{}.png".format(i + 1)))
    v[ans] += 1

print("馬: {}, シマウマ: {}".format(v[0], v[1]))
print("-------------------------------------------")

v = [0, 0]
for i in range(zebra_num):
    ans = int(judge("./img/test/zebra/{}.png".format(i + 1)))
    v[ans] += 1
print("馬: {}, シマウマ: {}".format(v[0], v[1]))
