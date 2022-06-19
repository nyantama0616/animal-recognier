
# 動画からフレームを取り出すファイル

import cv2
import os

def save_all_frames(video_path, dir_path, basename, ext='jpg'):
    cap = cv2.VideoCapture(video_path)
    MOD = 3 # ここで取り出すフレームの間隔を指定

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0
    i = 0
    while True:
        ret, frame = cap.read()
        if ret:
            if n % MOD == 0:
                cv2.imwrite('{}_{}.{}'.format(base_path, str(i).zfill(digit), ext), frame)
                i += 1
        else:
            return
        n += 1

save_all_frames('./mov/kabi.mov', './img/kabi/', 'kabi')
