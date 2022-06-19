# flickrというサイトから、動物の画像を大量にダウンロードするファイル
# 実行前に、horse, zebraの画像を格納する空のフォルダを作成する必要あり

from flickrapi import FlickrAPI
from urllib.request import urlretrieve
from pprint import pprint
import os,time

# APIのキーは商用・非商用に注意
key = "0d9d3ceba4836b1baa029457b30ebb2c"
secret = "8bd77f5a93928163"

wait_time = 1

def get_photos(animal_name):
    savedir = "./img/" + animal_name  + "/"

    flickr = FlickrAPI(key, secret, format="parsed-json")
    result = flickr.photos.search(
        #検索キーワード
        text = animal_name,
        #取得したい数
        per_page = 500,
        #検索するデータの種類
        media = "photos",
        #データのソート順（関連度の高い順）
        sort = "relevance",
        #有害コンテンツの除去
        safe_search = 1,
        #取得したいオプション値
        extras = "url_q, license"
    )
    photos = result["photos"]
    print(photos["photo"])

    for i, photo in enumerate(photos["photo"]):
        try:
            url_q = photo["url_q"]
        except:
            print("取得に失敗しました")
            continue

        filepath = savedir + photo["id"] + ".jpg"
        if os.path.exists(filepath): continue
        urlretrieve(url_q, filepath)
        time.sleep(wait_time)

start = time.time()

get_photos("horse")
get_photos("zebra")

print("処理時間", (time.time() - start), "秒")
