# !/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Created on NOV 24, 2020

@author: woshihaozhaojun@sina.com
"""
import os
from tqdm import tqdm
import json
from gensim.models import word2vec
input_dir = "click_seq_json"
output_file = "corpora/1.txt"
model_path = "model/word2vec"
vid2title_path = "map_json/vid2title.json"
target = "mzc00200ydcnajl"


def _convert_json_to_ssf(input_dir, output_file):
    """把json文件转化为word2vec模型可读得space separated file
    用于feed word2vec.LineSentence接口
    """
    out_fp = open(output_file, "w")
    for file in os.listdir(input_dir):
        input_file = os.path.join(input_dir, file)
        with open(input_file, "r") as fp:
            lines = fp.readlines()
            for line in tqdm(lines):
                val = json.loads(line).get("value", "")
                if val:
                    out_fp.write(val + "\n")
    out_fp.close()


def _get_vid_to_title(file_path):
    """生成视频id和名字的map
    """
    vid2title = dict()
    with open(file_path, "r", encoding="utf8") as fp:
        lines = fp.readlines()
        for line in tqdm(lines):
            dict_tmp = json.loads(line)
            if "cover_id" in dict_tmp and "title" in dict_tmp:
                vid = dict_tmp["cover_id"]
                title = dict_tmp["title"]
                vid2title[vid] = title
    return vid2title


_convert_json_to_ssf(input_dir, output_file)
vid2title = _get_vid_to_title(vid2title_path)
sentences = word2vec.LineSentence(output_file)
model = word2vec.Word2Vec(sentences, hs=1, min_count=5, window=5, size=100)
model.save(model_path)
res = model.similar_by_word(target, topn=20)
print(vid2title[target])
for item in res:
    print(vid2title.get(item[0], "未知"), item[1])

