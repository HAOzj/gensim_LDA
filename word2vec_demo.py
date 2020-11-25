# -*- coding: UTF-8 -*-
"""
Created on NOV 24, 2020

@author: woshihaozhaojun@sina.com
"""
import os
from Normalizer import print_run_time
from tqdm import tqdm
import json
from gensim.models import word2vec


@print_run_time
def _convert_json_to_ssf(input_dir, output_file):
    """把json文件转化为word2vec模型可读得space separated file
    用于feed word2vec.LineSentence接口
    """
    out_fp = open(output_file, "w")
    print("---start processing click sequence files---")
    for file in os.listdir(input_dir):
        input_file = os.path.join(input_dir, file)
        print("load", input_file)
        with open(input_file, "r") as fp:
            lines = fp.readlines()
            for line in tqdm(lines):
                click_seq = json.loads(line).get("click_seq", "")
                if click_seq:
                    click_seq = [vid.split("_")[0] for vid in click_seq]
                    out_fp.write(" ".join(click_seq) + "\n")
    out_fp.close()


@print_run_time
def get_vid_to_title(file_path):
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


@print_run_time
def load_sentence(output_file):
    return word2vec.LineSentence(output_file)


@print_run_time
def train_model(sentences):
    return word2vec.Word2Vec(sentences, hs=1, min_count=5, window=5, size=32)


@print_run_time
def find_similar(model, target, vid2title):
    res = model.similar_by_word(target, topn=20)
    print(vid2title[target])
    for item in res:
        print(vid2title.get(item[0], "未知"), item[1])


input_dir = "click_seq_json"
output_file = "corpora/1.txt"
model_path = "model/word2vec"
vid2title_path = "map_json/vid2title.json"
target = "mzc00200ydcnajl"


def main():
    _convert_json_to_ssf(input_dir, output_file)
    vid2title = get_vid_to_title(vid2title_path)
    sentences = load_sentence(output_file)
    model = train_model(sentences)
    model.save(model_path)
    find_similar(model, target, vid2title)


if __name__ == "__main__":
    main()

