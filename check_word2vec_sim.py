# -*- coding: UTF-8 -*-
"""
Created on NOV 25, 2020

@author: woshihaozhaojun@sina.com
"""
from gensim.models import word2vec
from word2vec_demo import (
    get_vid_to_title, model_path, vid2title_path, find_similar
)


target = input("请输入希望查看的视频的vid: ")
model = word2vec.Word2Vec.load(model_path)
vid2title = get_vid_to_title(vid2title_path)
find_similar(model, target, vid2title)



