# -*- coding: UTF-8 -*-
import os
import re
from gensim.corpora import MmCorpus
import gensim
import math
from Normalizer import (Normalizer, print_run_time, make_dir)

corpus_tfidf_mm = "corpus_tfidf.mm"
corpora_dir = "crawl_news"  # 爬取的新闻都放在该文件夹下,每个新闻一个.txt文件
corpora_path = "corpora.txt"
id2word_path = "id2word"
model_dir = "model"
model_fname = "crawl_news.model"
num_topics = 10

        
class LDA(object):
    def __init__(self, vectorizer="tfidf", stopwords_path=None):
        self.vectorizer = vectorizer
        self.stopwords_path = stopwords_path
        self.normalizer = Normalizer(self.stopwords_path)
        self.normalizer.load_stopwords()
        
    @print_run_time
    def tranform_corpora(self, corpora_dir, corpora_path, id2word_path):
        """转化语料

        1. 从{corpora_dir}文件夹下提取所有.txt文件作为语料
        2. 文件总每一行经过预处理后作为一行,存入{corpora_path}文件
        3. 保存id2word到{id2word_path}文件
        """
        self.corpora_dir = corpora_dir
        self.corpora_path = corpora_path
        self.id2word_path = id2word_path
        self._transform_corpora(self.normalizer, self.corpora_dir, self.corpora_path, self.id2word_path)
        
    @print_run_time
    def train_lda(self, model_dir, model_fname, num_topics):
        self.model_dir = model_dir
        self.model_fname = model_fname
        self.num_topics = num_topics
        self.model = self._train_lda(self.vectorizer, self.corpora_path, self.id2word_path, 
                                     self.model_dir, self.model_fname, self.num_topics)
        self.model_path = os.path.join(self.model_dir, self.vectorizer, self.model_fname)

    @staticmethod
    def _transform_corpora(normalizer, corpora_dir, corpora_path, id2word_path):
        """转化语料

        1. 从{corpora_dir}文件夹下提取所有.txt文件作为语料
        2. 文件总每一行经过预处理后作为一行,存入{corpora_path}文件
        3. 保存id2word到{id2word_path}文件

        Args:
            corpora_dir(path) :- 语料文件所在的文件夹
            corpora_path(path) :- 汇总所有语料的.txt文件
            id2word_path(path) :- gensim的字典文件
        """
        corpora = []
        if not os.path.isdir(corpora_dir):
            raise OSError(corpora_dir, "doesn't exist")

        if not os.path.isdir(os.path.dirname(corpora_path)):
            raise OSError(os.path.dirname(corpora_path), " doesn't exist")

        if not os.path.isdir(os.path.dirname(os.path.dirname(id2word_path))):
            raise OSError("the grandparent directory of ", id2word_path, " doesnt't exist")

        output_tfidf = open(corpora_path, 'a', encoding="utf8")
        for file in os.listdir(corpora_dir):
            if file.endswith('txt'):
                file = os.path.join(corpora_dir, file)
                print(file+' read')
                with open(file, encoding="utf8") as f:
                    lines = f.readlines()
                    for line in lines:
                        words = normalizer.tokenize(line)
                        if len(words) > 0:
                            corpora.append(words) 
                            output_tfidf.write('{}\n'.format(" ".join(words)))
                f.close()

        output_tfidf.close()    
        id2word = gensim.corpora.Dictionary(corpora)

        parent_dir = os.path.dirname(id2word_path)
        make_dir(parent_dir)
        if not os.path.isfile(id2word_path):
            id2word.save(id2word_path) 
            print('id2word saved') 
        else:
            print(id2word_path, ' already exists')

    @staticmethod
    def _train_lda(vectorizer, corpora_path, id2word_path, model_dir, model_fname=model_fname, num_topics=10):
        """训练和保存基于tfidf的lda模型

        基于{corpora_path}文件保存的语料和{id2word_path}保存的gensim字典来训练lda_tfidf模型,

        保存该模型到{model_dir}文件夹下

        Args:
            vectorizer(str) :- 向量化方法, choices=["bow", "tfidf"]
            corpora_path(path) :- 保存语料的.txt文件
            id2word_path(path) :- 保存gensim字典的文件
            model_dir(path) :- 保存gensim LDA模型的文件夹
            model_fname(path) :- 模型文件名
            num_topics(int) :- lda的超参,主题数
        """
        try:
            assert vectorizer in ["bow", "tfidf"]
        except AssertionError:
            raise AssertionError("vectorizer must be bow or tfidf")
        
        if not os.path.isdir(model_dir):
            raise OSError(model_dir, "doesn't exist")

        corpora = []
        with open(corpora_path, 'r', encoding="utf8") as fp:
            lines = fp.readlines()
            for line in lines:
                corpora.append(line.strip())
        id2word = gensim.corpora.Dictionary.load(id2word_path)
        corpus = [id2word.doc2bow(corpus.split(" ")) for corpus in corpora]
        
        # tfidf的话需要计算idf
        if vectorizer == "tfidf":
            MmCorpus.serialize(corpus_tfidf_mm, corpus)
            corpus = MmCorpus(corpus_tfidf_mm)
            
        model = gensim.models.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics)

        model_path = os.path.join(model_dir, vectorizer)
        make_dir(model_path)
        model_path = os.path.join(model_path, model_fname)
        if not os.path.isfile(model_path):
            model.save(model_path)
            print('model saved')
        else:
            print(f"{model_path} already exists")
        return model

    @staticmethod
    def analysis_topics(fname):
        """将各个主题的关键字打印出来

        把self.model.print_topics(10)保存到fname后打印出来
        """
        f = open(fname, 'r')
        lines = f.readlines()
        for line in lines:
            print(re.findall(r'\"([^\"]*)\"', line))

    @staticmethod
    def _short_long_similarity(model_path, normalizer, id2word_path, short, long):
        """计算长短文本的相似度

        Args:
            model_path(path) :- gensim.models.ldamodel的保存路径
            id2word_path(path) :- gensim.corpora.Dictionary的保存路径
            short(str) :- 短文本
            long(str) :- 长文本
        Returns:
            prob(float) :- 长短文本的匹配度
            theta(iterables) :- 长文本在lda模型下的主题分布概率,
                                每个元素为(主题的序号, 对应主题的概率)
        """
        lda = gensim.models.LdaModel.load(model_path)
        id2word = gensim.corpora.Dictionary.load(id2word_path)
        theta = lda[id2word.doc2bow(normalizer.tokenize(long))]
        short = normalizer.tokenize(short)
        short = set(short)
        short = id2word.doc2idx(short)
        prob = 0
        for word in short:
            prob_w = sum([lda.expElogbeta[k][word]*1000 * p_zk for (k, p_zk) in theta])
            prob += math.log(prob_w)
        prob = prob/len(short)
        prob -= math.log(1000)
        return prob, theta

    def short_long_sim(self, short, long):
        """用self.model计算长短文本相似度
        """
        return self._short_long_similarity(self.model_path, self.normalizer, self.id2word_path, short, long)


def main():
    lda = LDA()
    lda.tranform_corpora(corpora_dir, corpora_path, id2word_path)
    lda.train_lda(model_dir, model_fname, num_topics)
    lda.model.print_topics()
    lda.short_long_sim(short="好的", long="我的鬼鬼呦，你在说神马呢")


if __name__ == "__main__":
    main()

