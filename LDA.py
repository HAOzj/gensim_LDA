# -*- coding: UTF-8 -*-
import os
import re
from gensim.corpora import MmCorpus
import gensim
from Normalizer import (Normalizer, print_run_time, make_dir)

        
class LDA():
    def __init__(self, vectorizer = "tfidf", stopWordsFile=None):
        self.vectorizer = vectorizer
        self.stopWordsFile = stopWordsFile
        self.normalizer = Normalizer(self.stopWordsFile)
        self.normalizer.load_stopWords()
        
    @print_run_time
    def create_corporaListAndCorporaText(self, corpora_source, corpora_txt, id2word_fname):
        self.corpora_source = corpora_source
        self.corpora_txt = corpora_txt
        self.id2word_fname = id2word_fname
        self._create_corporaListAndCorporaText(self.normalizer, self.corpora_source, self.corpora_txt, self.id2word_fname)
        
    @print_run_time
    def createAndSave_lda(self, ldaModel_save_repo, num_topics):
        self.ldaModel_save_repo = ldaModel_save_repo
        self.num_topics = num_topics
        if self.vectorizer == "tfidf":
            self.model = self._createAndSave_lda_tfidf(self.corpora_txt, self.id2word_fname, self.ldaModel_save_repo, self.num_topics)
        else:
            self.model = self._createAndSave_lda_bow(self.corpora_txt, self.id2word_fname, self.ldaModel_save_repo, self.num_topics)

    @staticmethod
    def _create_corporaListAndCorporaText(normalizer, corpora_source, corpora_txt, id2word_fname):
        ''' 从{corpora_source}文件夹下提取所有.txt文件作为语料

        文件总每一行经过预处理后作为一行,存入{corpora_txt}文件

        并保存id2word到{id2word_fname}文件

        Args:
            corpora_source(path) :- 语料文件所在的文件夹
            corpora_txt(path) :- 汇总所有语料的.txt文件
            id2word_fname(path) :- gensim的字典文件
        '''
        corpora = []
        if not os.path.isdir(corpora_source):
            raise OSError(corpora_source, "doesn't exist")

        if not os.path.isdir(os.path.dirname(corpora_txt)):
            raise OSError(os.path.dirname(corpora_txt), " doesn't exist")

        if not os.path.isdir( os.path.dirname(os.path.dirname(id2word_fname))):
            raise OSError("the grandparent directory of ", id2word_fname, " doesnt't exist")

        output_tfidf = open(corpora_txt, 'a', encoding="utf8")
        for file in os.listdir(corpora_source):
            if file.endswith('txt'):
                file = os.path.join(corpora_source, file)
                print(file+' read')
                with open(file, encoding="utf8") as f:
                    lines = f.readlines()
                    for line in lines:
                        words = normalizer.tokenize(line)
                        if len(words) > 0 :
                            corpora.append(words) 
                            output_tfidf.write('{}\n'.format(" ".join(words)))
                f.close()

        output_tfidf.close()    
        id2word = gensim.corpora.Dictionary(corpora)

        parent_dir = os.path.dirname(id2word_fname)
        make_dir( parent_dir )
        if not os.path.isfile(id2word_fname):
            id2word.save(id2word_fname) 
            print('id2word saved') 
        else:
            print(id2word_fname,' already exists')


    @staticmethod
    def _createAndSave_lda_bow(corpora_txt, id2word_fname, ldaModel_save_repo, num_topics =10):
        '''  训练和保存基于bow的lda模型

        基于{corpora_txt}文件保存的语料和{id2word_fname}保存的gensim字典来训练lda_bow模型,

        主题数为{num_topics}

        保存该模型到{ldaModel_save_repo}文件夹下

        Args:
            corpora_txt(path) :- 保存语料的.txt文件
            id2word_fname(path) :- 保存gensim字典的文件
            ldaModel_save_repo(path) ：- 保存gensim LDA模型的文件夹
            num_topics(int) :- lda的超参,主题数
        '''
        if not os.path.isdir(ldaModel_save_repo):
            raise OSError(ldaModel_save_repo, "doesn't exist")


        corpora = []
        with open(corpora_txt, 'r', encoding="utf8") as fp:
            lines = fp.readlines()
            for line in lines :
                corpora.append(line.strip())
        id2word = gensim.corpora.Dictionary.load(id2word_fname)
        corpus = [id2word.doc2bow(corpus.split(" ")) for corpus in corpora]
        lda_bow = gensim.models.LdaModel(corpus= corpus, id2word= id2word, num_topics= num_topics)

        make_dir(ldaModel_save_repo+'/gensim_bow')
        if not os.path.isfile(ldaModel_save_repo+'/gensim_bow/crawl_news.model'):
            lda_bow.save(ldaModel_save_repo+'/gensim_bow/crawl_news.model')
            print('lda_bow saved')
        else :
            print(ldaModel_save_repo,'/gensim_bow/crawl_news.model already exists')

        return lda_bow

    @staticmethod
    def _createAndSave_lda_tfidf(corpora_txt, id2word_fname, ldaModel_save_repo, num_topics = 10):
        '''  训练和保存基于tfidf的lda模型

        基于{corpora_txt}文件保存的语料和{id2word_fname}保存的gensim字典来训练lda_tfidf模型,

        主题数为{num_topics}

        保存该模型到{ldaModel_save_repo}文件夹下

        Args:
            corpora_txt(path) :- 保存语料的.txt文件
            id2word_fname(path) :- 保存gensim字典的文件
            ldaModel_save_repo(path) ：- 保存gensim LDA模型的文件夹
            num_topics(int) :- lda的超参,主题数
        '''
        if not os.path.isdir(ldaModel_save_repo):
            raise OSError(ldaModel_save_repo, "doesn't exist")

        corpora = []
        with open(corpora_txt, 'r', encoding="utf8") as fp:
            lines = fp.readlines()
            for line in lines :
                corpora.append(line.strip())
        id2word = gensim.corpora.Dictionary.load(id2word_fname)
        

        MmCorpus.serialize('corpus_tfidf.mm', [id2word.doc2bow(corpus.split(" ")) for corpus in corpora])
        mm = MmCorpus('corpus_tfidf.mm')
        lda_tfidf = gensim.models.LdaModel(corpus= mm, id2word= id2word, num_topics= num_topics)

        make_dir(ldaModel_save_repo+'/gensim_tfidf')
        if not os.path.isfile(ldaModel_save_repo+'/gensim_tfidf/crawl_news.model'):
            lda_tfidf.save( ldaModel_save_repo+'/gensim_tfidf/crawl_news.model' )
            print('lda_tfidf saved')
        else:
            print(ldaModel_save_repo, '/gensim_tfidf/crawl_news.model already exists')
        return lda_tfidf

    @staticmethod
    def analysis_topics(fname):
        '''将各个主题的关键字打印出来
        '''
        f = open(fname, 'r')
        lines = f.readlines()
        for line in lines:
            print(re.findall(r'\"([^\"]*)\"', line))
