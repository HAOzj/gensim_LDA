import os
import sys 
import jieba
from gensim.corpora import MmCorpus
import gensim
from openpyxl import load_workbook
import xlrd
import math
import re
from text_normalisation import textNormalisation
def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

################# 开始基于语料文件建设两种LDA模型并保存模型 #################
###################################################################  start
def create_corporaListAndCorporaText(corpora_source, corpora_txt, id2word_fname):
    ''' 从{corpora_source}文件夹下提取所有.txt文件作为语料

    每个以'text'分割的条目作为一行,存入{corpora_txt}文件

    并保存id2word到{id2word_fname}文件

    Args:
        corpora_source(path) :- 语料文件所在的文件夹
        corpora_txt(path) :- 汇总所有语料的.txt文件
        id2word_fname(path) :- gensim的字典文件
    '''
    corpora = []
    if not os.path.isdir(corpora_source):
        print(corpora_source, "doesn't exist")
        raise FileNotFoundError

    if not os.path.isdir( os.path.dirname(corpora_txt) ):
        print(os.path.dirname(corpora_text)," doesn't exist")
        raise FileNotFoundError
    if not os.path.isdir( os.path.dirname(os.path.dirname(id2word_fname))):
        print("the grandparent directory of ",id2word_fname, " doesnt't exist")
        raise FileNotFoundError

    output_tfidf = open(corpora_txt, 'a')
    for file in os.listdir(corpora_source):
        if file.endswith('txt'):
            print(corpora_source+file,' read')
            with open(corpora_source+file) as f:
                lines = f.readlines()
                for line in lines:
                    words = textNormalisation(line)
                    if len(words)>0 :
                        corpora.append(words) 
                        output_tfidf.write('{}\n'.format(words))
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



def createAndSave_lda_bow(corpora_txt, id2word_fname, ldaModel_save_repo, num_topics =10):
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
    if not os.path.isdir( ldaModel_save_repo) :
        print(ldaModel_save_repo, "doesn't exist")
        raise FileNotFoundError        

    corpora = []
    with open(corpora_txt, 'r') as fp:
        lines = fp.readlines()
        for line in lines :
            corpora.append(line.strip())
    id2word = gensim.corpora.Dictionary.load(id2word_fname)
    corpus = [id2word.doc2bow(corpus) for corpus in corpora]
    lda_bow = gensim.models.LdaModel(corpus= corpus, id2word= id2word, num_topics= num_topics)

    make_dir(ldaModel_save_repo+'/gensim_bow')
    if not os.path.isfile(ldaModel_save_repo+'/gensim_bow/QA_JQW.model'):
        lda_bow.save( ldaModel_save_repo+'/gensim_bow/QA_JQW.model' )
        print('lda_bow saved')
    else :
        print(ldaModel_save_repo,'/gensim_bow/QA_JQW.model already exists')
      
        
def createAndSave_lda_tfidf(corpora, id2word, ldaModel_save_repo, num_topics = 10):
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
    if not os.path.isdir(ldaModel_save_repo) :
        print(ldaModel_save_repo, "doesn't exist")
        raise FileNotFoundError 

    corpora = []
    with open(corpora_txt, 'r') as fp:
        lines = fp.readlines()
        for line in lines :
            corpora.append(line.strip())
    id2word = gensim.corpora.Dictionary.load(id2word_fname)

    MmCorpus.serialize('corpus_tfidf.mm', [id2word.doc2bow(corpus) for corpus in corpora])  
    mm = MmCorpus('corpus_tfidf.mm')
    lda_tfidf = gensim.models.LdaModel(corpus= mm, id2word= id2word, num_topics= num_topics)

    make_dir(ldaModel_save_repo+'/gensim_tfidf')
    if not os.path.isfile(ldaModel_save_repo+'/gensim_tfidf/QA_JQW.model'):
        lda_tfidf.save( ldaModel_save_repo+'/gensim_tfidf/QA_JQW.model' )
        print('lda_tfidf saved')
    else :
        print(ldaModel_save_repo, '/gensim_tfidf/QA_JQW.model already exists')

def analysis_topics(fname):
    '''将各个主题的关键字打印出来
    '''
    f = open(fname, 'r')
    lines = f.readlines()
    for line in lines:
        print(re.findall(r'\"([^\"]*)\"', line))

def short_long_similarity(lda_fname, id2word_fname, short, long):
    '''计算长短文本的相似度

    Args:
        lda_fname(path) :- gensim.models.ldamodel的保存路径
        id2word_fnmae(path) :- gensim.corpora.Dictionary的保存路径
        short(str) :- 短文本
        long(str) :- 长文本
    Returns:
        prob(float) :- 长短文本的匹配度
        Theta(iterables) :- 长文本在lda模型下的主题分布概率,
                            每个元素为(主题的序号, 对应主题的概率)
    '''
    lda = gensim.models.LdaModel.load(lda_fname)
    id2word = gensim.corpora.Dictionary.load(id2word_fname)
    Theta = lda[ id2word.doc2bow( _wordNormalisation(long)) ]
    short = _wordNormalisation(short)
    short = set(short)
    prob = 1.0
    short = id2word.doc2idx(short)
    prob = 0
    for word in short :
        prob_w = sum([lda.expElogbeta[k][word]*1000 * p_zk for (k,p_zk) in Theta])
        prob +=  math.log(prob_w)
    prob = prob/ len(short)
    prob -= math.log(1000)
    return prob, Theta
#####################################################################  end

