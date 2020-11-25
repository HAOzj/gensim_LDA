# -*- coding: UTF-8 -*-
import os
import time
import jieba


def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


Q2B_DICT = {
    "12288": 32,  # 全角空格直接转换
    "12290": 46,  # 。-> .
    "12304": 91,  # 【 -> [
    "12305": 93,  # 】-> ]
    "8220": 34,  # “ -> "
    "8221": 34  # ” -> "
}


def print_run_time(func):
    """ 计算时间函数
    """

    def wrapper(*args, **kw):
        local_time = time.time()
        res = func(*args, **kw)
        print('Current function : {function}, time used : {temps}'.format(
            function=func.__name__, temps=time.time() - local_time)
        )
        return res

    return wrapper


class Normalizer(object):
    """文本标准化的类

    """
    def __init__(self, stopwords_path):
        self.stopwords_path = stopwords_path

        self.stopwords = "《 ， 。 ． 》 ; ［ ］（ ） “ ”  ！ ＆ \n \t \ / -' ( ) . text ' …  x000d - > x000d".split(' ')
        self.stopwords.append('"')
        self.stopwords.append(' ')

    @print_run_time
    def load_stopwords(self):
        if not os.path.isfile(self.stopwords_path):
            raise OSError(self.stopwords_path, "doesn't exist")

        with open(self.stopwords_path, 'r', encoding="utf8") as f:
            lines = f.readlines()
            for line in lines:
                line = self.conversion(line)
                self.stopwords.append(line.replace('\n', ''))

    @staticmethod
    def load_userdict_from_dir(term_dir):
        """收集术语.txt来建设字典"""
        if not os.path.isdir(term_dir):
            raise OSError(term_dir, "doesn't exist")
        for name in os.listdir(term_dir):
            if name.endswith("txt"):
                term_file = os.path.join(term_dir, name)
                jieba.load_userdict(term_file)
                print("load userdict : {}".format(term_file))

    def _DBC2SBC(self, ustring):
        """全角转半角
        """
        ss = ""

        for s in ustring:
            inside_code = ord(s)
            if str(inside_code) in Q2B_DICT:
                inside_code = Q2B_DICT[str(inside_code)]
            elif 65281 <= inside_code <= 65374:  # 全角字符根据关系转化
                inside_code -= 65248

            ss += chr(inside_code)
        return ss

    def conversion(self, string):
        """ 全半角\大小写转化
        """
        return self._DBC2SBC(string).lower()

    def tokenize(self, string):
        """ 文本标准化

        1. 全角转半角
        2. 大写转小写
        3. 分词后去stop words
        """
        string = self.conversion(string)
        string_list = list(jieba.cut(string, cut_all=False))
        string_list = [word for word in string_list if word not in self.stopwords]
        string_list = [word for word in string_list if not word.isspace()]
        return string_list
