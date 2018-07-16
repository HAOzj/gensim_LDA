import os 
import jieba

def load_userdict_from_dir(term_dir):
    '''收集术语.txt来建设字典'''
    if not os.path.isdir(term_dir):
        print(term_dir, "doesn't exist")
        raise FileNotFoundError
    terminology = []
    for name in os.listdir(term_dir):
        if name.endswith("txt"):
            term_file = os.path.join(term_dir, name) 
            jieba.load_userdict(term_file)
            print("load userdict : {}".format(term_file))

Q2B_DICT = {
    "12288" : 32, # 全角空格直接转换
    "12290" : 46, # 。-> .
    "12304" : 91, #  【 -> [
    "12305" : 93, # 】-> ]
    "8220" : 34, # “ -> "
    "8221" : 34 # ” -> "
}

def _DBC2SBC(ustring):
    '''全角转半角
    '''
    ss = ""

    for s in ustring:
        inside_code = ord(s)
        if str(inside_code) in Q2B_DICT:
            inside_code = Q2B_DICT[str(inside_code)]

        elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符根据关系转化
            inside_code -= 65248

        ss += chr(inside_code)
    return ss

def _conversion(string):
    ''' 全半角\大小写转化
    '''
    return _DBC2SBC(string).lower()

def _stopWords(stopWords_fname ):
    if not os.path.isfile(stopWords_fname):
        print(stopWords_fname, "doesn't exist")
        raise FileNotFoundError
    stopWords = "《 ， 。 ． 》 ; ［ ］（ ） “ ”  ！ ＆ \n \t \ / -' ( ) . text ' …  x000d - > x000d".split(' ')
    stopWords.append('"')
    with open(stopWords_fname, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = _conversion(line)
            stopWords.append(line.replace('\n',''))
    stopWords.append(' ')
    return stopWords

def textNormalisation(string, stopWords) :
    ''' 文本自动化

    1. 全角转半角
    2. 大写转小写
    3. 分词后去stop words
    4. 去掉空白
    '''
    string = _conversion(string)
    string_list = list(jieba.cut(string, cut_all = False))
    string_list = [word for word in string_list if word not in stopWords]
    string_list = [word for word in string_list if not word.isspace()]   
    return string_list

