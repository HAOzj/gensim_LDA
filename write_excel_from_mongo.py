"""
Created on the 25th Dec, 2017

@author: woshihaozhaojun@sina.com 
"""
from pymongo import MongoClient
import xlsxwriter
import xlwt


def write_xlsx_from_mongo(host, database, collections, xlsx_fname):
    """ 从Mongo中提取数据存入.xlsx文件

    Args:
        host(path) :- mongo服务器地址
        database(str) :- mongo服务器中一级目录
        collections(dict) :- (collection, 该collection的keys列表)的(key,value)对
        xlsx_fname(path) :- 要存入的excel文件地址
    """
    client = MongoClient(host)
    db = client[database]

    wb = xlsxwriter.Workbook(xlsx_fname)
    for collection, keys in collections.items():
        num_col = len(keys)
        ws = wb.add_worksheet(collection)
        row = 0

        for item in db[collection].find():
            for col in range(num_col):
                if isinstance(item[keys[col]], list):
                    cell = " ".join(item[keys[col]])
                else:
                    cell = item[keys[col]]
                ws.write(row, col, cell )
            row += 1
    wb.close()

    
def write_xls_from_mongo(host, database, collections, xls_fname):
    """将MongoDB中数据读入本地.xls文件

    Args :
        host(path) :- MongoDB的地址
        database(str) :- MongoDB中client名字
        collections(dict) :- (collection, 该collection的keys列表)的(key,value)对
        xls_fname(path) :- 要存入.xls文件地址
    """
    client = MongoClient(host)
    db = client[database]
    book = xlwt.Workbook()
    for collection, keys in collections.items():
        print(collection, " start ")
        ws = book.add_sheet(collection)
        num_col = len(keys)
        row = 0
        
        for item in db[collection].find():
            # .xls有行数限制,一旦超过65536行就换worksheet
            if (row+1) % 65536 == 0:
                ws = book.add_sheet(collection + "_" + str((row+1)/65536))
            for col in range(num_col):
                if isinstance(item[keys[col]], list):
                    cell = " ".join(item[keys[col]])
                else :
                    cell = item[keys[col]]
                ws.write(row % 65536, col, cell)
            row += 1
        print("It contains %d row" % row)
    book.save(xls_fname)
