# gensim_LDA
基于gensim模块,训练LDA(Latent Dirichlet Allocation)模型,用于计算长短文本的相似度.

长文本-短文本相似度匹配的应用可参考https://github.com/baidu/Familia


# deploy
1. 把语料放到crawl_news文件夹下,每个文章一个.txt文件
2. 
```cmd
pip install -r requirements.txt [-i  https://pypi.doubanio.com/simple/  --trusted-host pypi.doubanio.com]

python main.py [-s 龙妈] [-l "风暴降生丹妮莉丝 安达尔人、罗伊纳人及先民的女王 七国统治者兼全境守护 大草海的卡丽熙 不焚者 镣铐破碎者 弥撒 弥林的女王 龙母"]
```



