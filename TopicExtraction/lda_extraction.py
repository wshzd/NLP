#!/usr/bin/python
# -*- coding:utf-8 -*-

import codecs
from gensim import corpora,models
import pandas as pd


# Only run LDA on the source text directly
# param:label-just for load file
def RunAndSaveLDA(label,num_topics):
    # 加载stopwords
    stopword = [word.strip() for word in codecs.open('stop_words.txt',encoding='utf8')]

    # 加载文档
    texts = [[word.strip() for word in sentence.split(' ') if word not in stopword] for sentence in codecs.open('data/segment{}.txt'.format(label),encoding='utf8')]
    # texts = [[word.strip() for word in sentence.split(' ') if word not in stopword] for sentence in k_df['sentences']]
    # 用文本构建 Gensim 字典
    dictionary = corpora.Dictionary(texts)

    # 去除极端的词（和构建 tf-idf 矩阵时用到 min/max df 参数时很像）
    dictionary.filter_extremes(no_below=1, no_above=0.8)

    # 将字典转化为词典模型（bag of words）作为参考VSM
    corpus = [dictionary.doc2bow(text) for text in texts]

    lda = models.LdaModel(corpus, num_topics=num_topics,
                          id2word=dictionary,
                          update_every=5,
                          chunksize=10000,
                          passes=100)

    ################################### 输出并且保存文档和文档对应的主题###############################
    # 输出K个topic
    topics_matrix = lda.show_topics(num_topics=num_topics, num_words=20, formatted=False)
    print('目前正在处理的大类是：     ',label)
    print('topics_matrix===========', topics_matrix)

    # 输出每个sentence对应的topic
    lda_df = pd.DataFrame(columns=['sentences','lda_topic','lda_label'])
    index = 0
    for i,text in zip(lda.get_document_topics(corpus)[:],texts):
        listj = []
        for j in i:
            listj.append(j[1])
        bz = listj.index(max(listj))
        # print(i[bz][0],i,listj,listj.index(max(listj)))
        document = ''.join([word for word in text])
        topic = ' '.join([topic_word[0] for topic_word in topics_matrix[i[bz][0]][1]])
        lda_df.loc[index,'sentences'] = document
        lda_df.loc[index, 'lda_topic'] = topic
        lda_df.loc[index, 'lda_label'] = i[bz][0]
        index += 1
    lda_df.to_csv('data/onlyLDA{}.csv'.format(label), encoding='utf8', index=False)

if __name__ == '__main__':
    label = 0
    num_topics = 20
    RunAndSaveLDA(label,num_topics)




