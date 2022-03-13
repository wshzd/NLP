#!/usr/bin/python
# -*- coding:utf-8 -*-

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,TfidfTransformer


# select the best k for kmeans
def SelectK(maxK,totalList):
    from scipy.spatial.distance import cdist
    K = range(1, maxK)
    meandistortions = []
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(totalList)
        meandistortions.append(sum(np.min(cdist(totalList, kmeans.cluster_centers_, 'euclidean'), axis=1)) / np.array(totalList).shape[0])
    plt.plot(K, meandistortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('平均畸变程度')
    plt.xticks(K)
    plt.title('用肘部法则来确定最佳的K值')
    plt.show()
# input raw sentence to return tfidf-matrix
def TfidfFeatrue(sentences):
    # 定义向量化参数
    # tfidf_vectorizer = TfidfVectorizer(max_df=0.9, max_features=200000,
    #                                  min_df=0.01, lowercase=False,
    #                                  use_idf=True, tokenizer=None, ngram_range=(1,3))
    # tfidf_matrix = tfidf_vectorizer.fit_transform(sentences) # 向量化剧情简介文本

    count_vec = CountVectorizer(max_df=0.7, min_df=0.01)
    word_dict = {}
    counts_train = count_vec.fit_transform(sentences)
    for index, word in enumerate(count_vec.get_feature_names()):
        word_dict[index] = word
    tfidftransformer = TfidfTransformer()
    tfidf_train = tfidftransformer.fit(counts_train).transform(counts_train)
    return tfidf_train, word_dict
    
# run kmeans and save cluster result
def RunAndSaveKmeansTopic(tfidf_matrix, num_clusters, cluster_keywords, cluster_docs, word_dict):
    print('开始kmeans')
    start = time.time()
    km = KMeans(n_clusters=num_clusters, random_state=1000)
    km.fit(tfidf_matrix)
    clusters = km.labels_.tolist()
    cluster_dict = {}
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    print('结束kmeans')
    end = time.time()
    print('聚类的总共耗时为', (end-start)/3600)
    print('开始存储每个聚类的top20个topic words')

    f_docs = open(cluster_docs, 'w+')
    doc = 1
    for cluster in clusters:
        f_docs.write(str(str(doc)) + ',' + str(cluster) + '\n')
        doc += 1
        if cluster not in cluster_dict:
            cluster_dict[cluster] = 1
        else:
            cluster_dict[cluster] += 1
    f_docs.close()

    cluster = 1
    f_clusterwords = open(cluster_keywords, 'w+')
    for ind in order_centroids:  # 每个聚类选 50 个词
        words = []
        for index in ind[:20]:
            words.append(word_dict[index])
        print(cluster, ','.join(words))
        f_clusterwords.write(str(cluster) + '\t' + ','.join(words) + '\n')
        cluster += 1
        print('=========' * 5)
    f_clusterwords.close()    
    
    
    
    
    
    
