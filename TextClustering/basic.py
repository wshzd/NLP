# 下面介绍两种文本向量化
# 方法一：
def tfidf(corpus):
    # 词频矩阵：矩阵元素a[i][j] 表示j词在i类文本下的词频 
    vectorizer = CountVectorizer()
    # 统计每个词语的tf-idf权值
    transformer = TfidfTransformer() 
    freq_word_matrix = vectorizer.fit_transform(corpus)
    #获取词袋模型中的所有词语
    word = vectorizer.get_feature_names()
    tfidf = transformer.fit_transform(freq_word_matrix)
    # 元素w[i][j]表示j词在i类文本中的tf-idf权重
    weight = tfidf.toarray()
    
# 方法二：
def doc2vec():
    #训练并保存模型
    import gensim

    sentences = gensim.models.doc2vec.TaggedLineDocument(token_path)
    model = gensim.models.Doc2Vec(sentences, size=100, window=2, min_count=3)
    model.train(sentences,total_examples=model.corpus_count, epochs=1000)
    model.save('../model/demoDoc2Vec.pkl')
    
    
    
    
    
    
    
    
