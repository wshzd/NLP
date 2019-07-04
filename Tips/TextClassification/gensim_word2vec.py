from gensim.models.word2vec import Word2Vec

POS_SW_PATH = "../data/pos_cw.txt"
NEG_SW_PATH = "../data/neg_cw.txt"
N_DIM = 100
                         
# word2vec的数量
MIN_COUNT = 3
                       
# 保证出现的词数足够做才进入词典
w2v_EPOCH = 20
                      
# w2v的训练迭代次数
# 读取文件
pos_data = []
with open(POS_SW_PATH, encoding="utf8") as f:    
    for line in f:
        ll = line.strip().split("\t")
        pos_data.append(ll)
neg_data = []

with open(NEG_SW_PATH, encoding="utf8") as f:    
    for line in f:
        ll = line.strip().split("\t")
        neg_data.append(ll)
all_data = pos_data + neg_data
# word2vector词向量预备
imdb_w2v = Word2Vec(size=N_DIM, min_count=MIN_COUNT)
imdb_w2v.build_vocab(all_data)

# 把所有未进入词表的都转为unk_
for sentence_idx in range(len(all_data)):    
    for word_item_idx in range(len(all_data[sentence_idx])):
        if all_data[sentence_idx][word_item_idx] not in imdb_w2v.wv.vocab:
            all_data[sentence_idx][word_item_idx] = "unk_"

# 重新构建词汇表
imdb_w2v = Word2Vec(size=N_DIM, min_count=MIN_COUNT)
imdb_w2v.build_vocab(all_data)

# 训练
imdb_w2v.train(all_data, total_examples=len(all_data), epochs=w2v_EPOCH)

# 模型保存
# imdb_w2v.save("../data/w2v_model/word2vec_20190626.model")
SAVE_PATH = "../data/w2v_model/word2vec_20190626.model"
SAVE_PATH_WORD2ID_DICT = "../data/w2v_model/word2id_20190626.model"
fout_model = open(SAVE_PATH, "w", encoding="utf8")
fout_word2id_dict = open(SAVE_PATH_WORD2ID_DICT, "w", encoding="utf8")
idx = 0
for k,v in imdb_w2v.wv.vocab.items():
    str_get = "%s\t%s\n" % (k, "\t".join([str(i) for i in imdb_w2v.wv[k]]))
    fout_model.write(str_get)
    str_get = "%s\t%s\n" % (k, idx)
    fout_word2id_dict.write(str_get)
    idx = idx + 1
fout_model.close()
fout_word2id_dict.close()
