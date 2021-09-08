## 自监督学习大致分为Generative Learning vs Contrastive Learning
Generative Learning（生成式方法）这类方法以自编码器为代表，主要关注pixel label的loss。举例来说，在自编码器中对数据样本编码成特征再解码重构，这里认为重构的效果比较好则说明模型学到了比较好的特征表达，而重构的效果通过pixel label的loss来衡量。

Contrastive Learning（对比式方法）这类方法则是通过将数据分别与正例样本和负例样本在特征空间进行对比，来学习样本的特征表示。Contrastive Learning主要的难点在于如何构造正负样本。