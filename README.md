# 用多语义文本匹配模型EISM做文本匹配
### 我的这次实验在**imput encoding层**和**inference composition层**没有用语法树，用的是普通BILSTM，数据集是**SNLI**数据集，格式为（sentence1,sentence2,gold_label）,词向量经过glove预训练处理，我是直接在**https://nlp.stanford.edu/projects/glove/**上下载得到glove词向量，其中训练句子对数量为550146，测试集的句子对数量为10000
