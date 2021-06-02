import torch
import torch.nn as nn
from arg import lr,hidden_size,batch_size,linear_size,batch_size,classes,EPOCH
from load_dataset import TEXT


class ESIM(nn.Module):
    def __init__(self, TEXT):
        super(ESIM, self).__init__()
        # 定义embedding层
        self.embedding = nn.Embedding(*TEXT.vocab.vectors.size())
        '''
        nn.Embedding(*TEXT.vocab.vectors.size())
      ==nn.Embedding(*(36990,100))
      ==nn.Embedding(36990,100)
        '''
        self.embedding.weight.data.copy_(TEXT.vocab.vectors)
        # 定义输入编码层双向LSTM
        self.A_bilstm_input = nn.LSTM(TEXT.vocab.vectors.size()[1], hidden_size,
                                      batch_first=True, bidirectional=True)
        self.B_bilstm_input = nn.LSTM(TEXT.vocab.vectors.size()[1], hidden_size,
                                      batch_first=True, bidirectional=True)
        # 定义推理组合层双向LSTM
        self.A_bilstm_infer = nn.LSTM(8 * hidden_size, hidden_size,
                                      batch_first=True, bidirectional=True)
        self.B_bilstm_infer = nn.LSTM(8 * hidden_size, hidden_size,
                                      batch_first=True, bidirectional=True)
        # 定义预测层全连接网络
        self.linear = nn.Sequential(
            nn.Linear(8 * hidden_size, 2 * hidden_size),
            nn.ReLU(True),
            nn.Linear(2 * hidden_size, linear_size),
            nn.ReLU(True),
            nn.Linear(linear_size, classes)
        )

    def forward(self, a, b):
        # 词嵌入层
        emb_a = self.embedding(a)
        emb_b = self.embedding(b)

        # 输入编码层
        a_bar, (h, c) = self.A_bilstm_input(emb_a)
        b_bar, (h, c) = self.B_bilstm_input(emb_b)

        # 局部推理建模层
        e = torch.matmul(a_bar, b_bar.permute(0, 2, 1))
        a_tilde = torch.matmul(torch.softmax(e, dim=2), b_bar)
        b_tilde = torch.matmul(torch.softmax(e, dim=1).permute(0, 2, 1), a_bar)

        # 矩阵拼接
        ma = torch.cat([a_bar, a_tilde, a_bar - a_tilde, a_bar * a_tilde], dim=2)
        mb = torch.cat([b_bar, b_tilde, b_bar - b_tilde, b_bar * b_tilde], dim=2)

        # 推理组合层
        va, (h, c) = self.A_bilstm_infer(ma)
        vb, (h, c) = self.B_bilstm_infer(mb)

        va_avg = torch.mean(va, dim=1)
        va_max = torch.max(va, dim=1)[0]
        vb_avg = torch.mean(vb, dim=1)
        vb_max = torch.max(vb, dim=1)[0]
        v = torch.cat([va_avg, va_max, vb_avg, vb_max], dim=1)

        # 输出预测层
        out = self.linear(v)
        return out