import torch
import torch.nn as nn
from model import ESIM
from arg import lr,EPOCH,classes
from load_dataset import TEXT,test_iter,train_iter

#5 有gpu用gpu，否则cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net=ESIM(TEXT).to(device)

#6、定义优化方式和损失函数
optimizer=torch.optim.Adam(net.parameters(),lr=lr)
loss_func=torch.nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, batch in enumerate(train_iter):
        # 1、把索引转化为tensor变量，载入设备
        a = batch.sentence1.t()
        b = batch.sentence2.t()
        l = batch.gold_label

        # 2、计算模型输出
        out = net(a, b)

        # 3、预测结果传给loss
        loss = loss_func(out, l)

        # 4、固定格式
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            total = 0
            correct = 0
            for batch in test_iter:
                tst_a = batch.sentence1.t()
                tst_b = batch.sentence2.t()
                tst_l = batch.gold_label
                out = net(tst_a, tst_b)
                out = torch.argmax(out, dim=1).long()
                if out.size() == tst_l.size():
                    total += tst_l.size(0)
                    correct += (out == tst_l).sum().item()

            print('[Epoch ~ Step]:', epoch + 1, '~', step + 1, '训练loss:', loss.item())
            print('[Epoch]:', epoch + 1, '测试集准确率: ', (correct * 1.0 / total))
torch.save(net.state_dict(),"ESIM_model.pth") #这里的net即模型必须是实例化后的模型
