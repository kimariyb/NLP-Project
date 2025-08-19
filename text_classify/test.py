import torch
from models import MyRNN


def dm01_test_myrnn():
    # 1 实例化rnn对象
    myrnn = MyRNN(57, 128, 18)
    print('myrnn--->', myrnn)

    # 2 准备数据
    input = torch.randn(6, 57)
    print(input.shape)
    hidden = myrnn.init_hidden()

    # 3 给模型1次性的送数据
    # [seqlen, 57], [1, 1, 128]) -> [1,18], [1,1,128]
    output, hidden = myrnn(input, hidden)
    print('一次性的送数据：output->', output.shape, output)
    print('hidden->', hidden.shape)

    # 4 给模型1个字符1个字符的喂数据
    hidden = myrnn.init_hidden()
    for i in range(input.shape[0]):
        tmpinput = input[i].unsqueeze(0)
        output, hidden = myrnn(tmpinput, hidden)

    # 最后一次ouput
    print('一个字符一个字符的送数据output->', output.shape, output)


if __name__ == '__main__':
    dm01_test_myrnn()