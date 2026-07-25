import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
    # 读取数据
    rnn_data = pd.read_csv('./logs/RNN/RNN_log.csv', sep=',')
    lstm_data = pd.read_csv('./logs/LSTM/LSTM_log.csv', sep=',')
    gru_data = pd.read_csv('./logs/GRU/GRU_log.csv', sep=',')

    # 绘制训练集和验证集的损失曲线
    plt.plot(rnn_data['epoch'], rnn_data['train_loss'], label='RNN')
    plt.plot(lstm_data['epoch'], lstm_data['train_loss'], label='LSTM')
    plt.plot(gru_data['epoch'], gru_data['train_loss'], label='GRU')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.show()

    # 绘制训练集和验证集的准确率曲线
    plt.plot(rnn_data['epoch'], rnn_data['train_acc'], label='RNN')
    plt.plot(lstm_data['epoch'], lstm_data['train_acc'], label='LSTM')
    plt.plot(gru_data['epoch'], gru_data['train_acc'], label='GRU')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('acc.png')
    plt.show()
