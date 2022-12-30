# ############## 根据对txt文件 写入、读取数据，绘制曲线图##############
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    fp1 = open('cnn_save/loss.txt', 'r')
    total_loss = []
    i = 0
    for loss in fp1:
        loss = loss.strip('\n')  # 将\n去掉
        total_loss.append(loss)
        i += 1
        if i == 105:
            break
    fp1.close()
    total_loss = np.array(total_loss, dtype=float)  # 将其转换成numpy的数组，并定义数据类型为float

    fp2 = open('cnn_save/acc.txt', 'r')
    total_acc = []
    i = 0
    for acc in fp2:
        acc = acc.strip('\n')  # 将\n去掉
        acc = acc.split(' ')
        total_acc.append(acc[0])
        i += 1
        if i == 105:
            break
    fp2.close()
    total_acc = np.array(total_acc, dtype=float)  # 将其转换成numpy的数组，并定义数据类型为float

    X = np.linspace(0, i - 1, i)
    # Y1 = total_loss
    Y2 = total_acc

    plt.figure(figsize=(8, 6))  # 定义图的大小
    plt.title("Train Result")

    plt.xlabel("Train Epoch")
    # plt.ylabel("Train Loss")
    plt.ylabel("Test Acc")

    # plt.plot(X, Y1)
    plt.plot(X, Y2)
    plt.show()
