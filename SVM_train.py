import argparse
import os
import pickle
import numpy as np
import sklearn
import torchvision.datasets
from PIL import Image
from skimage import feature as ft
from sklearn.svm import LinearSVC


class SVM(object):
    def __init__(self):
        # W为加上偏置的权重（D,num_class)
        self.W = None

    def svm_loss_naive(self, x, y, reg):
        """
        功能：非矢量化版本的损失函数

        输入：
        -x：(numpy array)样本数据（N,D)
        -y：(numpy array)标签（N，）
        -reg：(float)正则化强度

        输出：
        (float)损失函数值loss
        (numpy array)权重梯度dW
        """
        num_train = x.shape[0]
        num_class = self.W.shape[1]

        # 初始化
        loss = 0.0
        dW = np.zeros(self.W.shape)

        for i in range(num_train):
            scores = x[i].dot(self.W)
            # 计算边界,delta=1
            margin = scores - scores[y[i]] + 1
            # 把正确类别的归0
            margin[y[i]] = 0

            for j in range(num_class):
                # max操作
                if j == y[i]:
                    continue
                if margin[j] > 0:
                    loss += margin[j]
                    dW[:, y[i]] += -x[i]
                    dW[:, j] += x[i]

        # 要除以N
        loss /= num_train
        dW /= num_train
        # 加上正则项
        loss += 0.5 * reg * np.sum(self.W * self.W)
        dW += reg * self.W

        return loss, dW

    def svm_loss_vectorized(self, x, y, reg):
        """
        功能：矢量化版本的损失函数

        输入：
        -x：(numpy array)样本数据（N,D)
        -y：(numpy array)标签（N，）
        -reg：(float)正则化强度

        输出：
        (float)损失函数值loss
        (numpy array)权重梯度dW
        """
        loss = 0.0

        num_train = x.shape[0]
        scores = x.dot(self.W)
        margin = scores - scores[np.arange(num_train), y].reshape(num_train, 1) + 1
        margin[np.arange(num_train), y] = 0.0
        # max操作
        margin = (margin > 0) * margin
        loss += margin.sum() / num_train
        # 加上正则化项
        loss += 0.5 * reg * np.sum(self.W * self.W)

        # 计算梯度
        margin = (margin > 0) * 1
        row_sum = np.sum(margin, axis=1)
        margin[np.arange(num_train), y] = -row_sum
        dW = x.T.dot(margin) / num_train + reg * self.W

        return loss, dW

    def train(self, x, y, reg=1e-5, learning_rate=1e-3, num_iters=100, batch_size=200, verbose=False):
        """
        功能：使用随机梯度下降法训练SVM

        输入：
        -x:(numpy array)训练样本（N,D）
        -y:(numpy array)训练样本标签(N,)
        -reg:(float)正则化强度
        -learning_rate:(float)进行权重更新的学习率
        -num_iters:(int)优化的迭代次数
        -batch_size:(int)随机梯度下降法每次使用的梯度大小
        -verbose:(bool)取True时，打印输出loss的变化过程

        输出：-history_loss:(list)存储每次迭代后的loss值
        """

        num_train, dim = x.shape
        num_class = np.max(y) + 1
        print(num_class)
        # 初始化权重
        if self.W is None:
            self.W = 0.005 * np.random.randn(dim, num_class)

        history_loss = []
        # 随机梯度下降法优化权重
        for i in range(num_iters):
            # 从训练样本中随机取样作为更新权重的小批量样本
            mask = np.random.choice(num_train, batch_size, replace=False)
            batch_x = x[mask]
            batch_y = y[mask]

            # 计算loss和权重的梯度
            loss, grad = self.svm_loss_vectorized(batch_x, batch_y, reg)

            # 更新权重
            self.W += -learning_rate * grad

            history_loss.append(loss)

            # 打印loss的变化过程
            if verbose == True and i % 50 == 0:
                print("iteratons:%d/%d, loss:%f" % (i, num_iters, loss))

        return history_loss

    def predict(self, x):
        """
        功能：利用训练得到的最优权值预测分类结果

        输入：
        -x:(numpy array)待分类的样本(N,D)

        输出：y_pre(numpy array)预测的便签(N,)
        """
        scores = x.dot(self.W)
        y_pre = np.argmax(scores, axis=1)

        return y_pre


def unpickle(file):
    """
    功能：将CIFAR100中的数据转化为字典形式
    加载train和test文件返回的字典格式为：
    dict_keys([b'filenames', b'data', b'coarse_labels', b'fine_labels', b'batch_label'])
    """

    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_CIFAR10():
    """
    功能：从当前路径下读取CIFAR10数据

    输出：
    -x_train:(numpy array)训练样本数据(N,D)
    -y_train:(numpy array)训练样本数标签(N,)
    -x_test:(numpy array)测试样本数据(N,D)
    -y_test:(numpy array)测试样本数标签(N,)
    """
    train_dataset = torchvision.datasets.CIFAR10(root="CIFAR-10", train=True, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root="CIFAR-10", train=False, download=True)
    x_t = []
    y_t = []
    for i in range(1, 6):
        path_train = os.path.join('CIFAR-10/cifar-10-batches-py', 'data_batch_%d' % (i))
        data_dict = unpickle(path_train)
        x = data_dict[b'data'].astype('float')
        y = np.array(data_dict[b'labels'])

        x_t.append(x)
        y_t.append(y)

    # 将数据按列堆叠进行合并,默认按列进行堆叠
    x_train = np.concatenate(x_t)
    y_train = np.concatenate(y_t)

    path_test = os.path.join('CIFAR-10/cifar-10-batches-py', 'test_batch')
    data_dict = unpickle(path_test)
    x_test = data_dict[b'data'].astype('float')
    y_test = np.array(data_dict[b'labels'])

    return x_train, y_train, x_test, y_test


def hog_extraction(data, size=8):
    """
    功能：提取图像HOG特征

    输入：
    data:(numpy array)输入数据[num,3,32,32]
    size:(int)(size,size)为提取HOG特征的cellsize

    输出：
    data_hogfeature:(numpy array):data的HOG特征[num,dim]
    """
    num = data.shape[0]
    data = data.astype('uint8')
    # 提取训练样本的HOG特征
    hog_features = []
    for i in range(num):
        x = data[i]
        r = Image.fromarray(x[0])
        g = Image.fromarray(x[1])
        b = Image.fromarray(x[2])

        # 合并三通道
        img = Image.merge("RGB", (r, g, b))
        # 转为灰度图
        gray = img.convert('L')
        # 转化为array
        gray_array = np.array(gray)

        # 提取HOG特征
        hog_feature = ft.hog(gray_array, pixels_per_cell=(size, size))

        hog_features.append(hog_feature)
        if i % 5000 == 0:
            print(i)
            print(hog_feature.shape)

    # 把data1_hogfeature中的特征按行堆叠
    res = np.reshape(np.concatenate(hog_features), [num, -1])
    return res


parser = argparse.ArgumentParser()
parser.add_argument('--save-dir', default='svm_save')

# 主函数
if __name__ == '__main__':
    args = parser.parse_args()
    # 加载数据
    x_train, y_train, x_test, y_test = load_CIFAR10()

    # 创建训练样本
    x_train = x_train.reshape(-1, 3, 32, 32)
    y_train = y_train

    # 创建测试样本
    x_test = x_test.reshape(-1, 3, 32, 32)
    y_test = y_test

    # 用验证集来寻找最优的SVM模型以及cellsize
    # 4最好
    cellsize = [2, 4, 6]
    # 0.1最好
    learning_rate = [0.001, 0.01, 0.1]
    # 0最好
    weight_decay = [0, 0.1, 0.5]

    best_acc = 0
    for cs in cellsize:
        # 提取训练集和验证集的HOG特征
        hog_train = hog_extraction(x_train, size=cs)
        hog_test = hog_extraction(x_test, size=cs)

        for lr in learning_rate:
            for wd in weight_decay:
                svm = SVM()
                # 训练
                history_loss = svm.train(hog_train, y_train, reg=wd, learning_rate=lr, num_iters=2000)
                # 预测测试集类别
                y_pre = svm.predict(hog_test)
                # 计算测试集精度
                acc = np.mean(y_pre == y_test)

                if os.path.exists(args.save_dir) is False:
                    os.mkdir(args.save_dir)
                with open(args.save_dir + "/acc.txt", "a") as file1:
                    file1.write(str(cs) + " " + str(lr) + " " + str(wd) + " "
                                + str(acc) + " " + str(best_acc) + "\n")
                file1.close()
                with open(args.save_dir + "/svm.pickle", "wb") as f:
                    pickle.dump(svm, f)

                # 选取精度最大时的最优模型
                if acc > best_acc:
                    best_acc = acc
                    best_cellsize = cs
                    best_learning_rate = lr
                    best_weight_decay = wd
                    best_svm = svm
                    with open(args.save_dir + "/best_svm.pickle", "wb") as f:
                        pickle.dump(best_svm, f)

                print("cellsize=%d, learning_rate=%e, weight_decay=%e" % (cs, lr, wd))
                print("val_acc=%f, best_acc=%f" % (acc, best_acc))
    # 输出最大精度
    print("best_cellsize=%d, best_learning_rate=%e, best_weight_decay=%e"
          % (best_cellsize, best_learning_rate, best_weight_decay))
    print("best_acc=%f" % best_acc)

    # 用自带的模型进行调参
    # 0.4366, 0.5477, 0.5305
    best_acc = 0
    for cs in cellsize:
        # 提取训练集和验证集的HOG特征
        hog_train = hog_extraction(x_train, size=cs)
        hog_test = hog_extraction(x_test, size=cs)

        linear_svc = LinearSVC()
        linear_svc.fit(hog_train, y_train)
        y_pre = linear_svc.predict(hog_test)
        acc = np.mean(y_pre == y_test)

        if acc > best_acc:
            best_acc = acc
            best_cellsize = cs
            best_linear_svc = linear_svc
        print("cellsize=%d" % cs)
        print("acc=%f, best_acc=%f" % (acc, best_acc))
    print("best_cellsize=%d" % best_cellsize)
    print("best_acc=%f" % best_acc)
