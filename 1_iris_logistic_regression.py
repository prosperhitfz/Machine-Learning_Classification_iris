import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import decomposition
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class ManualLogisticRegression:  # 手写Logistic回归算法
    def __init__(self, lr, iteration_num):
        self.lr = lr  # 学习率（步长）
        self.iteration_num = iteration_num  # 迭代次数

    def sigmoid(self, z):  # 定义sigmoid函数
        return 1 / (1 + np.exp(-z))

    def train(self, feature, label):  # 定义拟合函数（训练过程）
        intercept = np.ones((feature.shape[0], 1))
        feature = np.concatenate((intercept, feature), axis=1)
        # print(feature)
        # 学习权重（参数）初始化(theta也是omega,二者等价)
        self.theta = np.zeros(feature.shape[1])
        # 开始迭代计算训练，用直线对数据进行拟合
        for i in range(self.iteration_num):
            z = np.dot(feature, self.theta)  # 求自变量（数据特征feature）
            h = self.sigmoid(z)  # 计算sigmoid函数的值
            # 计算一阶导（梯度更新)
            gradient = np.dot(feature.T, (h - label)) / label.size
            self.theta -= self.lr * gradient  # 梯度下降法学习权重（参数）更新
            # 损失函数更新公式
            loss = (-label * np.log(h) - (1 - label) * np.log(1 - h)).mean()
            if i % 10000 == 0:  # 每一万次迭代输出一次loss进行观察
                print('loss: ', loss*100, '%')
        return self.theta

    def predict(self, feature):  # 定义预测函数
        intercept = np.ones((feature.shape[0], 1))
        feature = np.concatenate((intercept, feature), axis=1)
        return self.sigmoid(np.dot(feature, self.theta)).round()


iris = datasets.load_iris()  # 使用iris数据集
pca = decomposition.PCA(n_components=2)
iris_pca = pca.fit_transform(iris.data)  # PCA将4维特征维度降维至2维
iris_label = (iris.target != 0) * 1  # 数据标签
# 这里由于已知setosa和其他两类（vericolor和virginca）是线性可分的，但后面两类是线性不可分的
# 而且logistic回归使用线性可分问题，故最终决定将数据处理为setosa为一类（标签0），其他两类为另一类（标签1），做二分类问题
# print(iris_label)
X_train, X_test, y_train, y_test = train_test_split(iris_pca, iris_label, test_size=0.2, random_state=1)


# 手写的logistic regression模型训练拟合
model = ManualLogisticRegression(lr=0.01, iteration_num=500000)
omega = model.train(X_train, y_train)  # 看损失函数数值是否在下降，即正确率是否在提高
print('手写logistic回归训练集训练得到的分界线公式：', omega[0], '+', omega[1], '* x', '+', omega[2], '* y = 0')
print('\n')
res = model.predict(X_test)
# print(res)
# 分类可视化
plt.subplot(1, 2, 1)
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='g', label='training setosa')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='b', label='training versicolor&virginica')
x1 = arange(-2, 0.75, 0.1)
y1 = (-omega[0] - omega[1] * x1) / omega[2]  # 分界线公式w0+w1x1+w2x2=0,其中x1为x，x2为y
plt.plot(x1, y1, label='手写logistic分界线')
plt.legend()
plt.title('logistic回归分类训练集分类结果展示')

plt.subplot(1, 2, 2)
plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], color='g')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='b')
x2 = arange(-2, 0.75, 0.1)
y2 = (-omega[0] - omega[1] * x2) / omega[2]  # 分界线公式w0+w1x1+w2x2=0,其中x1为x，x2为y
plt.plot(x2, y2, label='手写logistic分界线')
plt.legend()
plt.title('logistic回归分类测试集分类结果展示')


# sklearn封装好的LogisticRegression进行拟合预测
model1 = LogisticRegression(C=100)
model1.fit(iris_pca, iris_label)
# score = model1.score(X_test, y_test)
# print('测试集准确率：', score, '\n')
print('sklearn封装好的logistic回归模型参数权重：', model1.coef_, '\n')
print('sklearn封装好的logistic回归模型截距：', model1.intercept_, '\n')
print('sklearn封装好的logistic回归分界线公式为：',
      float(model1.intercept_), '+', model1.coef_[0][0], '* x', '+', model1.coef_[0][1], '* y = 0')
# 分类可视化
plt.subplot(1, 2, 1)
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='g')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='b')
x3 = arange(-2, 0.75, 0.1)
y3 = (-float(model1.intercept_) - model1.coef_[0][0]*x3) / model1.coef_[0][1]  # 分界线公式w0+w1x1+w2x2=0,其中x1为x，x2为y
plt.plot(x3, y3, label='sklearn封装好的logistic分界线')
plt.legend()
plt.title('logistic回归训练集分类结果展示')

plt.subplot(1, 2, 2)
plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], color='g', label='testing setosa')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='b', label='testing versicolor&virginica')
x4 = arange(-2, 0.75, 0.1)
y4 = (-float(model1.intercept_) - model1.coef_[0][0]*x4) / model1.coef_[0][1]  # 分界线公式w0+w1x1+w2x2=0,其中x1为x，x2为y
plt.plot(x4, y4, label='sklearn封装好的logistic分界线')
plt.title('logistic回归测试集分类结果展示')
plt.legend()
plt.show()


'''
x_axis, y_axis = np.meshgrid(np.linspace(iris_pca[:, 0].min(), iris_pca[:, 0].max()),
                             np.linspace(iris_pca[:, 1].min(), iris_pca[:, 1].max()))  # 画网格图
grid = np.c_[x_axis.ravel(), y_axis.ravel()]
# 画分界线
prediction = model.predict(grid).reshape(x_axis.shape)
plt.contour(x_axis, y_axis, prediction, [0.5])
'''