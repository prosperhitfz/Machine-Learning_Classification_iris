import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import math

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def Manual_Gaussian_Discriminant_Analysis(X, Y):  # 手动复现高斯判别分析GDA关键参数
    theta1 = len(Y[Y == 0]) / len(Y)
    mu0 = X[Y == 0].mean(axis=0)
    mu1 = X[Y == 1].mean(axis=0)
    X0 = X[Y == 0]
    X1 = X[Y == 1]
    n0_s0 = np.dot(X0.T, X0) - len(Y[Y == 0]) * np.dot(mu0.reshape(X.shape[1], 1), mu0.reshape(X.shape[1], 1).T)
    n1_s1 = np.dot(X1.T, X1) - len(Y[Y == 1]) * np.dot(mu1.reshape(X.shape[1], 1), mu1.reshape(X.shape[1], 1).T)
    sigma = (n0_s0 + n1_s1) / len(X)
    return theta1, mu0, mu1, sigma

'''
def predict(test_data, theta1, mu1, mu2, sigma):
    pre_Y0 = np.zeros(test_data.shape[1])
    for i in range(len(test_data)):  # print(test_data[i], '\n')
        print((test_data[i] - mu1).T* np.linalg.inv(sigma) * (test_data[i] - mu1))
        pre_Y0[i] = math.exp((-0.5) * (test_data[i] - mu1).T * np.linalg.inv(sigma) * (test_data[i] - mu1))
        print(pre_Y0)
    # pre_Y0_res =
'''

iris = datasets.load_iris()  # 使用iris数据集
pca = decomposition.PCA(n_components=2)
iris_pca = pca.fit_transform(iris.data)  # PCA将4维特征维度降维至2维
iris_label = (iris.target != 0) * 1  # 数据标签
# 这里已知setosa和其他两类（vericolor和virginca）是线性可分的，但后面两类是线性不可分的
# 决定将数据处理为setosa为一类（标签0），其他两类为另一类（标签1），做二分类问题
X_train, X_test, y_train, y_test = train_test_split(iris_pca, iris_label, test_size=0.15, random_state=1)

print('手动GDA训练集拟合参数：')
theta1, mu0, mu1, sigma = Manual_Gaussian_Discriminant_Analysis(X_train, y_train)
print('theta1:', theta1)
print('mu0:', mu0)
print('mu1:', mu1)
print('sigma:', sigma)
print('\n')
# predict(X_test, theta1, mu0, mu1, sigma)
print('手动GDA测试集拟合参数：')
theta1, mu0, mu1, sigma = Manual_Gaussian_Discriminant_Analysis(X_test, y_test)
print('theta1:', theta1)
print('mu0:', mu0)
print('mu1:', mu1)
print('sigma:', sigma)
print('\n')
print('手动GDA全部数据集拟合的标准参考参数：')
theta1, mu0, mu1, sigma = Manual_Gaussian_Discriminant_Analysis(iris_pca, iris_label)
print('theta1:', theta1)
print('mu0:', mu0)
print('mu1:', mu1)
print('sigma:', sigma)
print('\n')


# 应用sklearn中已经封装好的高斯判别函数进行数据分类
data = iris_pca  # iris数据特征做X
data_label = iris.target  # iris数据标签
kernel1 = 1.0 * RBF([1.0])
GDA_classification_isotropic = \
    GaussianProcessClassifier(kernel=kernel1).fit(data, data_label)
kernel2 = 1.0 * RBF([1.0, 1.0])
GDA_classification_anisotropic = \
    GaussianProcessClassifier(kernel=kernel2).fit(data, data_label)
# 可视化展示
# 画出网格图
xx, yy = np.meshgrid(np.arange(data[:, 0].min() - 1, data[:, 0].max() + 1, 0.1), np.arange(data[:, 1].min() - 1, data[:, 1].max() + 1, 0.1))
for i, model in enumerate((GDA_classification_isotropic, GDA_classification_anisotropic)):
    plt.subplot(1, 2, i + 1)
    # 拟合后预测结果
    prediction_res = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    # 预测结果区域显示(网格填色)
    prediction_res = prediction_res.reshape((xx.shape[0], xx.shape[1], 3))
    plt.imshow(prediction_res, extent=(data[:, 0].min() - 1, data[:, 0].max() + 1, data[:, 1].min() - 1, data[:, 1].max() + 1))
    # 给出源数据标签展示（同一颜色为同类别数据）
    plt.scatter(data[:, 0], data[:, 1], c=np.array(["c", "m", "y"])[data_label])
    if i == 0:
        plt.title('各向同性核函数GDA分类结果')
    elif i == 1:
        plt.title('各向异性核函数GDA分类结果')
plt.show()
