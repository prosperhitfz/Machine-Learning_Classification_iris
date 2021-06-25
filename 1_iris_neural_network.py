import matplotlib.pyplot as plt
import mglearn
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn import decomposition

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


iris = datasets.load_iris()  # 使用iris数据集
pca = decomposition.PCA(n_components=2)
iris_pca = pca.fit_transform(iris.data)  # PCA将4维特征维度降维至2维
iris_label = (iris.target != 0) * 1  # 数据标签
X_train, X_test, y_train, y_test = train_test_split(iris_pca, iris_label, test_size=0.25, random_state=1)
# 神经网络之多层感知机，输入层+两层隐藏层各6个神经元+输出层一层
mlp = MLPClassifier(hidden_layer_sizes=(6, 6), max_iter=500000)
mlp.fit(X_train, y_train)  # 训练数据拟合
print(mlp.coefs_)
print(mlp.intercepts_)
train_prediction_res = mlp.predict(X_train)
test_prediction_res = mlp.predict(X_test)
print('二分类训练集分类准确率：', accuracy_score(y_train, train_prediction_res))
print('二分类测试集分类准确率：', accuracy_score(y_test, test_prediction_res))
# 可视化展示
plt.subplot(2, 2, 1)
plt.title('二分类训练集数据原类别')
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.subplot(2, 2, 2)
plt.title('二分类训练集数据预测类别')
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], train_prediction_res)
plt.subplot(2, 2, 3)
plt.title('二分类测试集数据原类别')
mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_test)
plt.subplot(2, 2, 4)
plt.title('二分类测试集数据预测类别')
mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], test_prediction_res)
plt.show()


X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=1)
# 神经网络之多层感知机，输入层+两层隐藏层各8个神经元+输出层一层
mlp1 = MLPClassifier(hidden_layer_sizes=(8, 8), max_iter=500000)
mlp1.fit(X_train, y_train)  # 训练数据拟合
print(mlp1.coefs_)
print(mlp1.intercepts_)
train_prediction_res = mlp1.predict(X_train)
test_prediction_res = mlp1.predict(X_test)
print('多分类训练集分类准确率：', accuracy_score(y_train, train_prediction_res))
print('多分类测试集分类准确率：', accuracy_score(y_test, test_prediction_res))
# 可视化展示
plt.subplot(2, 2, 1)
plt.title('多分类训练集数据原类别')
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.subplot(2, 2, 2)
plt.title('多分类训练集数据预测类别')
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], train_prediction_res)
plt.subplot(2, 2, 3)
plt.title('多分类测试集数据原类别')
mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_test)
plt.subplot(2, 2, 4)
plt.title('多分类测试集数据预测类别')
mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], test_prediction_res)
plt.show()
