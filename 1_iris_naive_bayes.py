import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import decomposition
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class NaiveBayes:
    def __init__(self, data, label):
        self.irisdata = data  # iris数据特征
        self.irislabel = label  # iris数据标签
        self.label_info = {}  # 数据集中标签的具体分类和对应标签的个数
        self.data_number_prob = {}  # 存储每个类别下每个特征每种情况出现的概率

    def train(self):
        # 初始化一个字典，用于存放同一标签的对应数据（标签0，1，2的数据分类好收入字典中）
        classified_data_dic = {}
        for i in range(150):
            # 先将标签放进字典的key内
            if self.irislabel[i] not in classified_data_dic:
                # 初始化对应key后面的value数组
                classified_data_dic[self.irislabel[i]] = []
            # 将数据接在对应标签key的后面value数组内
            classified_data_dic[self.irislabel[i]].append(self.irisdata[i])

        for label, data in classified_data_dic.items():
            if label not in self.data_number_prob:
                self.data_number_prob[label] = {}
            data = np.array(data)  # 将data转化为numpy数组型，便于下续运算
            # 初始化空字典，
            # 存放每个标签(3)下所有数据(50+50+50)的
            # 每一维(4)每个相同的特征数占全部特征数的占比
            probability = {}
            for i in range(len(data[0])):
                if i not in probability:
                    probability[i] = {}  # 将标签放入probability的key中
                data_list = list(set(data[:, i]))  # 初始化每个标签内的存放数组
                for j in data_list:
                    if j not in probability[i]:  # j是一组数据特征的某一维度的特征数值
                        probability[i][j] = 0
                    # 存入这一个特征数值占这一标签下全部特征数值的百分比
                    probability[i][j] = np.sum(data[:, i] == j) / float(len(data[:, i]))
            probability[0] = [1 / float(len(data[:, 0]))]  # 针对特征数值不存在的情况
            self.data_number_prob[label] = probability

    def classify(self, data):
        # 初始化数据的标签可能性数组（即该数据属于每个种类0，1，2的概率）
        possibility = np.ones(3)
        # 利用朴素贝叶斯公式计算对应标签的可能性
        for i in self.label_info:
            for j in self.data_number_prob[i]:
                if data[j] not in self.data_number_prob[i][j]:
                    possibility[i] *= self.data_number_prob[i][0][0]
                else:
                    possibility[i] *= self.data_number_prob[i][j][data[j]]
        # 给出分类结果，即possibility中可能性最大的那个对应标签
        prediction_label = np.where(possibility == np.max(possibility))[0][0]
        # 注：np.where返回的是一个二维数组，
        # 第一个位置是最大值的那个对应序号，第二个是数据类型
        return prediction_label

    def test(self):
        labelList = [i for i in iris.target]  # 获取数据标签的种类(0, 1和2)
        labelNum = len(labelList)  # 获取数据种类的个数（3种）
        for i in range(labelNum):  # 获取每种标签的数据个数占全部数据总个数的比例（均为33.3%）
            self.label_info[labelList[i]] = np.sum(self.irislabel == labelList[i]) / float(len(self.irislabel))
        self.train()
        prediction_labels = []  # 初始化预测的测试集数据标签
        correct_number = 0  # 初始化正确分类的数据个数
        for j in self.irisdata:
            prediction_labels.append(self.classify(j))  # 写入预测标签
        for k in range(len(self.irislabel)):
            if prediction_labels[k] == self.irislabel[k]:  # 预测标签与原标签比较
                correct_number += 1
        print('真实测试集数据标签：', self.irislabel)
        print('预测的测试集数据标签', prediction_labels)
        print('预测准确率：', correct_number / len(self.irislabel), '\n')  # 计算预测准确率
        return np.array(prediction_labels)


# 数据预处理
iris = datasets.load_iris()  # 使用iris数据集
pca = decomposition.PCA(n_components=2)
iris_pca = pca.fit_transform(iris.data)  # PCA将4维特征维度降维至2维
# print(iris_label)
X_train, X_test, y_train, y_test = train_test_split(iris_pca, iris.target, test_size=0.3, random_state=1)

# 使用手写朴素贝叶斯分类器模型
print('手写朴素贝叶斯分类器模型结果：')
model = NaiveBayes(iris.data, iris.target)
prediction_res = model.test()
# 数据可视化
plt.subplot(1, 2, 1)
plt.scatter(iris.data[iris.target == 0][:, 0], iris.data[iris.target == 0][:, 1], color='c', label='original setosa')
plt.scatter(iris.data[iris.target == 1][:, 0], iris.data[iris.target == 1][:, 1], color='m', label='original versicolor')
plt.scatter(iris.data[iris.target == 2][:, 0], iris.data[iris.target == 2][:, 1], color='y', label='original virginica')
plt.title('原始数据隶属类别展示')
plt.legend()
plt.subplot(1, 2, 2)
plt.scatter(iris.data[prediction_res == 0][:, 0], iris.data[prediction_res == 0][:, 1], color='c',
            label='predicted setosa')
plt.scatter(iris.data[prediction_res == 1][:, 0], iris.data[prediction_res == 1][:, 1], color='m',
            label='predicted versicolor')
plt.scatter(iris.data[prediction_res == 2][:, 0], iris.data[prediction_res == 2][:, 1], color='y',
            label='predicted virginica')
plt.title('手写朴素贝叶斯分类器经训练后预测的全体数据集数据隶属类别展示')
plt.legend()
plt.show()

# 使用sklearn内封装好的朴素贝叶斯分类器模型
model1 = GaussianNB()
# 训练集训练朴素贝叶斯分类器
model1.fit(X_train, y_train)
# 用训练得到的模型对测试集数据进行预测，得到的预测标签储存在y_test_prediction中
y_test_prediction = model1.predict(X_test)
print('sklearn内封装好的朴素贝叶斯分类器模型结果：')
print('真实测试集数据标签：', y_test)
print('预测的测试集数据标签', y_test_prediction)
print('预测准确率：', accuracy_score(y_test, y_test_prediction))
# 分类可视化
# 画出网格图
xx, yy = np.meshgrid(np.arange(iris_pca[:, 0].min() - 1, iris_pca[:, 0].max() + 1, 0.1),
                     np.arange(iris_pca[:, 1].min() - 1, iris_pca[:, 1].max() + 1, 0.1))
# 拟合后预测结果
prediction_res = model1.predict_proba(np.c_[xx.ravel(), yy.ravel()])
plt.subplot(1, 2, 1)
# 预测结果区域显示(网格填色)
prediction_res = prediction_res.reshape((xx.shape[0], xx.shape[1], 3))
plt.imshow(prediction_res, extent=(
    iris_pca[:, 0].min() - 1, iris_pca[:, 0].max() + 1, iris_pca[:, 1].min() - 1, iris_pca[:, 1].max() + 1))
# 给出源数据原始标签展示（同一颜色为实际的同类别数据）
plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], color='c', label='original setosa')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='m', label='original versicolor')
plt.scatter(X_test[y_test == 2][:, 0], X_test[y_test == 2][:, 1], color='y', label='original virginica')
plt.title('原测试数据隶属类别展示')
plt.legend()
plt.subplot(1, 2, 2)
# 预测结果区域显示(网格填色)
prediction_res = prediction_res.reshape((xx.shape[0], xx.shape[1], 3))
plt.imshow(prediction_res, extent=(
    iris_pca[:, 0].min() - 1, iris_pca[:, 0].max() + 1, iris_pca[:, 1].min() - 1, iris_pca[:, 1].max() + 1))
# 给出源数据预测标签展示（同一颜色为预测的同类别数据）
plt.scatter(X_test[y_test_prediction == 0][:, 0], X_test[y_test_prediction == 0][:, 1], color='c',
            label='predicted setosa')
plt.scatter(X_test[y_test_prediction == 1][:, 0], X_test[y_test_prediction == 1][:, 1], color='m',
            label='predicted versicolor')
plt.scatter(X_test[y_test_prediction == 2][:, 0], X_test[y_test_prediction == 2][:, 1], color='y',
            label='predicted virginica')
plt.title('sklearn内封装好的朴素贝叶斯分类器经训练后预测的测试集数据隶属类别展示')
plt.legend()
plt.show()
