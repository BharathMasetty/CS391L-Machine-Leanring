from mlxtend.data import loadlocal_mnist
import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import time


# Assuming that data is of 784 * 10000
# Returns Compressed Data
class PCA:
    def __init__(self, data):
        self.data = data
        self.mean = np.mean(self.data, axis=1)
        self.mean = self.mean[:, np.newaxis]
        data_Norm = self.data - self.mean
        self.COV = np.cov(data_Norm)
        self.w, self.v = LA.eig(self.COV)

    def reducedEIGSPACE(self, dim):
        return self.v[:, 0:dim]

    def Projection(self, Samples, v_reduced):
        Samples_norm = Samples - self.mean
        temp = np.empty((min(v_reduced.shape), Samples.shape[1]))
        for i in range(Samples.shape[1]):
            temp[:, i] = np.dot(np.transpose(v_reduced), Samples_norm[:, i])
        return temp

    def Inv_Projection(self, com_data, v_reduced):
        temp = np.empty((max(v_reduced.shape), com_data.shape[1]))
        for i in range(com_data.shape[1]):
            temp[:, i] = np.dot(v_reduced, com_data[:, i])
        return temp


class KNN:
    def __init__(self, data_train, label_train, test_data, test_label):
        self.data_train = np.transpose(data_train)
        self.label_train = np.transpose(label_train)
        self.test_data = np.transpose(test_data)
        self.test_label = np.transpose(test_label)

    def VariableK(self, max_k):
        k_range = range(0, max_k)
        scores = []
        scores_list = []
        Classification_time = []
        for k in k_range:
            start_time = time.time()
            Classifier = KNeighborsClassifier(n_neighbors=k + 1)
            Classifier.fit(self.data_train, self.label_train)
            y_predict = Classifier.predict(self.test_data)
            scores_list.append(metrics.accuracy_score(self.test_label, y_predict))
            print(str(k) + 'Done')
            Classification_time.append(time.time() - start_time)

        # Plotting accuracy vs K
        return scores_list, Classification_time

    def FixedK(self, k):
        Classifier = KNeighborsClassifier(n_neighbors=k)
        Classifier.fit(self.data_train, self.label_train)
        y_predict = Classifier.predict(self.test_data)
        score = metrics.accuracy_score(self.test_label, y_predict)
        return score


# Function to plot images
def PlotImg(ImgVector, name):
    img = ImgVector.reshape(28, 28)
    plt.imsave(str(name) + '.png', img, cmap='gray')


# Importing the training data
X, y = loadlocal_mnist(images_path='/media/bharath/Storage/UTSPRING2020/MACHINE_LEARNING/ASSIGNMENT_1/train-images'
                                   '-idx3-ubyte',
                       labels_path='/media/bharath/Storage/UTSPRING2020/MACHINE_LEARNING/ASSIGNMENT_1/train-labels'
                                   '-idx1-ubyte')
# Importing the testing data
X_test, y_test = loadlocal_mnist(images_path='/media/bharath/Storage/UTSPRING2020/MACHINE_LEARNING/ASSIGNMENT_1/t10k'
                                             '-images-idx3-ubyte',
                                 labels_path='/media/bharath/Storage/UTSPRING2020/MACHINE_LEARNING/ASSIGNMENT_1/t10k'
                                             '-labels-idx1-ubyte')

# Choosing first 10000 training images
X = X[0:10000, :]
y = y[0:10000]
X_test = X_test[0:5000, :]
y_test = y_test[0:5000]

# Each Column Vector is an image of 784 pixels
X = np.transpose(X)
X_test = np.transpose(X_test)

# TODO: Plotting the first 20 1) Raw Images 2) Eigen Vectors 3) Projected Images 4) Retrieved Images
# Raw Images 20
for i in range(0, 20):
    PlotImg(X_test[:, i], i + 1)

# Eigen Vector
MyPCA = PCA(X)
V_red = MyPCA.reducedEIGSPACE(50)
V_red = V_red.real
for j in range(0, 20):
    PlotImg(V_red[:, j], j + 20 + 1)

# Retrieved Images
X_test_Projected = MyPCA.Projection(X_test, V_red)
X_retrieved = MyPCA.Inv_Projection(X_test_Projected, V_red)
for k in range(0, 20):
    PlotImg(X_retrieved[:, k], k + 20 + 20 + 1)

# TODO : KNN Without PCA, 10000 training, 500 Test, K - 1 to 10 -- DONE
MyKNN = KNN(X, y, X_test, y_test)
result = MyKNN.VariableK(10)
k_range = range(1, 11)
plt.plot(k_range, result)
plt.xlabel('Number of Neighbours')
plt.ylabel('Classification Accuracy')
plt.savefig('KNN_Variable_K')

# TODO: PCA with 5000 Trianing Data and Variable Eigen Vectors (1 to 50)--  DONE
Varying number of eigen vectors
MyPCA = PCA(X)
score_list = []
k_range = range(1, 50)
for k in k_range:
    v_red = MyPCA.reducedEIGSPACE(k)
    X_train_Projected = MyPCA.Projection(X, v_red)
    X_test_Projected = MyPCA.Projection(X_test[:, 0:5000], v_red)
    MyKNN2 = KNN(X_train_Projected, y, X_test_Projected, y_test[0:5000])
    score = MyKNN2.FixedK(8)
    print(score)
    score_list.append(score)

plt.plot(k_range, score_list)
plt.xlabel('Number of Eigen Vectors')
plt.ylabel('Classification Accuracy')
plt.savefig('PCA_Variable_EV')

# TODO: PCA Classification w/ 50 EV and Variable training data
training_sets = np.linspace(start=100, stop=10000, num=99)
score_list = []
for i in training_sets:
    X_train = X[:, 0:i.astype(int)]
    y_train = y[0:i.astype(int)]
    MyPCA2 = PCA(X_train)
    V_red = MyPCA2.reducedEIGSPACE(50)
    X_train_Projected = MyPCA2.Projection(X_train, V_red)
    X_test_Projected = MyPCA2.Projection(X_test, V_red)
    MyKNN3 = KNN(X_train_Projected, y_train, X_test_Projected, y_test)
    score = MyKNN3.FixedK(8)
    print(score)
    score_list.append(score)

plt.plot(training_sets, score_list)
plt.xlabel('Size of Training Data')
plt.ylabel('Classification Accuracy')
plt.savefig('PCA_Variable_training')

# TODO: KNN w/ vs W/O PCA varying Neighbour Size
MyKNN = KNN(X, y, X_test, y_test)
result, time_WO_PCA = MyKNN.VariableK(10)
k_range = range(1, 10)
plt.plot(k_range, result)
plt.xlabel('Number of Neighbours')
plt.ylabel('Classification Accuracy')
plt.savefig('KNN_Variable_K')

MyPCA = PCA(X)
V_red = MyPCA.reducedEIGSPACE(50)
X_train_pca = MyPCA.Projection(X, V_red)
X_test_pca = MyPCA.Projection(X_test, V_red)
MyKNN2 = KNN(X_train_pca, y, X_test_pca, y_test)
scores_list, time_w_PCA = MyKNN2.VariableK(10)

# Gawd Level Plot
fig, ax1 = plt.subplots()

ax1.set_xlabel('Number of Neighbours')
ax1.set_ylabel('Accuracy')
ax1.plot(k_range, result, 'r', label='without PCA')
ax1.plot(k_range, scores_list, 'b', label='with PCA')
ax1.legend(loc="upper left")

ax2 = ax1.twinx()
ax2.set_ylabel('Computation Time')
ax1.plot(k_range, time_WO_PCA, 'y', label='Time without PCA')
ax1.plot(k_range, time_w_PCA, 'g', label='Time with PCA')
ax1.legend(loc="upper right")
fig.tight_layout()
plt.savefig('FIG7.png')
