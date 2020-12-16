import os
os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from sklearn import svm
from sklearn.metrics import accuracy_score
import gzip


def make_tensor(data_xy):
    data_x, data_y = data_xy
    data_x = np.array(data_x, dtype=np.float32)
    data_y = np.array(data_y, dtype=np.int32)
    return data_x, data_y

def load_pickle(f):
    try:
        import cPickle as thepickle
    except ImportError:
        import _pickle as thepickle

    try:
        ret = thepickle.load(f, encoding='latin1')
    except TypeError:
        ret = thepickle.load(f)

    return ret

def load_data(data_file):
    print('loading data ...')
    f = gzip.open(data_file, 'rb')
    train_set, valid_set, test_set = load_pickle(f)
    f.close()

    train_set_x, train_set_y = make_tensor(train_set) # (50000, 784) (50000,)
    valid_set_x, valid_set_y = make_tensor(valid_set) # (10000, 784) (10000,)
    test_set_x, test_set_y = make_tensor(test_set) # (10000, 784) (10000,)

    return train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y


class CCA:
    def __init__(self, n_components=1, r1=1e-4, r2=1e-4):
        self.n_components = n_components
        self.r1 = r1
        self.r2 = r2
        self.w = [None, None]
        self.m = [None, None]

    def fit(self, X1, X2):
        N = X1.shape[0]
        f1 = X1.shape[1]
        f2 = X2.shape[1]

        self.m[0] = np.mean(X1, axis=0, keepdims=True) # [1, f1]
        self.m[1] = np.mean(X2, axis=0, keepdims=True)
        H1bar = X1 - self.m[0]
        H2bar = X2 - self.m[1]

        SigmaHat12 = (1.0 / (N - 1)) * np.dot(H1bar.T, H2bar)
        SigmaHat11 = (1.0 / (N - 1)) * np.dot(H1bar.T, H1bar) + self.r1 * np.identity(f1)
        SigmaHat22 = (1.0 / (N - 1)) * np.dot(H2bar.T, H2bar) + self.r2 * np.identity(f2)

        [D1, V1] = np.linalg.eigh(SigmaHat11)
        [D2, V2] = np.linalg.eigh(SigmaHat22)
        SigmaHat11RootInv = np.dot(np.dot(V1, np.diag(D1 ** -0.5)), V1.T)
        SigmaHat22RootInv = np.dot(np.dot(V2, np.diag(D2 ** -0.5)), V2.T)

        Tval = np.dot(np.dot(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)

        [U, D, V] = np.linalg.svd(Tval)
        V = V.T
        self.w[0] = np.dot(SigmaHat11RootInv, U[:, 0:self.n_components])
        self.w[1] = np.dot(SigmaHat22RootInv, V[:, 0:self.n_components])
        D = D[0:self.n_components]

    def _get_result(self, x, idx):
        result = x - self.m[idx].reshape([1, -1]).repeat(len(x), axis=0)
        result = np.dot(result, self.w[idx])
        return result

    def test(self, X1, X2):
        return self._get_result(X1, 0), self._get_result(X2, 1)



# load data
# Note: y1 == y2, two views are from the same number.
X1_train, y1_train, X1_valid, y1_valid, X1_test, y1_test = load_data('../data/mnist/noisymnist_view1.gz')
X2_train, y2_train, X2_valid, y2_valid, X2_test, y2_test = load_data('../data/mnist/noisymnist_view2.gz')


# CCA feature extraction
model = CCA(n_components=10)
model.fit(X1_train, X2_train)

# manual project
Z1_train, Z2_train = model.test(X1_train, X2_train)
Z1_valid, Z2_valid = model.test(X1_valid, X2_valid)
Z1_test, Z2_test = model.test(X1_test, X2_test)

# T-SNE of X1
tsne = TSNE()
pca = PCA(n_components=10)

X1_pca = pca.fit_transform(X1_test)

X1_tsne = tsne.fit_transform(X1_pca)

plt.scatter(X1_tsne[:, 0], X1_tsne[:, 1], c=y1_test, cmap='tab10')
plt.show()

# T-SNE of Z1
tsne = TSNE()
Z1_tsne = tsne.fit_transform(Z1_test)

plt.scatter(Z1_tsne[:, 0], Z1_tsne[:, 1], c=y1_test, cmap='tab10')
plt.show()

# quantitatively measure learned corr
def calc_corr(Z1, Z2):
    N, F = Z1.shape
    #corrs = np.corrcoef(Z1.T, Z2.T) # this get cov and cross-cov
    corrs = [np.corrcoef(z1, z2)[0, 1] for z1, z2 in zip(Z1.T, Z2.T)] # only cross-cov
    return corrs

print(np.mean(calc_corr(Z1_valid, Z2_valid)))
print(np.mean(calc_corr(Z1_test, Z2_test)))

# SVM baseline
clf = svm.LinearSVC(C=0.01, dual=False)
clf.fit(X1_train, y1_train)

train_acc = accuracy_score(y1_train, clf.predict(X1_train))
valid_acc = accuracy_score(y1_valid, clf.predict(X1_valid))
test_acc = accuracy_score(y1_test, clf.predict(X1_test))

print(train_acc, valid_acc, test_acc)

# SVM classify
clf = svm.LinearSVC(C=0.01, dual=False)
clf.fit(Z1_train, y1_train)

train_acc = accuracy_score(y1_train, clf.predict(Z1_train))
valid_acc = accuracy_score(y1_valid, clf.predict(Z1_valid))
test_acc = accuracy_score(y1_test, clf.predict(Z1_test))

print(train_acc, valid_acc, test_acc)
