import torch
import numpy as np

try:
    from lightonopu.opu import OPU
except:
    pass
from sklearn.preprocessing import LabelBinarizer
from dt_utils import mnist


class RidgeClassifier:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, x, y):
        lb = LabelBinarizer(pos_label=1, neg_label=-1)
        labels = [i for i in range(int(torch.max(y)) + 1)]
        lb.fit(labels)
        y = lb.transform(y)
        y = torch.from_numpy(y).float()

        u, d, v = torch.svd(x.float())

        d = torch.from_numpy(np.diag(d / (d ** 2 + self.alpha)))
        self.beta = v @ d @ u.t() @ y
        return self

    def score(self, x, y):
        return (torch.argmax((x @ self.beta), dim=1) + 1 == y).mean()

    def predictions(self, x):
        return x.float() @ self.beta


def doubletrouble(X_train_rf, y_train, X_test_rf, y_test, rf, k, alpha, data_train, data_test, h):
    pred_train, pred_test = torch.zeros(len(y_train), 10), torch.zeros(len(y_test), 10)
    rf_index = 0
    for i in range(1, len(k)):
        print('ensembling k = {}'.format(k[i]))
        for it in range(k[i] - k[i - 1]):
            a, b = rf_index * rf, (rf_index + 1) * rf
            rc = RidgeClassifier(alpha=alpha).fit(X_train_rf[:, a:b], y_train)
            pred_train += rc.predictions(X_train_rf[:, a:b])
            pred_test += rc.predictions(X_test_rf[:, a:b])
            rf_index += 1

        train_score = (np.argmax(pred_train, axis=1) == y_train).float().mean()
        test_score = (np.argmax(pred_test, axis=1) == y_test).float().mean()
        print('k = {}, train = {}, test = {}'.format(k[i], train_score, test_score))
        data_train[i, h] = train_score
        data_test[i, h] = test_score

    return data_train, data_test


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('parameters')
    parser.add_argument("-alpha", default=1, type=float, help='regularization for ridge classifier')
    parser.add_argument("-n_train_samples", default=5000, type=int, help='# of data points for the training set')
    args = parser.parse_args()
    alpha = args.alpha  # alpha need to be strictly positive.
    nts = args.n_train_samples

    k = [0, 1, 2, 5, 10]  # list of ensembling.
    rfs = [int(i * 1e3) for i in range(1, 11)]  # list of rf
    rfs.reverse()

    data_train = np.zeros((len(k), len(rfs)))
    data_test = np.zeros((len(k), len(rfs)))

    train_dl, test_dl = mnist(batch_size=nts, binary=True, encoder='autoencoder')

    X_train, y_train = iter(train_dl).next()
    X_test, y_test = iter(test_dl).next()
    X_train, X_test = X_train.view(X_train.shape[0], -1), X_test.view(X_test.shape[0], -1)

    # --- OPU ---
    r_opu = OPU(n_components=int(max(rfs) * max(k)), verbose_level=1)
    r_opu.open()
    X_train_rf = r_opu.transform1d(X_train)
    X_test_rf = r_opu.transform1d(X_test)
    r_opu.close()
    # -----------

    print('Check data dim. X_train = {}, y_train = {}'.format(X_train_rf.shape, y_train.shape))

    for i, rf in enumerate(rfs):
        print('rf: {}'.format(rf))
        data_train, data_test = doubletrouble(X_train_rf, y_train, X_test_rf, y_test, rf, k, alpha, data_train,
                                              data_test, i)
    np.savez('doubletrouble_alpha_{}.npz'.format(alpha), data_train=data_train[1:], data_test=data_test[1:],
             rf=rfs, k=k[1:])
