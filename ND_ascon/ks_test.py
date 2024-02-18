import numpy as np
from pickle import dump
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from tensorflow.keras.regularizers import l2
from scipy.stats import ks_2samp


import ascon as ac


def ks_test(n, group_size):
    model = load_model('./saved_model/best4.h5')
    X = ac.make_real_data(group_size)
    X = model.predict(X)
    X_eval, Y_eval = ac.make_td_diff(n, group_size, 4)
    X_eval = model.predict(X_eval, batch_size=10000).reshape(n, group_size)
    correct = 0
    for i, sample in enumerate(X_eval):
        ks_value = ks_2samp(sample, X.reshape(-1))
        if ks_value.pvalue > 0.05 and Y_eval[i] == 1:
            correct += 1
        if ks_value.pvalue <= 0.05 and Y_eval[i] == 0:
            correct += 1
    print(correct/n)


if __name__ == '__main__':
    ks_test(10**5, 4096)



