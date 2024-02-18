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
    X_eval, Y_eval = ac.make_td_diff(n // group_size, group_size, 4)
    X_eval = np.sort(model.predict(X_eval, batch_size=10000).reshape(n // group_size, group_size))
    model_s = load_model('./saved_model/best_s=32.h5')
    loss, accuracy = model_s.evaluate(X_eval, Y_eval)
    
    print('loss=', loss)
    print('acc=', accuracy)


if __name__ == '__main__':
    ks_test(10**6, 32)



