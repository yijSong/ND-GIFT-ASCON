import numpy as np
from pickle import dump
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from tensorflow.keras.regularizers import l2

import gift128 as gift


wdir = './saved_model/'


def make_checkpoint(datei):
    res = ModelCheckpoint(datei, monitor='val_loss', save_best_only=True)
    return res


def train(n, m, num_epochs, num_rounds, batch_size, group_size):
    X, Y = gift.make_td_diff(n, 1, num_rounds)
    X_eval, Y_eval = gift.make_td_diff(m, 1, num_rounds)

    model = Sequential()
    model.add(Dense(128, input_dim=128, activation='relu', kernel_regularizer=l2(10**-5)))
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(10**-5)))
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(10**-5)))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(10**-5)))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    check = make_checkpoint(wdir + 'best' + str(num_rounds) + '.h5')

    h = model.fit(X, Y, epochs=num_epochs, batch_size=batch_size, shuffle=True,
                  validation_data=(X_eval, Y_eval), callbacks=[check])
    print("Best validation accuracy: ", np.max(h.history['val_accuracy']))

    model_s = Sequential()
    model_s.add(Dense(64, input_dim=group_size, activation='relu', kernel_regularizer=l2(10 ** -5)))
    model_s.add(Dense(64, activation='relu', kernel_regularizer=l2(10 ** -5)))
    model_s.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(10 ** -5)))

    X, Y = gift.make_td_diff(n // group_size, group_size, num_rounds)
    X_eval, Y_eval = gift.make_td_diff(m // group_size, group_size, num_rounds)
    X = np.sort(model.predict(X).reshape(n // group_size, group_size))
    X_eval = np.sort(model.predict(X_eval).reshape(m // group_size, group_size))

    model_s.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    check = make_checkpoint(wdir + 'best' + str(num_rounds) + '_s=' + str(group_size) + '.h5')

    h = model_s.fit(X, Y, epochs=num_epochs, batch_size=batch_size, shuffle=True,
                    validation_data=(X_eval, Y_eval), callbacks=[check])
    print("Best validation accuracy: ", np.max(h.history['val_accuracy']))


if __name__ == '__main__':
    train(n=10 ** 7, m=10 ** 6, num_epochs=50, num_rounds=6, batch_size=10000, group_size=4)


