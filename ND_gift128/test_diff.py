import numpy as np
from pickle import dump
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from tensorflow.keras.regularizers import l2
import pandas as pd
import gift128 as gift


diffs = [np.zeros(128, dtype=np.uint8)]


def generate_random_array(size, num_ones):
    if num_ones > size:
        raise ValueError("Number of ones cannot exceed array size.")
    exists = True
    array = np.zeros(size, dtype=np.uint8)
    while exists:
        indices = np.random.choice(size, num_ones, replace=False)
        array[indices] = 1
        np.random.shuffle(array)
        exists = np.any(np.all(np.array(diffs) == array, axis=1))
        if exists:
            array = np.zeros(size, dtype=np.uint8)
    diffs.append(array)
    return array


def train(n, num_epochs, num_rounds, batch_size, diff, f):

    X, Y = gift.make_with_diff(n, num_rounds, diff)
    X_eval, Y_eval = gift.make_with_diff(n // 10, num_rounds, diff)

    model = Sequential()
    model.add(Dense(128, input_dim=128, activation='relu', kernel_regularizer=l2(10**-5)))
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(10**-5)))
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(10**-5)))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(10**-5)))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    h = model.fit(X, Y, epochs=num_epochs, batch_size=batch_size, shuffle=True,
                  validation_data=(X_eval, Y_eval))
    print("Best validation accuracy: ", np.max(h.history['val_accuracy']))
    f.write(str(np.max(h.history['val_accuracy'])))
    f.write("\n")
    return np.max(h.history['val_accuracy'])


if __name__ == '__main__':
    f = open("results_hm=2.txt", "w")
    acc_list = []
    for i in range(128):
        print(i)
        diff = generate_random_array(128, 2)
        # diff = np.zeros(128, dtype=np.uint8)
        # diff[i] = 1
        diff = np.packbits(diff)
        f.write(str(diff))
        f.write(": ")
        acc = train(n=2 * 10 ** 6, num_epochs=50, num_rounds=6, batch_size=10000, diff=diff, f=f)
        acc_list.append(acc)
    bins = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    segments = pd.cut(acc_list, bins, right=False)
    f.write(str(segments.value_counts()))
    f.close()



