import numpy as np
from pickle import dump

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D, Concatenate
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2

import gift128 as gift

bs = 10000
wdir = './freshly_trained_nets/'


def cyclic_lr(num_epochs, high_lr, low_lr):
  res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr);
  return(res);


def make_checkpoint(datei):
  res = ModelCheckpoint(datei, monitor='val_loss', save_best_only = True);
  return(res);


#make residual tower of convolutional blocks
def make_resnet(group_size=2, num_blocks=16, num_filters=32, num_outputs=1, d1=64, d2=64, word_size=8, ks=3, depth=5, reg_param=0.001, final_activation='sigmoid'):
  #Input and preprocessing layers
  inp = Input(shape=(group_size * 2 * num_blocks * word_size, ))          # (-1, 2 * 2 * 4 * 64)
  rs = Reshape((group_size, 2 * num_blocks, word_size))(inp)      # (-1, 2, 8, 64)
  perm = Permute((1, 3, 2))(rs)         # (-1, 2, 64, 8)
  # perm = inp
  #add a single residual layer that will expand the data to num_filters channels
  #this is a bit-sliced layer
  conv0 = Conv2D(num_filters, kernel_size=1, strides=1, padding='same', kernel_regularizer=l2(reg_param))(perm);
  conv0 = BatchNormalization()(conv0)
  conv0 = Activation('relu')(conv0)

  #add residual blocks
  shortcut = conv0
  for i in range(depth):
    conv1 = Conv2D(num_filters, kernel_size=ks, strides=1, padding='same', kernel_regularizer=l2(reg_param))(shortcut);
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv2 = Conv2D(num_filters, kernel_size=ks, strides=1, padding='same', kernel_regularizer=l2(reg_param))(conv1);
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    shortcut = Add()([shortcut, conv2])
  #add prediction head
  flat1 = Flatten()(shortcut)
  dense1 = Dense(d1,kernel_regularizer=l2(reg_param))(flat1)
  dense1 = BatchNormalization()(dense1)
  dense1 = Activation('relu')(dense1)
  dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1)
  dense2 = BatchNormalization()(dense2)
  dense2 = Activation('relu')(dense2)
  out = Dense(num_outputs, activation=final_activation, kernel_regularizer=l2(reg_param))(dense2)
  model = Model(inputs=inp, outputs=out)
  return(model)


def train_gift_distinguisher(num_epochs, num_rounds=7, group_size=2, depth=1):
    #create the network
    net = make_resnet(group_size=group_size, depth=depth, reg_param=10**-5);
    net.compile(optimizer='adam', loss='mse', metrics=['acc']);
    #generate training and validation data
    X, Y = gift.make_td(n=10**7 // group_size, s=group_size, nr=num_rounds)
    X_eval, Y_eval = gift.make_td(n=10**6 // group_size, s=group_size, nr=num_rounds)
    #set up model checkpoint
    check = make_checkpoint(wdir+'best'+str(num_rounds)+'depth'+str(depth)+'.h5');
    #create learnrate schedule
    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001));
    #train and evaluate
    h = net.fit(X, Y, epochs=num_epochs, batch_size=bs, shuffle=True,
                validation_data=(X_eval, Y_eval), callbacks=[lr, check]);
    np.save(wdir+'h'+str(num_rounds)+'r_depth'+str(depth)+'.npy', h.history['val_acc']);
    np.save(wdir+'h'+str(num_rounds)+'r_depth'+str(depth)+'.npy', h.history['val_loss']);
    dump(h.history, open(wdir+'hist'+str(num_rounds)+'r_depth'+str(depth)+'.p', 'wb'));
    print("Best validation accuracy: ", np.max(h.history['val_acc']));

    return(net, h)


if __name__ == '__main__':
    train_gift_distinguisher(num_epochs=50, num_rounds=7, group_size=2)