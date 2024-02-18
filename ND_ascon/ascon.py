import numpy as np
from os import urandom

constants = np.array([0xf0, 0xe1, 0xd2, 0xc3, 0xb4, 0xa5, 0x96, 0x87, 0x78, 0x69, 0x5a, 0x4b, 0x3c, 0x2d, 0x1e, 0x0f], dtype=np.uint64)


def permutation(state, rounds, t):
    for i in range(rounds):
        add_constants(state, i, rounds)
        sbox(state, t)
        linear(state)


def add_constants(state, i, a):
    state[:, 2] ^= constants[12 - a + i]


def rotate(x, l):
    temp = np.bitwise_xor(np.right_shift(x, l), np.left_shift(x, 64 - l))
    return temp  


def sbox(x, t):
    x[:, 0] ^= x[:, 4]
    x[:, 4] ^= x[:, 3]
    x[:, 2] ^= x[:, 1]
    t[:, 0] = x[:, 0]
    t[:, 1] = x[:, 1]
    t[:, 2] = x[:, 2]
    t[:, 3] = x[:, 3]
    t[:, 4] = x[:, 4]
    t[:, 0] = ~t[:, 0]
    t[:, 1] = ~t[:, 1]
    t[:, 2] = ~t[:, 2]
    t[:, 3] = ~t[:, 3]
    t[:, 4] = ~t[:, 4]
    t[:, 0] &= x[:, 1]
    t[:, 1] &= x[:, 2]
    t[:, 2] &= x[:, 3]
    t[:, 3] &= x[:, 4]
    t[:, 4] &= x[:, 0]
    x[:, 0] ^= t[:, 1]
    x[:, 1] ^= t[:, 2]
    x[:, 2] ^= t[:, 3]
    x[:, 3] ^= t[:, 4]
    x[:, 4] ^= t[:, 0]
    x[:, 1] ^= x[:, 0]
    x[:, 0] ^= x[:, 4]
    x[:, 3] ^= x[:, 2]
    x[:, 2] = ~x[:, 2]


def linear(state):
    temp0 = rotate(state[:, 0], 19)
    temp1 = rotate(state[:, 0], 28)
    state[:, 0] ^= temp0 ^ temp1
    temp0 = rotate(state[:, 1], 61)
    temp1 = rotate(state[:, 1], 39)
    state[:, 1] ^= temp0 ^ temp1
    temp0 = rotate(state[:, 2], 1)
    temp1 = rotate(state[:, 2], 6)
    state[:, 2] ^= temp0 ^ temp1
    temp0 = rotate(state[:, 3], 10)
    temp1 = rotate(state[:, 3], 17)
    state[:, 3] ^= temp0 ^ temp1
    temp0 = rotate(state[:, 4], 7)
    temp1 = rotate(state[:, 4], 41)
    state[:, 4] ^= temp0 ^ temp1


def to_binary(arr):
    binary_array = np.unpackbits(arr.view(np.uint8).reshape(-1, 8)[:, ::-1], axis=1).reshape(-1, 320)
    return binary_array


def make_td(n, s, nr):
    Y = np.frombuffer(urandom(n), dtype=np.uint8) & 1
    t = np.zeros((n * s, 5), dtype=np.uint64)
    state0 = np.random.randint(0, 2 ** 64, size=(n, 5 * s), dtype=np.uint64)
    diff = np.zeros(5 * s, dtype=np.uint64)
    diff[np.arange(0, 5 * s, 5)] = 1
    state1 = state0 ^ diff
    num_rand_samples = np.sum(Y == 0)
    state1[Y == 0, :] = np.random.randint(0, 2 ** 64, size=(num_rand_samples, 5 * s), dtype=np.uint64)
    state0 = state0.reshape(n * s, 5)
    state1 = state1.reshape(n * s, 5)

    permutation(state0, nr, t)
    permutation(state1, nr, t)
    X = np.concatenate((to_binary(state0), to_binary(state1)), axis=1)

    return X.reshape(n, 2*s*320), Y


def make_td_diff(n, s, nr):
    Y = np.frombuffer(urandom(n), dtype=np.uint8) & 1
    t = np.zeros((n*s, 5), dtype=np.uint64)
    state0 = np.random.randint(0, 2**64, size=(n, 5*s), dtype=np.uint64)
    diff = np.zeros(5*s, dtype=np.uint64)
    diff[np.arange(0, 5*s, 5)] = 1
    state1 = state0 ^ diff
    num_rand_samples = np.sum(Y == 0)
    state1[Y == 0, :] = np.random.randint(0, 2**64, size=(num_rand_samples, 5*s), dtype=np.uint64)
    state0 = state0.reshape(n*s, 5)
    state1 = state1.reshape(n*s, 5)
    permutation(state0, nr, t)
    permutation(state1, nr, t)
    X = to_binary(state0 ^ state1)
    return X, Y


def make_real_data(s):
    t = np.zeros((s, 5), dtype=np.uint64)
    state0 = np.random.randint(0, 2 ** 64, size=(s, 5), dtype=np.uint64)
    diff = np.zeros((s, 5), dtype=np.uint64)
    diff[:, 0] = 1
    state1 = state0 ^ diff
    permutation(state0, 4, t)
    permutation(state1, 4, t)
    X = to_binary(state0 ^ state1)
    return X




