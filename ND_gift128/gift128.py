import numpy as np
from os import urandom

TAGBYTES = 16
COFB_ENCRYPT = True
COFB_DECRYPT = False


GIFT_RC = [0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3E, 0x3D, 0x3B, 0x37, 0x2F,
           0x1E, 0x3C, 0x39, 0x33, 0x27, 0x0E, 0x1D, 0x3A, 0x35, 0x2B,
           0x16, 0x2C, 0x18, 0x30, 0x21, 0x02, 0x05, 0x0B, 0x17, 0x2E,
           0x1C, 0x38, 0x31, 0x23, 0x06, 0x0D, 0x1B, 0x36, 0x2D, 0x1A]


def rowperm(S, B0_pos, B1_pos, B2_pos, B3_pos):
    T = np.zeros_like(S)
    for b in range(8):
        T |= ((S >> (4 * b + 0)) & 0x1) << (b + 8 * B0_pos)
        T |= ((S >> (4 * b + 1)) & 0x1) << (b + 8 * B1_pos)
        T |= ((S >> (4 * b + 2)) & 0x1) << (b + 8 * B2_pos)
        T |= ((S >> (4 * b + 3)) & 0x1) << (b + 8 * B3_pos)
    return T


def giftb128(P, K, C, nr):
    S = np.zeros((P.shape[0], 4), dtype=np.uint32)
    W = np.zeros((P.shape[0], 8), dtype=np.uint16)

    S[:, 0] = (np.uint32(P[:, 0]) << 24) | (np.uint32(P[:, 1]) << 16) | (np.uint32(P[:, 2]) << 8) | np.uint32(P[:, 3])
    S[:, 1] = (np.uint32(P[:, 4]) << 24) | (np.uint32(P[:, 5]) << 16) | (np.uint32(P[:, 6]) << 8) | np.uint32(P[:, 7])
    S[:, 2] = (np.uint32(P[:, 8]) << 24) | (np.uint32(P[:, 9]) << 16) | (np.uint32(P[:, 10]) << 8) | np.uint32(P[:, 11])
    S[:, 3] = (np.uint32(P[:, 12]) << 24) | (np.uint32(P[:, 13]) << 16) | (np.uint32(P[:, 14]) << 8) | np.uint32(P[:, 15])

    W[:, 0] = (np.uint16(K[:, 0]) << 8) | np.uint16(K[:, 1])
    W[:, 1] = (np.uint16(K[:, 2]) << 8) | np.uint16(K[:, 3])
    W[:, 2] = (np.uint16(K[:, 4]) << 8) | np.uint16(K[:, 5])
    W[:, 3] = (np.uint16(K[:, 6]) << 8) | np.uint16(K[:, 7])
    W[:, 4] = (np.uint16(K[:, 8]) << 8) | np.uint16(K[:, 9])
    W[:, 5] = (np.uint16(K[:, 10]) << 8) | np.uint16(K[:, 11])
    W[:, 6] = (np.uint16(K[:, 12]) << 8) | np.uint16(K[:, 13])
    W[:, 7] = (np.uint16(K[:, 14]) << 8) | np.uint16(K[:, 15])

    for round in range(nr):
        S[:, 1] ^= S[:, 0] & S[:, 2]
        S[:, 0] ^= S[:, 1] & S[:, 3]
        S[:, 2] ^= S[:, 0] | S[:, 1]
        S[:, 3] ^= S[:, 2]
        S[:, 1] ^= S[:, 3]
        S[:, 3] ^= 0xFFFFFFFF
        S[:, 2] ^= S[:, 0] & S[:, 1]

        T = S[:, 0].copy()
        S[:, 0] = S[:, 3]
        S[:, 3] = T

        S[:, 0] = rowperm(S[:, 0], 0, 3, 2, 1)
        S[:, 1] = rowperm(S[:, 1], 1, 0, 3, 2)
        S[:, 2] = rowperm(S[:, 2], 2, 1, 0, 3)
        S[:, 3] = rowperm(S[:, 3], 3, 2, 1, 0)

        S[:, 2] ^= (np.uint32(W[:, 2]) << 16) | np.uint32(W[:, 3])
        S[:, 1] ^= (np.uint32(W[:, 6]) << 16) | np.uint32(W[:, 7])

        S[:, 3] ^= (0x80000000 ^ GIFT_RC[round])

        T6 = ((W[:, 6] >> 2) & 0xFFFF) | ((W[:, 6] << 14) & 0xFFFF)
        T7 = ((W[:, 7] >> 12) & 0xFFFF) | ((W[:, 7] << 4) & 0xFFFF)
        W[:, 7] = W[:, 5]
        W[:, 6] = W[:, 4]
        W[:, 5] = W[:, 3]
        W[:, 4] = W[:, 2]
        W[:, 3] = W[:, 1]
        W[:, 2] = W[:, 0]
        W[:, 1] = T7
        W[:, 0] = T6

    C[:, 0] = (S[:, 0] >> 24) & 0xFF
    C[:, 1] = (S[:, 0] >> 16) & 0xFF
    C[:, 2] = (S[:, 0] >> 8) & 0xFF
    C[:, 3] = S[:, 0] & 0xFF
    C[:, 4] = (S[:, 1] >> 24) & 0xFF
    C[:, 5] = (S[:, 1] >> 16) & 0xFF
    C[:, 6] = (S[:, 1] >> 8) & 0xFF
    C[:, 7] = S[:, 1] & 0xFF
    C[:, 8] = (S[:, 2] >> 24) & 0xFF
    C[:, 9] = (S[:, 2] >> 16) & 0xFF
    C[:, 10] = (S[:, 2] >> 8) & 0xFF
    C[:, 11] = S[:, 2] & 0xFF
    C[:, 12] = (S[:, 3] >> 24) & 0xFF
    C[:, 13] = (S[:, 3] >> 16) & 0xFF
    C[:, 14] = (S[:, 3] >> 8) & 0xFF
    C[:, 15] = S[:, 3] & 0xFF


def to_binary(arr):
    binary_array = np.unpackbits(arr, axis=1)
    return binary_array


def make_td(n, s, nr):
    Y = np.frombuffer(urandom(n), dtype=np.uint8) & 1

    plain0 = np.random.randint(0, 2 ** 8, size=(n, 16 * s), dtype=np.uint8)
    diff = np.zeros(16 * s, dtype=np.uint8)
    diff[np.arange(4, 16 * s, 16)] = 8
    plain1 = plain0 ^ diff
    num_rand_samples = np.sum(Y == 0)
    plain1[Y == 0, :] = np.random.randint(0, 2**8, size=(num_rand_samples, 16*s), dtype=np.uint8)
    key = np.random.randint(0, 2 ** 8, size=(n*s, 16), dtype=np.uint8)
    c0 = np.zeros((n*s, 16), dtype=np.uint8)
    c1 = np.zeros((n*s, 16), dtype=np.uint8)
    plain0 = plain0.reshape(n*s, 16)
    plain1 = plain1.reshape(n*s, 16)
    giftb128(plain0, key, c0, nr)
    giftb128(plain1, key, c1, nr)

    X = np.concatenate((to_binary(c0), to_binary(c1)), axis=1)

    return X.reshape(n, 2*s*128), Y


def make_td_diff(n, s, nr):
    Y = np.frombuffer(urandom(n), dtype=np.uint8) & 1

    plain0 = np.random.randint(0, 2 ** 8, size=(n, 16 * s), dtype=np.uint8)
    diff = np.zeros(16 * s, dtype=np.uint8)
    diff[np.arange(4, 16 * s, 16)] = 8
    plain1 = plain0 ^ diff
    num_rand_samples = np.sum(Y == 0)
    plain1[Y == 0, :] = np.random.randint(0, 2 ** 8, size=(num_rand_samples, 16 * s), dtype=np.uint8)
    key = np.random.randint(0, 2 ** 8, size=(n * s, 16), dtype=np.uint8)
    c0 = np.zeros((n * s, 16), dtype=np.uint8)
    c1 = np.zeros((n * s, 16), dtype=np.uint8)
    plain0 = plain0.reshape(n * s, 16)
    plain1 = plain1.reshape(n * s, 16)
    giftb128(plain0, key, c0, nr)
    giftb128(plain1, key, c1, nr)

    X = to_binary(c0 ^ c1)

    return X, Y


def make_with_diff(n, nr, diff):
    Y = np.frombuffer(urandom(n), dtype=np.uint8) & 1

    plain0 = np.random.randint(0, 2 ** 8, size=(n, 16), dtype=np.uint8)
    plain1 = plain0 ^ diff
    num_rand_samples = np.sum(Y == 0)
    plain1[Y == 0, :] = np.random.randint(0, 2 ** 8, size=(num_rand_samples, 16), dtype=np.uint8)
    key = np.random.randint(0, 2 ** 8, size=(n, 16), dtype=np.uint8)
    c0 = np.zeros((n, 16), dtype=np.uint8)
    c1 = np.zeros((n, 16), dtype=np.uint8)
    giftb128(plain0, key, c0, nr)
    giftb128(plain1, key, c1, nr)

    X = to_binary(c0 ^ c1)

    return X, Y


def make_real_data(s, nr):
    plain0 = np.random.randint(0, 2 ** 8, size=(s, 16), dtype=np.uint8)
    diff = np.zeros((s, 16), dtype=np.uint8)
    diff[:, 4] = 8
    plain1 = plain0 ^ diff
    key = np.random.randint(0, 2 ** 8, size=(s, 16), dtype=np.uint8)
    c0 = np.zeros((s, 16), dtype=np.uint8)
    c1 = np.zeros((s, 16), dtype=np.uint8)
    giftb128(plain0, key, c0, nr)
    giftb128(plain1, key, c1, nr)

    X = to_binary(c0 ^ c1)
    return X





