import numpy as np

def disparity_to_color(I):
    _map = np.array([[0, 0, 0, 114], [0, 0, 1, 185], [1, 0, 0, 114], [1, 0, 1, 174],
                     [0, 1, 0, 114], [0, 1, 1, 185], [1, 1, 0, 114], [1, 1, 1, 0]]
                    )
    max_disp = 1.0 * I.max()
    I = np.minimum(I / max_disp, np.ones_like(I))

    A = I.transpose()
    num_A = A.shape[0] * A.shape[1]

    bins = _map[0:_map.shape[0] - 1, 3]
    cbins = np.cumsum(bins)
    cbins_end = cbins[-1]
    bins = bins / (1.0 * cbins_end)
    cbins = cbins[0:len(cbins) - 1] / (1.0 * cbins_end)

    A = A.reshape(1, num_A)
    B = np.tile(A, (6, 1))
    C = np.tile(np.array(cbins).reshape(-1, 1), (1, num_A))

    ind = np.minimum(sum(B > C), 6)
    bins = 1 / bins
    cbins = np.insert(cbins, 0, 0)

    A = np.multiply(A - cbins[ind], bins[ind])
    K1 = np.multiply(_map[ind, 0:3], np.tile(1 - A, (3, 1)).T)
    K2 = np.multiply(_map[ind + 1, 0:3], np.tile(A, (3, 1)).T)
    K3 = np.minimum(np.maximum(K1 + K2, 0), 1)

    return np.reshape(K3, (I.shape[1], I.shape[0], 3)).T