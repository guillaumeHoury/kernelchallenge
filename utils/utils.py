import multiprocessing as mp
import numpy as np


def count_coocurrences(list1, list2, normalized=True):
    """
    Count the number of matching elements in two sorted lists
    """
    count = 0
    i, j = 0, 0
    N, M = len(list1), len(list2)

    while i < N and j < M:
        if list1[i] == list2[j]:
            count += 1
            i += 1
            j += 1
        elif list1[i] < list2[j]:
            i += 1
        else:
            j += 1

    if normalized:
        return 2*count/(len(list1) + len(list2))
    else:
        return count


def cooccurrence_matrix(array1, array2):
    """
    Compute the co-occurence matrix between the two arrays.
    array1: list of lists
    array2: list of lists
    """
    K, L = len(array1), len(array2)

    mesh1, mesh2 = np.meshgrid(np.array(array1), np.array(array2))

    with mp.Pool() as p:
        M = p.starmap(count_coocurrences, zip(mesh1.flatten(), mesh2.flatten()))

    M = np.array(M).reshape(L, K).transpose()

    return M

