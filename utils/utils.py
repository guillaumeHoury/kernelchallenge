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
    return: np.array of shape (len(array1), len(array2))
    """
    K, L = len(array1), len(array2)

    mesh1, mesh2 = np.meshgrid(np.array(array1), np.array(array2))

    with mp.Pool() as p:
        M = p.starmap(count_coocurrences, zip(mesh1.flatten(), mesh2.flatten()))

    M = np.array(M).reshape(L, K).transpose()

    return M


def create_hash_dict(array):
    """
    From an array of list of hashes, create a dictionary of lists that links each hash to the indices containing it in
    the array, as well as the nb of occurences in this array element.

    :param array: list of lists
    :return: dictionary of list of tuples. Each key is a hash sequence, the tuples (i, cur_count, array_length)
    correspond to the array index containing the hash, and cur_count is the nb of occurences of this hash in the array
    element i. array_length stores the length of the list contained in the ith element of the array, and used for
    normalization purpose.
    """
    hash_dict = dict()
    for i in range(len(array)):
        hashes, counts = np.unique(array[i], return_counts=True)
        for cur_hash, cur_count in zip(hashes, counts):
            if cur_hash not in hash_dict:
                hash_dict[cur_hash] = [(i, cur_count, len(array[i]))]
            else:
                hash_dict[cur_hash].append((i, cur_count, len(array[i])))

    return hash_dict


def get_hash_coocurrences(list):
    """
    Computes the co-occurrence vector between the input and all the lists appearing in the global variable
    shared_hash_dict.
    :param list: A list of hashes
    :return: The co-occurrence vector of the list and the lists of the (corresponding to one row of the co-occurrence
    matrix)0).
    """
    count_vector = np.zeros(shared_vector_size)
    hashes, counts = np.unique(list, return_counts=True)
    for cur_hash, cur_count in zip(hashes, counts):
        if cur_hash in shared_hash_dict:
            for i, hash_count, list_length in shared_hash_dict[cur_hash]:
                count_vector[i] += 2 * min(cur_count, hash_count) / (len(list) + list_length)

    return count_vector


def optimized_coocurrence_matrix(array1, array2, distributed=True):
    """
    Computes the co-occurence matrix between array1 and array2, using a dict of list to avoid redundant computations.
    :param array1: list of lists
    :param array2: list of lists
    :return: np.array of shape (len(array1), len(array2))
    """

    # Stores the hash occurrences of array2 in a dictionary
    hash_dict = create_hash_dict(array2)

    def init(hash_dict, vector_size):
        # Share common variables across the different distributed processes
        global shared_hash_dict, shared_vector_size
        shared_hash_dict, shared_vector_size = hash_dict, vector_size

    # Compute each row of the co-occurence matrix
    if distributed:
        with mp.Pool(initializer=init, initargs=(hash_dict, len(array2),)) as p:
            vect_list = p.map(get_hash_coocurrences, array1)
    else:
        init(hash_dict, len(array2))
        vect_list = [get_hash_coocurrences(cur_list) for cur_list in array1]

    K = np.vstack(vect_list)

    return K