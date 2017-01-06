from collections import Counter


def most_common_label(labels):
    """
    Finds the most occurring element in labels
    :param labels: list of labels
    :return: most occurring label
    """

    # counts = Counter(list_lbl_poly).most_common()
    # most_common = next((x[0] for i, x in enumerate(counts) if x[0]), None)
    # return most_common

    counts = Counter(labels).most_common()
    for elem, occur in counts:
        if elem != 0:  # must be different from zero
            return elem

    return counts[0][0]
# --------------------------------------


def find_pattern(array, pattern):
    """
    Finds a given pattern in a array

    :param array: array where to look for a given pattern
    :param pattern: pattern to look for
    :return: number of repetitions of the pattern
    """
    i = 0
    nrepetitions = 0
    found = False
    for elem_array in array:
        if i < len(pattern):  # if pattern as not yet entirely been found
            if elem_array == pattern[i]:
                i += 1
            else:
                if found:  # descending slope
                    nrepetitions += 1
                found = False
                i = 0
        else:  # pattern has been found
            found = True
            i = 0

    if found:
        nrepetitions += 1

    return nrepetitions
# --------------------------------------


def add_list_to_list(list1, list2add):
    """
    Appends list2add to list1
    :param list1:
    :param list2add:
    """
    for l in list2add:
        list1.append(l)
# --------------------------------------


def remove_duplicates(listduplicates):
    """
    Removes duplicates in list
    :param listduplicates:
    :return:
    """
    # Can be replaced by :
    return list(set(listduplicates))

    # unique = list()
    # # Remove duplicates
    # for elem in listduplicates:
    #     if elem not in unique:
    #         unique.append(elem)
    #
    # return unique
