
def minimum_edit_distance(s1, s2):
    """
    Computes the Levenshtein distence between 2 strings s1, s2
    Taken from https://rosettacode.org/wiki/Levenshtein_distance#Python
    :param s1:
    :param s2:
    :return: Levenshtein distance
    """
    if len(s1) > len(s2):
        s1, s2 = s2,s1
    distances = range(len(s1) + 1)
    for index2, char2 in enumerate(s2):
        newDistances = [index2+1]
        for index1, char1 in enumerate(s1):
            if char1 == char2:
                newDistances.append(distances[index1])
            else:
                newDistances.append(1 + min((distances[index1],
                                             distances[index1+1],
                                             newDistances[-1])))
        distances = newDistances
    return distances[-1]
# ------------------------------------------------------------------


def count_correct_characters(predicted_string, reference_string):
    """
    Counts the number of correctly recognized characters within the 'reference' string
    :param predicted_string: prediction
    :param reference_string: the correct label
    :return: number of correctly predicted digits in the number/label
    """
    n_correct_characters = 0

    for char in predicted_string:
        if char in reference_string:
            reference_string = reference_string.replace(char, '', 1)
            n_correct_characters += 1

    return n_correct_characters
