import numpy as np


def k_nearest(chip_compare, k: int, list_of_chips):
    """
    Compares a chip to a list of chips and returns a sorted list of the nearest chips.
    """
    chip_distances = []

    for chip in list_of_chips:
        dist = np.linalg.norm(chip[:2] - chip_compare[:2])  # distance computed with euclidian norm in R2 (faster)

        chip_distances.append({"chip": chip, "distance": dist})

    sort_list = sorted(chip_distances, key=lambda k: k['distance'])
    nearest = sort_list[0:k]

    return nearest


def classifyMesh(k: int, data: [], mesh):
    """
    Classify mesh colours based on nearest k-value.
    :param k:
    :param data: Data to be compared
    :param mesh: Input mesh
    :return: Array of classified data.
    """
    classified = []

    for point in mesh:
        # find k nearest
        nearest = k_nearest(point, k, data)
        # predict color
        classified.append(classifyChip(nearest))
    return np.array(classified)


def classifyErrors(k: int, X_set):
    """
    Make predictions on the training set and root out errors.
    :param k:
    :param X_set:
    :return:
    """

    fail = 0
    # on all chips in training set X
    for chip in X_set:
        # get k nearest chips to chip in X set
        y_set = k_nearest(chip, k, X_set)

        result: int = classifyChip(y_set)
        compare: int = int(chip[2])

        # make prediction on x chip and compare to real value
        if compare != result:
            fail += 1

    return fail


def classifyChip(list_of_nearest):
    """
    Predicts the outcome of a microchip by comparing with the nearest chips.
    :param list_of_nearest: A list of near microchips.
    :return: prediction FAIL OR OK
    """
    fail = 0
    for entry in list_of_nearest:
        chip = entry['chip']
        if chip[2] == 0:  # chip is failed (0)
            fail += 1

    result = fail > (len(list_of_nearest) / 2)  # more than half failed

    if result:
        return 0
    else:
        return 1


def nearestX(x_value, k: int, list_of_chips):
    """
    Fetches the k-nearest X-values of a point and returns an array sorted by distance.
    """
    chip_distances = []
    array = np.zeros((100, 3), dtype=float)
    for idx, chip in enumerate(list_of_chips):
        dist = np.abs(x_value - chip[:1])  # abs distance between (x1,x2)

        array[idx, 0] = chip[0]
        array[idx, 1] = chip[1]
        array[idx, 2] = dist
        chip_distances.append((chip, dist))

    sorted = array[array[:, 2].argsort()]

    return sorted[:k]


def MSE(y_real, y_predict):
    """ Returns the mean square error of real vs predicted y values."""
    return np.square(np.subtract(y_real, y_predict)).mean()


def predictY(x_value, k_value, training_set):
    """
    Predicts Y-values by comparing the nearest X-values of the set and
    averaging corresponding Y-values.

    :param x_steps: Steps of the graph in X-axis.
    :param k_value: KNN neighbours
    :param training_set: X_values of training set.
    :return: npArray of (STEPS, Y_PREDICT)
    """
    # find the k nearest chips in x-axis
    nearest = nearestX(x_value, k_value, training_set)

    # predict y-value by averaging nearest y-values
    y_predict = np.average(nearest[:, 1])

    return y_predict


def predictSet(k_value, set):
    """
    Predicts Y-values for an entire set.
    :param k_value:
    :param set:
    :return:
    """

    test_predict = []

    for x in set:
        test_predict.append(predictY(x[0], k_value, set))

    return np.array(test_predict).T