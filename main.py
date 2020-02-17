import math
import random
import numpy as np
import matplotlib.pyplot as plt
import headers.constants as constants

from headers.cmac import CMAC


def generate_dataset(function, input_space_size=constants.plot_input_space_size, minimum_input_value=0,
                     maximum_input_value=360, dataset_split_factor=0.7):
    """ generate dataset for CMAC with 70:30 train to test ratio
    :param function: mathematical function to test and train CMAC, by default consider sinusoid
    :param input_space_size: total no. of points for entire dataset
    :param minimum_input_value: default value 0
    :param maximum_input_value: default value 360
    :param dataset_split_factor: specify the ratio to split training and testing data points
    :return: list of parameters for the dataset
    """
    # Define step size to get 100 data points
    step_size = (maximum_input_value - minimum_input_value) / float(input_space_size)
    # Get 100 x and y for the sinusoid function
    input_space = [math.radians(step_size * (i + 1)) for i in range(0, input_space_size)]
    output_space = [function(input_space[i]) for i in range(0, input_space_size)]
    # Get size of training and testing dataset
    training_set_size = int(input_space_size * dataset_split_factor)
    testing_set_size = input_space_size - training_set_size
    # Generate a zeros array for training dataset
    unsorted_training_set_input = np.zeros(training_set_size).tolist()
    unsorted_training_set_output = np.zeros(training_set_size).tolist()
    training_set_global_indices = np.zeros(training_set_size).tolist()
    # Generate a zeros array for testing dataset
    unsorted_testing_set_input = np.zeros(testing_set_size).tolist()
    unsorted_testing_set_true_output = np.zeros(testing_set_size).tolist()
    testing_set_global_indices = np.zeros(testing_set_size).tolist()

    count = 0
    randomized_range_values = [x for x in range(0, input_space_size)]
    random.shuffle(randomized_range_values)

    input_step_size = (math.radians(maximum_input_value) - math.radians(minimum_input_value)) / float(input_space_size)
    # Add data to empty arrays
    for i in randomized_range_values:
        if count < training_set_size:
            unsorted_training_set_input[count] = input_space[i]
            unsorted_training_set_output[count] = output_space[i]
            training_set_global_indices[count] = i
        else:
            unsorted_testing_set_input[count - training_set_size] = input_space[i] + (random.randrange(0, 10) * 0.01)
            output_space[i] = function(unsorted_testing_set_input[count - training_set_size])
            unsorted_testing_set_true_output[count - training_set_size] = output_space[i]
            testing_set_global_indices[count - training_set_size] = i
        # Increment count to add data at the correct index of the array
        count = count + 1

    return [input_space, output_space, unsorted_training_set_input, unsorted_training_set_output,
            training_set_global_indices, unsorted_testing_set_input, unsorted_testing_set_true_output,
            testing_set_global_indices, input_step_size]


def run_cmac(function):
    """ run CMAC for a given function, function can be provided using math library
    :param function: mathematical function to test and train CMAC, by default consider cosine
    :return: nothing
    """

    # Generate dataset for the sinusoid
    dataset = generate_dataset(function)
    # Run CMAC for various generalization factors
    discrete_cmac = [CMAC(i, dataset, 'DISCRETE') for i in
                     range(constants.min_generalization_factor, constants.max_generalization_factor + 1)]
    continuous_cmac = [CMAC(i, dataset, 'CONTINUOUS') for i in
                       range(constants.min_generalization_factor, constants.max_generalization_factor + 1)]

    training_errors_discrete_cmac = np.zeros(constants.max_generalization_factor).tolist()
    testing_errors_discrete_cmac = np.zeros(constants.max_generalization_factor).tolist()

    training_errors_continuous_cmac = np.zeros(constants.max_generalization_factor).tolist()
    testing_errors_continuous_cmac = np.zeros(constants.max_generalization_factor).tolist()

    convergence_times_discrete = np.zeros(constants.max_generalization_factor).tolist()
    convergence_times_continuous = np.zeros(constants.max_generalization_factor).tolist()

    best_discrete_cmac = -1
    best_continuous_cmac = -1

    lowest_testing_error_discrete = -1
    lowest_testing_error_continuous = -1

    # Plot a cmac with no generalization effect
    print('\nPlot Generalization Factor = ' + str(constants.plot_generalization_factor + 1) + ' with Errors \n')
    continuous_cmac[constants.plot_generalization_factor].execute()
    continuous_cmac[constants.plot_generalization_factor].plot_graphs()

    discrete_cmac[constants.plot_generalization_factor].execute()
    discrete_cmac[constants.plot_generalization_factor].plot_graphs()

    # Plot a cmac with the best generalization effect
    print('\nGeneralization Factor Variance - CMAC Performance \n ')
    for i in range(0, constants.max_generalization_factor):
        training_errors_discrete_cmac[i], testing_errors_discrete_cmac[i] = discrete_cmac[i].execute()
        training_errors_continuous_cmac[i], testing_errors_continuous_cmac[i] = continuous_cmac[i].execute()

        print('Generalization Factor - ' + str(i + 1) + ' Continuous Testing Error - ' + str(
            round(testing_errors_continuous_cmac[i], 3)) + ' Continuous Convergence Time - ' + str(
            round(continuous_cmac[i].convergence_time, 2)) + ' Discrete Testing Error - ' + str(
            round(testing_errors_discrete_cmac[i], 3)))

        convergence_times_discrete[i] = discrete_cmac[i].convergence_time
        convergence_times_continuous[i] = continuous_cmac[i].convergence_time

        if testing_errors_discrete_cmac[i] > lowest_testing_error_discrete:
            lowest_testing_error_discrete = testing_errors_discrete_cmac[i]
            best_discrete_cmac = i

        if testing_errors_continuous_cmac[i] > lowest_testing_error_continuous:
            lowest_testing_error_continuous = testing_errors_continuous_cmac[i]
            best_continuous_cmac = i

    if best_discrete_cmac is not -1:
        discrete_cmac[best_discrete_cmac].plot_graphs()
    else:
        print("Error - Discrete CMAC")

    if best_continuous_cmac is not -1:
        continuous_cmac[best_continuous_cmac].plot_graphs()
    else:
        print("Error - Continuous CMAC")

    # Plot performance graphs with increasing generalization factor
    plot_performance(training_errors_discrete_cmac, testing_errors_discrete_cmac, training_errors_continuous_cmac,
                     testing_errors_continuous_cmac, convergence_times_discrete, convergence_times_continuous,
                     'GeneralizationFactor')


def plot_performance(training_errors_discrete_cmac, testing_errors_discrete_cmac, training_errors_continuous_cmac,
                     testing_errors_continuous_cmac, convergence_times_discrete, convergence_times_continuous,
                     x_label):
    """ plot cmac performance graphs
    :param training_errors_discrete_cmac: list of training errors for discrete CMAC
    :param testing_errors_discrete_cmac: list of testing errors for discrete CMAC
    :param training_errors_continuous_cmac: list of training errors for continuous CMAC
    :param testing_errors_continuous_cmac: list of testing errors for continuous CMAC
    :param convergence_times_discrete: list of time of convergence for discrete CMAC
    :param convergence_times_continuous: list of time of convergence for continuous CMAC
    :param x_label: x-label for graphs
    :return: nothing
    """

    if x_label is str('GeneralizationFactor'):
        range_values = range(1, constants.max_generalization_factor + 1)
        value = 'Input Space Size = ' + str(constants.plot_input_space_size)
        x_label = 'Generalization Factor'
    else:
        range_values = range(constants.min_input_space_size, constants.max_input_space_size,
                             constants.input_space_step_size)
        value = 'Generalization Factor = ' + str(constants.plot_generalization_factor)

    # Plot discrete CMAC performance results
    plt.figure(figsize=(20, 10))
    # Plot testing accuracy graph for discrete CMAC
    plt.subplot(121)
    plt.plot(range_values, testing_errors_discrete_cmac, 'b', label='DISCRETE CMAC')
    plt.xlabel(x_label)
    plt.ylabel('Testing Accuracy')
    plt.title('Discrete CMAC ' + '\n' + value + '\n' + 'Testing Accuracy vs ' + x_label)
    # Plot convergence time graph for discrete CMAC
    plt.subplot(122)
    plt.plot(convergence_times_discrete, testing_errors_discrete_cmac)
    plt.xlabel('Convergence Times')
    plt.ylabel('Testing Accuracy')
    plt.title('Discrete CMAC ' + '\n' + value + '\n' + 'Testing Accuracy vs ' + 'Convergence Times')
    plt.show()
    # Plot training accuracy graph for discrete CMAC
    plt.figure(figsize=(20, 10))
    plt.plot(range_values, training_errors_discrete_cmac, 'b', label='DISCRETE CMAC')
    plt.xlabel(x_label)
    plt.ylabel('Training Accuracy')
    plt.title('Discrete CMAC ' + '\n' + value + '\n' + 'Training Accuracy vs ' + x_label)

    # Plot continuous CMAC performance results
    plt.figure(figsize=(20, 10))
    # Plot testing accuracy graph for continuous CMAC
    plt.subplot(121)
    plt.plot(range_values, testing_errors_continuous_cmac, 'b', label='CONTINUOUS CMAC')
    plt.xlabel(x_label)
    plt.ylabel('Testing Accuracy')
    plt.title('Continuous CMAC ' + '\n' + value + '\n' + 'Testing Accuracy vs ' + x_label)
    # Plot convergence time graph for continuous CMAC
    plt.subplot(122)
    plt.plot(convergence_times_continuous, testing_errors_continuous_cmac)
    plt.xlabel('Convergence Times')
    plt.ylabel('Testing Accuracy')
    plt.title('Continuous CMAC ' + '\n' + value + '\n' + 'Testing Accuracy vs ' + 'Convergence Times')
    plt.show()
    # Plot training accuracy graph for discrete CMAC
    plt.figure(figsize=(20, 10))
    plt.plot(range_values, training_errors_continuous_cmac, 'b', label='CONTINUOUS CMAC')
    plt.xlabel(x_label)
    plt.ylabel('Training Accuracy')
    plt.title('Continuous CMAC ' + '\n' + value + '\n' + 'Training Accuracy vs ' + x_label)

    # Plot graphs for comparison between discrete and continuous CMACs
    plt.figure(figsize=(20, 11))
    # Compare testing errors of discrete and continuous CMAC
    plt.subplot(121)
    plt.plot(range_values, testing_errors_discrete_cmac, 'b', label='DISCRETE CMAC')
    plt.plot(range_values, testing_errors_continuous_cmac, 'r', label='CONTINUOUS CMAC')
    plt.xlabel(x_label)
    plt.ylabel('Testing Accuracy')
    plt.legend(loc='lower right')
    plt.title('CMAC ' + '\n' + value + '\n' + 'Testing Accuracy vs ' + x_label)
    # Compare convergence times of discrete and continuous CMAC
    plt.subplot(122)
    plt.plot(range_values, convergence_times_discrete, 'b', label='DISCRETE CMAC')
    plt.plot(range_values, convergence_times_continuous, 'r', label='CONTINUOUS CMAC')
    plt.xlabel(x_label)
    plt.ylabel('Convergence Times')
    plt.legend(loc='lower right')
    plt.title('\n' + 'Convergence Times vs ' + x_label)
    plt.show()


if __name__ == '__main__':
    # Run CMAC for sinusoidal function
    run_cmac(math.sin)
