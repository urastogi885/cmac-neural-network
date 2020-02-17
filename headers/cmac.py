import math
import time
import numpy as np
import matplotlib.pyplot as plt
import headers.constants as constants


class CMAC:

    def __init__(self, generalization_factor, data_set, cmac_type='CONTINUOUS', local_convergence_threshold=0.01,
                 learning_rate=0.15, global_convergence_threshold=0.01, max_global_convergence_iterations=500):
        """ initialize the CMAC class
        :param generalization_factor: factor for overall convergence
        :param data_set: entire dataset including both training and testing data
        :param cmac_type: only 2 types, discrete or continuous
        :param local_convergence_threshold: error threshold for local convergence
        :param learning_rate: rate of learning
        :param global_convergence_threshold: error threshold for global convergence
        :param max_global_convergence_iterations: No. of times to to execute CMAC to achieve global convergence
        :return nothing
        """
        
        self.generalization_factor = generalization_factor
        self.neighborhood_parsing_factor = int(math.floor(generalization_factor / 2))

        self.input_space = data_set[0]
        self.output_space = data_set[1]
        self.input_step_size = data_set[8]
        self.input_space_size = len(self.input_space)

        self.training_set_input = data_set[2]
        self.training_set_output = data_set[3]
        self.training_set_size = len(data_set[2])
        self.training_set_global_indices = data_set[4]
        self.training_set_cmac_output = np.zeros(self.training_set_size).tolist()

        self.testing_set_input = data_set[5]
        self.testing_set_output = data_set[6]
        self.testing_set_global_indices = data_set[7]
        self.testing_set_size = len(data_set[5])
        self.testing_set_cmac_output = np.zeros(self.testing_set_size).tolist()

        self.weights = np.zeros(self.input_space_size).tolist()

        self.cmac_type = cmac_type
        self.learning_rate = learning_rate
        self.local_convergence_threshold = local_convergence_threshold

        self.global_convergence_threshold = global_convergence_threshold
        self.max_global_convergence_iterations = max_global_convergence_iterations

        self.training_error = 1.0
        self.testing_error = 1.0
        self.convergence = False
        self.convergence_time = 1000

    def train(self):
        """ train CMAC until local convergence is achieved
        :param: none
        :return: nothing
        """
        for i in range(0, self.training_set_size):

            local_convergence = False

            global_index = self.training_set_global_indices[i]
            error = 0
            iteration = 0
            generalization_factor_offset = 0

            # Compute An Offset To Account For When Computing Weights For Edge Cases,
            # Where You May Not Get The Specified Neighborhood Size
            if i - self.neighborhood_parsing_factor < 0:
                generalization_factor_offset = i - self.neighborhood_parsing_factor

            if i + self.neighborhood_parsing_factor >= self.training_set_size:
                generalization_factor_offset = self.training_set_size - (i + self.neighborhood_parsing_factor)

            # Repeat Till You Achieve Local Convergence
            while local_convergence is False:
                local_cmac_output = 0
                # Compute Weights Based On The Neighborhood Of The Data Point, 
                # That Is Specified By 'generalization_factor'	
                for j in range(0, self.generalization_factor):
                    global_neighborhood_index = global_index - (j - self.neighborhood_parsing_factor)

                    # Make Sure The Index Is Not Beyond Limit When Taking Into Account The Generalization Factor
                    if 0 <= global_neighborhood_index < len(self.weights):
                        # Update Weights According To The Computed Error
                        self.weights[global_neighborhood_index] = self.weights[global_neighborhood_index] + (error / (
                                self.generalization_factor + generalization_factor_offset)) * self.learning_rate
                        # Compute the output
                        local_cmac_output = local_cmac_output + (
                                self.input_space[global_neighborhood_index] * self.weights[global_neighborhood_index])

                error = self.training_set_output[i] - local_cmac_output

                iteration = iteration + 1

                if iteration > 25:
                    break

                # Local Convergence Is Achieved If Absolute value Of Error Is Within The Threshold
                if abs(self.training_set_output[i] - local_cmac_output) <= self.local_convergence_threshold:
                    local_convergence = True

    def test(self, data_set_type):
        """ test CMAC with specified data type
        :param: data_set_type: declare testing or training data
        :return: nothing
        """
        cumulative_error = 0

        if data_set_type is 'TestingData':
            data_set_input = self.testing_set_input
            data_set_true_output = self.testing_set_output
            data_set_global_indices = self.testing_set_global_indices
        else:
            data_set_input = self.training_set_input
            data_set_true_output = self.training_set_output
            data_set_global_indices = self.training_set_global_indices

        local_cmac_output = [0 for i in range(0, len(data_set_input))]

        for i in range(0, len(data_set_input)):

            weight = 0

            if data_set_type is 'TestingData':
                # Find index of nearest value in Input Space, to the element in the testing data_set
                global_index = find_nearest(self.input_space, data_set_input[i])
            else:
                global_index = data_set_global_indices[i]

            # Calculate the difference between nearest value and actual value in terms of input step size
            percentage_difference_in_value = (self.input_space[global_index] - data_set_input[i]) / float(
                self.input_step_size)
            # If the actual value is lesser than nearest value,
            # slide window to the left, partial overlap for first and last element
            if percentage_difference_in_value < 0:
                max_offset = 0
                min_offset = -1
            # If the actual value is higher than the nearest value, slide window to the right,
            # partial overlap for first and last element
            elif percentage_difference_in_value > 0:
                max_offset = 1
                min_offset = 0
            # If its equal, then dont slide the window , all the elements must be completely overlapped
            else:
                max_offset = 0
                min_offset = 0

            # Compute CMAC output based on weights of all the elements in the neighborhood
            for j in range(min_offset, self.generalization_factor + max_offset):
                global_neighborhood_index = global_index - (j - self.neighborhood_parsing_factor)
                # Make sure global neighborhood index does not go out of bounds
                # Use complete overlap for Discrete CMAC
                # Use partial overlap for Continuous CMAC by generating a sliding window
                if 0 <= global_neighborhood_index < len(self.weights):
                    if j is min_offset:
                        if self.cmac_type is 'DISCRETE':
                            weight = self.weights[global_neighborhood_index]
                        if self.cmac_type is 'CONTINUOUS':
                            weight = self.weights[global_neighborhood_index] * (1 - abs(percentage_difference_in_value))
                    elif j is self.generalization_factor + max_offset - 1:
                        if self.cmac_type is 'DISCRETE':
                            weight = 0
                        if self.cmac_type is 'CONTINUOUS':
                            weight = self.weights[global_neighborhood_index] * abs(percentage_difference_in_value)
                    else:
                        weight = self.weights[global_neighborhood_index]

                    # Compute CMAC output
                    local_cmac_output[i] = local_cmac_output[i] + (self.input_space[global_neighborhood_index] * weight)

            # Accumulate the errors
            cumulative_error = cumulative_error + abs(data_set_true_output[i] - local_cmac_output[i])

        return local_cmac_output, cumulative_error

    def execute(self):
        """ train and test data until global convergence is achieved
        :param: none
        :return: a tuple with training and testing accuracy
        """

        iterations = 0
        self.convergence_time = time.time()

        # Repeat Until Number Of Iterations Exceed Max Iterations Required For Convergence
        # And Break If Global Convergence Occurs
        while iterations < self.max_global_convergence_iterations:
            self.train()

            self.training_set_cmac_output, training_cumulative_error = self.test('TrainingData')
            self.training_error = training_cumulative_error / self.training_set_size

            self.testing_set_cmac_output, testing_cumulative_error = self.test('TestingData')
            self.testing_error = testing_cumulative_error / self.testing_set_size

            iterations = iterations + 1

            # If Testing Error Is Below Convergence Threshold,
            # Then Global Convergence Is Achieved Within The Specified Maximum Number Of Iterations
            if self.testing_error <= self.global_convergence_threshold:
                self.convergence = True
                break

            # Calculate convergence time
        self.convergence_time = time.time() - self.convergence_time

        return (1 - self.training_error) * 100, (1 - self.testing_error) * 100

    def plot_graphs(self):
        """ plot graphs for generalization factors
        :param: none
        :return: nothing
        """

        sorted_training_set_input = [x for (y, x) in
                                     sorted(zip(self.training_set_global_indices, self.training_set_input))]
        sorted_training_set_output = [x for (y, x) in
                                      sorted(zip(self.training_set_global_indices, self.training_set_cmac_output))]
        sorted_testing_set_input = [x for (y, x) in
                                    sorted(zip(self.testing_set_global_indices, self.testing_set_input))]
        sorted_testing_set_output = [x for (y, x) in
                                     sorted(zip(self.testing_set_global_indices, self.testing_set_cmac_output))]

        plt.figure(figsize=(20, 11))
        plt.suptitle(str(self.cmac_type) + ' CMAC With Fixed Generalization Factor and Input Space Size')
        plt.subplot(221)
        plt.plot(self.training_set_input, self.training_set_output, 'bo', label='True Output')
        plt.plot(sorted_training_set_input, sorted_training_set_output, 'r', label='CMAC Output')
        plt.title('Generalization_Factor = ' + str(self.generalization_factor) + ' \nTraining Data')
        plt.ylabel('Output')
        plt.xlabel('Input')
        plt.legend(loc='upper right', shadow=True)
        plt.ylim((constants.minimum_output_value, constants.maximum_output_value))

        plt.subplot(222)
        plt.plot(sorted_training_set_input,
                 (1 - abs(np.array(self.training_set_output) - np.array(self.training_set_cmac_output))) * 100, 'r')
        plt.ylabel('Training Accuracy')
        plt.xlabel('Input')
        plt.title('\nTraining Accuracy vs Input' +
                  '\nCumulative Training Accuracy = ' + str((1 - self.training_error) * 100))
        plt.subplot(223)
        plt.plot(self.testing_set_input, self.testing_set_output, 'bo', label='True Output')
        plt.plot(sorted_testing_set_input, sorted_testing_set_output, 'r', label='CMAC Output')
        plt.title('Generalization_Factor = ' + str(self.generalization_factor) + ' \nTest Data')
        plt.ylabel('Output')
        plt.xlabel('Input')
        plt.legend(loc='upper right', shadow=True)
        plt.ylim((constants.minimum_output_value, constants.maximum_output_value))
        plt.subplot(224)
        plt.plot(sorted_testing_set_input,
                 (1 - abs(np.array(self.testing_set_output) - np.array(self.testing_set_cmac_output))) * 100, 'r')
        plt.ylabel('Testing Accuracy')
        plt.xlabel('Input')
        plt.title('\nTesting Accuracy vs Input' +
                  '\nCumulative Testing Accuracy = ' + str((1 - self.testing_error) * 100))
        plt.show()


def find_nearest(array, value):
    """ find the index of the element closest to the given value in the array
    :param array: array to search closest element
    :param value: value to search the closest element
    :return: index of the closest element
    """
    idx = (np.abs(np.array(array) - value)).argmin()
    return idx
