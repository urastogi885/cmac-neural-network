"""
define constant variables that will be used as standards for the entire project
"""

# Define generalization to ignore any generalization effects
plot_generalization_factor = 0
# Define maximum no. of points in the dataset
plot_input_space_size = 100

max_generalization_factor = 35
min_generalization_factor = 1

max_input_space_size = 700
min_input_space_size = 100
input_space_step_size = 100
input_space_split_data_size = int((max_input_space_size - min_input_space_size) / float(input_space_step_size))
# Define max and min values of the function
minimum_output_value = -1.0
maximum_output_value = 1.0
