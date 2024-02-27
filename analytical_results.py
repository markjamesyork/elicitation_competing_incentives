#This script creates charts from the analytical results of our paper 
#"Belief Elicitation from Agents with Competing Incentives."

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def required_budget(f_min, n, l, alpha, briber_incentive):
	return briber_incentive * 12 * alpha * l**2 / (f_min * n)

def plot_heatmap_fmin():
    # Generate grids for Min Allocation Probability and L
    min_allocation_probabilities = np.arange(0, 1.01, 0.01)
    n_values = np.arange(1, 100001, 1)
    alpha_values = [(1-i)/2 for i in min_allocation_probabilities]
    l_values = [min(2*i**.5, 10) for i in n_values]
    briber_incentive = 1.

    # Prepare matrix to hold the required budget values
    budget_matrix = np.zeros((len(min_allocation_probabilities), len(n_values)))
    
    # Calculate required budget for each combination of min_allocation_probability and L
    for i, prob in enumerate(min_allocation_probabilities):
        for j, n in enumerate(n_values):
            budget_matrix[i, j] = required_budget(prob, n, l_values[j], alpha_values[i], briber_incentive)
    
    # Plotting with logarithmic color scale
    plt.figure(figsize=(10, 8))
    plt.imshow(budget_matrix, cmap='viridis', aspect='auto', 
	extent=[n_values.min(), n_values.max(), 
                       min_allocation_probabilities.max(), min_allocation_probabilities.min()],
               norm=LogNorm())  # Apply logarithmic normalization
    plt.colorbar(label='Budget Required')
    plt.title('Heat Map of Budget Required (Logarithmic Scale)')
    plt.xlabel('n')
    plt.ylabel('Min Allocation Probability $f_{min}$')
    plt.gca().invert_yaxis()  # Invert y-axis to have 0 start at the bottom
    plt.savefig('required_budget_log_scale.png')  # Save with a new name to avoid overwriting the linear scale image
    plt.show()
    plt.close()
    

    # Extract the rows for min_allocation_probability = 0.1, .5
    # Ploting budget required vs L for min_allocation_probability = 0.1
    budget_at_0_1_prob = budget_matrix[10, :]
    budget_at_0_5_prob = budget_matrix[50, :]
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, budget_at_0_1_prob, linestyle='-')
    plt.plot(n_values, budget_at_0_5_prob, linestyle='-')
    plt.title('Budget Required vs. n')
    plt.xlabel('n')
    plt.ylabel('Budget Required')
    plt.legend(['$f_{min} = .1$', '$f_{min} = .5$'])
    plt.grid(True)

    # Set both axes to logarithmic scale
    plt.xscale('log')
    plt.yscale('log')

    plt.savefig('required_budget_2d.png')  # Save with a new name to avoid overwriting
    plt.show()
    # Extract the row for min_allocation_probability = 0.1
    # Since arrays are 0-indexed and the step is 0.01, 0.1 corresponds to index 10



def bounded_linear_target_decision_function(f_min, n, r, t=None, alpha=None, l=None):
    #This function takes an average report r and number of recommenders m,
    #then calculates the allocation based on these reports

    #0 Parameter settings
    f_bar = (1+f_min)/2
    epsilon = (1-f_min)/2
    if t==None: t = .5
    if alpha==None: alpha = epsilon
    if l==None:
        if n <= 25: l = np.sqrt(n) / .5
        else: l = 10. 

    #1 Allocation calculation
    allocation = f_bar + np.clip(l*(r-t),-epsilon, epsilon) #r is the average of the reports AFTER they have been clipped within alpha of t
    '''
    print('f_min', f_min)
    print('n', n)
    print('t', t)
    print('alpha', alpha)
    print('l', l)
    '''
    return allocation

def plot_decision_function_results():
    n_values = [1, 25]
    fmin_values = [0, 0.1, 0.5]
    r_values = np.arange(0, 1.01, 0.001)  # From 0 to 1, inclusive, with step 0.01
    plt.figure(figsize=(12, 8))
    
    # Nested loops to calculate and plot results for each combination of n and fmin
    for n in n_values:
        for fmin in fmin_values:
            allocations = [bounded_linear_target_decision_function(fmin, n, r, t=.95) for r in r_values]
            plt.plot(r_values, allocations, label=f'n={n}, fmin={fmin}')
    
    # Setting the plot labels and legend
    plt.xlabel('r')
    plt.ylabel('Allocation')
    plt.title('Allocation vs. r for various values of n and f_min')
    plt.legend()
    plt.grid(True)
    plt.show()

# Execute the updated function to display the plot
#plot_decision_function_results()

#plot_heatmap_fmin()


