# -*- coding: utf-8 -*-
"""
Created on Sun Feb 4 2024

@author: Mark York
"""

'''
This simulation implements an elicitation system which gathers predictions about the
probability of an uncertain binary future event occuring. The system pays agents with
a scaled quadratic score and sets the probability for taking an action with a piecewise
linear function of the average prediction. Agents derive a utility of their expected
quadratic score given the true event probability and their outside incentive c times
the action probability.
'''
from analytical_results import bounded_linear_target_decision_function, required_budget
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import beta
import pandas as pd
import datetime as dt
from concurrent.futures import ProcessPoolExecutor, as_completed


def simulate_event_probability(a, b):
    return beta.rvs(a, b)

def simulate_agent_predictions(n_agents, true_prob, rec_accuracy_a_plus_b):
    return beta.rvs(rec_accuracy_a_plus_b * true_prob, rec_accuracy_a_plus_b * (1 - true_prob), size=n_agents)

def calculate_aggregated_report(reports, t, alpha):
    clipped_reports = t + np.clip(reports - t, -alpha, alpha)
    return np.mean(clipped_reports)

def calculate_exp_scores(reports, true_prob, budget, n):
    #We use the adaptive payment mechanism from our paper, calibrated on the bounded linear target decision function
    expected_scores = (budget / (3 * n)) * (true_prob * reports - np.square(reports) / 2 + 1)
    return np.mean(expected_scores)

def run_simulation(n, f_min, n_runs, mean_repayment_prob, borrower_a_plus_b, rec_a_plus_b, t):
    '''This function simulates a recommendation, lending and repayment system
    n = number of recommenders
    f_min = minimum allocation
    n_runs = number of times to repeat the simulation
    mean_repayemnt_probability = mean repayemnt probabilty across all borrowers
    borrower_a_plus_b = the sum of the a and b parameters of the beta distribution from which borrower repayment probabilties are drawn. A higher value means a tighter concentration around the mean.
    rec_a_plus_b = the sum of the a and b parameters of the beta distribution of recommender beliefs.
    t = the center of the sloped part of the allocation function along the x axis.
    '''

    print('t = %f' % t)

    #0 Initial Setup
    exp_recommender_payout = []
    cash_deployed = []
    exp_cash_repaid = []

    #1 Parameter Settings
    #if t == None: t = .5
    alpha = (1-f_min)/2
    if n <= 25: l = np.sqrt(n) / .5
    else: l = 10. 
    budget = required_budget(f_min, n, l, alpha, 1)
    interest_rate = .2 #Interest that borrowers repay in the case that they do repay (repayemnts are binary)

    #2 Simulation Loop
    for _ in range(n_runs):
        #2.1 Probability draws, reports and action
        true_prob = simulate_event_probability(mean_repayment_prob*borrower_a_plus_b, (1- mean_repayment_prob)*borrower_a_plus_b)
        reports = simulate_agent_predictions(n, true_prob, rec_a_plus_b) #assumes that recommenders report predictions truthfully
        aggregated_report = calculate_aggregated_report(reports, t, alpha)
        action = bounded_linear_target_decision_function(f_min, n, aggregated_report, t, alpha, l)

        #2.1 Direct Results
        exp_recommender_payout += [n*calculate_exp_scores(reports, true_prob, budget, n)]
        cash_deployed += [action]
        exp_cash_repaid += [action * (1+interest_rate) * true_prob]
        
        '''
        print('')
        print('true_prob', true_prob)
        #print('reports', reports)
        print('aggregated_report', aggregated_report)
        print('action ', action )
        print('exp_recommender_payout', exp_recommender_payout[-1])
        print('cash_deployed', cash_deployed[-1])
        print('exp_cash_repaid', exp_cash_repaid[-1])
        '''

    #3 Calculated Results
    total_cost = np.mean(exp_recommender_payout) + np.mean(cash_deployed)
    profit = np.mean(exp_cash_repaid) - total_cost
    ROI = profit / np.mean(cash_deployed) #We consider ROI only on cash deployed and not on recommender payout, as recommender payout comes straight from revenue when loans are repaid.
    
    return np.mean(exp_recommender_payout), np.mean(cash_deployed), total_cost, profit, ROI, np.mean(exp_cash_repaid)


def chart_accuracy():
    # Parameters setup
    n_agents = 50
    n_runs = 10000
    p = 0.6  # Mean of the true event probability; close to Uganda lending data
    repayment_prob_a_plus_b = 10   # Sum of coefficients a and b
    a = repayment_prob_a_plus_b * p
    b = repayment_prob_a_plus_b * (1 - p)
    rec_accuracy_a_plus_b = 18 #Based on Uganda recommender distribution, assuming centered on mean repayment 
    scale = 1 # Scale of the quadratic score
    L = 2    # Maximum slope of the piecewise linear function
    c = 0.5  # Constant for action probability
    min_x = .5 #average rating below which the system deterministically makes a decision of 0.

    # Simulation
    n_agents_vector = list(np.arange(50)+1)

    avg_payout_vector = []
    rmse_vector = []
    analytical_vector = []

    for _ in range(len(n_agents_vector)):
        avg_payout, worst_case_payout, rmse = run_simulation(n_agents_vector[_], n_runs, p, a, b, scale, L, c)
        avg_payout_vector.append(avg_payout)
        rmse_vector.append(rmse)
        analytical_vector.append(.08191780219 / np.sqrt(n_agents_vector[_]))

    # Plotting the RMSE values
    plt.figure(figsize=(10, 6))  # Optional: specifies the figure size
    plt.plot(n_agents_vector, rmse_vector, marker='o')  # Plot with markers at each data point
    plt.plot(n_agents_vector, analytical_vector, marker='o')  # Plot with markers at each data point
    plt.title('Root Mean Squared Error of Avg. Report vs. True Probability')  # Title of the plot
    plt.xlabel('Number of Agents')  # Label for the x-axis
    plt.ylabel('RMSE')  # Label for the y-axis
    plt.legend(['Simulated', 'Analytical with $\zeta=.85$'])
    plt.grid(True)  # Optional: adds a grid for easier reading
    plt.show()


def run_simulation_wrapper(args):
    '''Function to run the simulation in parallel'''
    i, prob, j, t_val, n, n_runs, mean_repayment_prob, borrower_a_plus_b, rec_a_plus_b = args
    result = run_simulation(n, prob, n_runs, mean_repayment_prob, borrower_a_plus_b, rec_a_plus_b, t_val)
    return i, j, result  # Assuming profit is the fourth element in the result


def main():
    #Simulation running loop
    n = 1000000
    f_min = .05
    n_runs = 50
    mean_repayment_prob = .85
    borrower_a_plus_b = 10
    rec_a_plus_b = 18
    parallel = True

    # Generate grids for f_min and t
    min_allocation_probabilities = np.arange(.02, 1., 0.02)
    t_values = np.arange(.5, 1.0, .02)

    # Prepare matrix to hold the required budget values
    profit_matrix = np.zeros((len(min_allocation_probabilities), len(t_values)))
    mean_rec_payout_matrix = profit_matrix.copy()
    mean_cash_deployed_matrix = profit_matrix.copy()
    ROI_matrix = profit_matrix.copy()
    exp_cash_repaid_matrix = profit_matrix.copy()

    start = dt.datetime.now()

    if parallel == False:
        # Calculate required budget for each combination of min_allocation_probability and L
        for i, prob in enumerate(min_allocation_probabilities):
            print('f_min = %f' % prob)
            for j, t_val in enumerate(t_values):
                #print('t = %f' % t_val)
                result = run_simulation(n, prob, n_runs, mean_repayment_prob, borrower_a_plus_b, rec_a_plus_b, t_val) #result = [mean_rec_payout, mean_cash_deployed, total_cost, profit, ROI, exp_cash_repaid]
                mean_rec_payout_matrix[i, j] = result[0]
                mean_cash_deployed_matrix[i, j] = result[1]
                profit_matrix[i, j] = result[3]
                ROI_matrix[i, j] = result[4]
                exp_cash_repaid_matrix[i, j] = result[5]

    elif parallel == True:
        # Create a list of all parameter combinations
        task_args = [
                    (i, prob, j, t_val, n, n_runs, mean_repayment_prob, borrower_a_plus_b, rec_a_plus_b)
                    for i, prob in enumerate(min_allocation_probabilities)
                    for j, t_val in enumerate(t_values)
                    ]

        with ProcessPoolExecutor() as executor:
            future_to_params = {executor.submit(run_simulation_wrapper, args): args for args in task_args}

            for future in as_completed(future_to_params):
                i, j, result = future.result()
                mean_rec_payout_matrix[i, j] = result[0]
                mean_cash_deployed_matrix[i, j] = result[1]
                profit_matrix[i, j] = result[3]
                ROI_matrix[i, j] = result[4]
                exp_cash_repaid_matrix[i, j] = result[5]

    print('Time elapsed: ', dt.datetime.now() - start)

    # Writing each array to its own CSV file
    names = ['mean_rec_payout', 'mean_cash_deployed', 'profit', 'ROI', 'exp_cash_repaid']
    arrays = [mean_rec_payout_matrix, mean_cash_deployed_matrix, profit_matrix, ROI_matrix, exp_cash_repaid_matrix]
    for array, name in zip(arrays, names):
        filename = f"csv_files/{name}_n{n}.csv"  # Constructs the file name
        pd.DataFrame(array).to_csv(filename, index=False)  # Writes the array to a CSV file
        print(f"Array {name} written to {filename}")  # Optional: prints confirmation


    # Plotting with logarithmic color scale
    variable = 'ROI'
    plt.figure(figsize=(10, 8))
    plt.imshow(locals()[variable + '_matrix'], cmap='viridis', aspect='auto', 
    extent=[t_values.min(), t_values.max(), 
                       min_allocation_probabilities.max(), min_allocation_probabilities.min()])
               #norm=LogNorm())  # Apply logarithmic normalization
    plt.colorbar(label=variable)
    plt.title('%s Versus $f_{min}$ and $t$' % variable)
    plt.xlabel('t')
    plt.ylabel('Min Allocation Probability $f_{min}$')
    plt.gca().invert_yaxis()  # Invert y-axis to have 0 start at the bottom
    plt.savefig('charts/%s_n%d.png' % (variable, n))
    plt.show()
    plt.close()

# This check ensures that the following code runs only when the script is executed directly,
# not when it's imported as a module in another script
if __name__ == '__main__':
    main()





'''
#Results Printing Statements
mean_rec_payout, mean_cash_deployed, total_cost, profit, ROI, exp_cash_repaid = run_simulation(n, f_min, n_runs, mean_repayment_prob, borrower_a_plus_b, rec_a_plus_b, t)
print('mean_rec_payout', np.round(mean_rec_payout,2 ))
print('mean_cash_deployed', np.round(mean_cash_deployed,2 ))
print('total_cost', np.round(total_cost,2 ))
print('profit', np.round(profit,2 ))
print('ROI', np.round(ROI,2 ))
print('exp_cash_repaid', exp_cash_repaid)
'''