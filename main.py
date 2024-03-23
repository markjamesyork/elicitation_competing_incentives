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

    #3 Calculated Results
    total_cost = np.mean(exp_recommender_payout) + np.mean(cash_deployed)
    profit = np.mean(exp_cash_repaid) - total_cost
    ROI = profit / np.mean(cash_deployed) #We consider ROI only on cash deployed and not on recommender payout, as recommender payout comes straight from revenue when loans are repaid.
    
    return np.mean(exp_recommender_payout), np.mean(cash_deployed), total_cost, profit, ROI, np.mean(exp_cash_repaid)


def run_simulation_wrapper(args):
    '''Function to run the simulation in parallel'''
    i, prob, j, t_val, n, n_runs, mean_repayment_prob, borrower_a_plus_b, rec_a_plus_b = args
    result = run_simulation(n, prob, n_runs, mean_repayment_prob, borrower_a_plus_b, rec_a_plus_b, t_val)
    return i, j, n, result  # Return n as well to track which n the result corresponds to

def main():
    # Calls the simulation for various parameter settings in parallel
    n_runs = 5000
    mean_repayment_prob = .85
    borrower_a_plus_b = 10
    rec_a_plus_b = 18
    n_values = [10**i for i in range(7)]

    # Generate grids for f_min and t
    min_allocation_probabilities = np.arange(.02, 1., 0.02)
    t_values = np.arange(.5, 1.0, .02)

    # Structures for capturing the best f_min and t values
    best_profit_values = []
    best_roi_values = []

    for n in n_values:
        profit_matrix = np.zeros((len(min_allocation_probabilities), len(t_values)))
        mean_rec_payout_matrix = profit_matrix.copy()
        mean_cash_deployed_matrix = profit_matrix.copy()
        ROI_matrix = profit_matrix.copy()
        exp_cash_repaid_matrix = profit_matrix.copy()


        start = dt.datetime.now()

        # Create a list of all parameter combinations
        task_args = [
            (i, prob, j, t_val, n, n_runs, mean_repayment_prob, borrower_a_plus_b, rec_a_plus_b)
            for i, prob in enumerate(min_allocation_probabilities)
            for j, t_val in enumerate(t_values)
        ]

        with ProcessPoolExecutor() as executor:
            future_to_params = {executor.submit(run_simulation_wrapper, args): args for args in task_args}

            for future in as_completed(future_to_params):
                i, j, n_current, result = future.result()
                mean_rec_payout_matrix[i, j] = result[0]
                mean_cash_deployed_matrix[i, j] = result[1]
                profit_matrix[i, j] = result[3]
                ROI_matrix[i, j] = result[4]
                exp_cash_repaid_matrix[i, j] = result[5]


        # Record the highest ROI and profit and their corresponding f_min and t
        max_profit_index = np.unravel_index(np.argmax(profit_matrix, axis=None), profit_matrix.shape)
        max_roi_index = np.unravel_index(np.argmax(ROI_matrix, axis=None), ROI_matrix.shape)
        best_profit_values.append((n, min_allocation_probabilities[max_profit_index[0]], t_values[max_profit_index[1]], np.max(profit_matrix)))
        best_roi_values.append((n, min_allocation_probabilities[max_roi_index[0]], t_values[max_roi_index[1]], np.max(ROI_matrix)))


        # Writing each array to its own CSV file
        names = ['mean_rec_payout', 'mean_cash_deployed', 'profit', 'ROI', 'exp_cash_repaid']
        arrays = [mean_rec_payout_matrix, mean_cash_deployed_matrix, profit_matrix, ROI_matrix, exp_cash_repaid_matrix]
        for array, name in zip(arrays, names):
            filename = f"csv_files/{name}_n{n}.csv"  # Constructs the file name
            pd.DataFrame(array).to_csv(filename, index=False)  # Writes the array to a CSV file
            print(f"Array {name} written to {filename}")  # Optional: prints confirmation


        # Plotting variables with logarithmic color scale
        variables = ['ROI', 'profit']
        for variable in variables:
            plt.figure(figsize=(10, 8))
            plt.imshow(locals()[variable + '_matrix'], cmap='viridis', aspect='auto', 
            extent=[t_values.min(), t_values.max(), 
                               min_allocation_probabilities.max(), min_allocation_probabilities.min()])
                       #norm=LogNorm())  # Apply logarithmic normalization
            plt.colorbar(label=variable)
            plt.title('%s Versus $f_{min}$ and $t$ when n = %d' % (variable, n))
            plt.xlabel('t')
            plt.ylabel('Min Allocation Probability $f_{min}$')
            plt.gca().invert_yaxis()  # Invert y-axis to have 0 start at the bottom
            plt.savefig('charts/%s_n%d.png' % (variable, n))
            #plt.show()
            plt.close()

        print(f'n={n}: Time elapsed: ', dt.datetime.now() - start)

        #Printing best profit fmin, t and value
        print('max_profit_t', min_allocation_probabilities[max_profit_index[0]])
        print('max_profit_fmin', t_values[max_profit_index[1]])
        print('max_roi_t', min_allocation_probabilities[max_roi_index[0]])
        print('max_roi_fmin', t_values[max_roi_index[1]])
        print('best_profit_values', best_profit_values)
        print('best_roi_values', best_roi_values)


    # Convert results to arrays for plotting
    best_profit_array = np.array(best_profit_values)
    best_roi_array = np.array(best_roi_values)

    # Plot Best Profit and ROI vs. n
    for label, data in zip(['Profit', 'ROI'], [best_profit_array, best_roi_array]):
        plt.figure()
        plt.xscale('log')
        plt.plot(data[:, 0], data[:, 3], '-o', label=f'Max {label}')
        plt.xlabel('n (log scale)')
        plt.ylabel(f'Max {label}')
        plt.title(f'Max {label} vs n (log scale)')
        plt.legend()
        plt.savefig(f'Max_{label}_vs_n.pdf')
        plt.close()


if __name__ == '__main__':
    main()


###Out-of-use functions###
##########################

def chart_accuracy():
    # Parameters setup
    n_agents = 50
    n_runs = 10000
    p = 0.85  # Mean of the true event probability; close to Uganda lending data
    repayment_prob_a_plus_b = 9.8959   # Sum of coefficients a and b
    a = repayment_prob_a_plus_b * p
    b = repayment_prob_a_plus_b * (1 - p)
    rec_accuracy_a_plus_b = 12.7418 #Based on Uganda recommender distribution, assuming centered on mean repayment 
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

        


