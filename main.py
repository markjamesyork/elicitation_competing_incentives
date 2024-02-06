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

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

def simulate_event_probability(a, b):
    return beta.rvs(a, b)

def simulate_agent_predictions(n_agents, true_prob, rec_accuracy_a_plus_b):
    return beta.rvs(rec_accuracy_a_plus_b * true_prob, rec_accuracy_a_plus_b * (1 - true_prob), size=n_agents)

def calculate_average_report(reports):
    return np.mean(reports)

def determine_action_probability(avg, min_x, L): #check whether this qualifies as a bounded linear decision function defined in Defn. 8.
    # Sets allocation probability according to a linear piecewise function of avg, with a minimum probability of 0 and maximum of up to 1.
    if avg < min_x:
        return 0.
    elif min_x < avg < 1 / L + min_x:
        return (avg - min_x) * L
    else:
        return 1.

def calculate_expected_utility(reports, true_prob, action_prob, scale, c):
    # We use the expected beta quadratic scoring rule given the agents' reports and the true probabilities
    scores = scale * (2 * true_prob * reports + 1 - true_prob - reports**2) #Expected beta quadratic scores given reports and the true probability
    # Utility includes both the score and a constant times the action probability
    utilities = scores + c * action_prob
    return scores, utilities

def run_simulation(n_agents, n_runs, p, a, b, scale, L, c):
    avg_payout = []
    worst_case_payout = []
    squared_error = []
    
    for _ in range(n_runs):
        true_prob = simulate_event_probability(a, b)
        predictions = simulate_agent_predictions(n_agents, true_prob, rec_accuracy_a_plus_b)
        reports = predictions.copy()
        avg_report = calculate_average_report(reports)
        action_prob = determine_action_probability(avg_report, min_x, L)
        scores, utilities = calculate_expected_utility(reports, true_prob, action_prob, scale, c)
        
        avg_payout.append(np.mean(scores))
        worst_case_payout.append(np.max(scores))
        squared_error.append((avg_report - true_prob)**2)
    
    return np.mean(avg_payout), np.max(worst_case_payout), np.sqrt(np.mean(squared_error))

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

for _ in range(len(n_agents_vector)):
    avg_payout, worst_case_payout, rmse = run_simulation(n_agents_vector[_], n_runs, p, a, b, scale, L, c)
    avg_payout_vector.append(avg_payout)
    rmse_vector.append(rmse)

# Plotting the RMSE values
plt.figure(figsize=(10, 6))  # Optional: specifies the figure size
plt.plot(n_agents_vector, rmse_vector, marker='o')  # Plot with markers at each data point
plt.title('Root Mean Squared Error of Avg. Report vs. True Probability')  # Title of the plot
plt.xlabel('Number of Agents')  # Label for the x-axis
plt.ylabel('RMSE')  # Label for the y-axis
plt.grid(True)  # Optional: adds a grid for easier reading
plt.show()

#print(f"Average Payout: {avg_payout}")
#print(f"Worst-Case Payout: {worst_case_payout}")