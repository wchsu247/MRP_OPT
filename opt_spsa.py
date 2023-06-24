import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from numpy.random import randint
from numpy.random import rand
import replications_of_sim as ros
import sys
MAX_INT=sys.maxsize

# index setting
m_limit = 10000
total_count = 0
c = []
spm_best = [0,9999]


# -----------------------------------------------------------
def initial_sol(T, product_size, item_size, lower_bound = 0, upper_bound = 64):
	return np.random.randint(lower_bound, upper_bound, size=(T, item_size))


# return a integer of the optimization solution (weighted cost) 
def spsa_fun(T, product_size, item_size, opt_count_limit = 10):
	'''
		Input: initial solution of arrival
		opt_count_limit: # iterations for the SPSA algorithm
	'''

	# -----------------------------------------------------------
	# index setting (1)
	alpha = 0.602 # from (Spall, 1998)
	gamma = 0.167
	a = .101 # found empirically using HyperOpt
	A = .193
	c = 1.3
	u = initial_sol(T, product_size, item_size)
	# print(u)

	best_obj = MAX_INT
	best_obj_list = []
    
	for k in range(opt_count_limit):

		print(">> Case %d" %(k+1))
		# index setting (2)

		# a_k = 1 / (k+1)
		a_k = a / (A + k + 1)**alpha 	# a_k = 1 / (k+1)
		c_k = c / (k + 1)**gamma		# c_k = 1 / ((1 + k) ** (1 / 6))

		# Step 2: Generation of simultaneous perturbation vector
		# choose each component from a bernoulli +-1 distribution with
		# probability of .5 for each +-1 outcome.
		delta_k = np.random.choice([-1,1], size=(T, item_size), p=[.5, .5])

		# Step 3: Function evaluations
		thetaplus = np.asarray(u + c_k*delta_k, dtype = 'int')
		thetaminus = np.asarray(u - c_k*delta_k, dtype = 'int')
		y_thetaplus = ros.replications_of_sim(T, product_size, item_size, thetaplus)
		y_thetaminus = ros.replications_of_sim(T, product_size, item_size, thetaminus)

		# Step 4: Gradient approximation
		g_k = np.dot((y_thetaplus - y_thetaminus) / (2.0*c_k), delta_k)
		# print(g_k)

		# Step 5: Update u estimate
		u = np.asarray(np.where(u-a_k*g_k<0, 0, u-a_k*g_k), dtype = 'int')
		obj_value = ros.replications_of_sim(T, product_size, item_size, u)

		# Step 6: Check for convergence
		if obj_value < best_obj:
			best_obj = obj_value
		best_obj_list.append(best_obj)

	print("The best fitness:   %d" %(best_obj))
	# -----------------------------------------------------------------------------------
	'''# visualization
	plt.figure(figsize = (15,8))
	plt.xlabel("Iteration",fontsize = 15)
	plt.ylabel("Fitness",fontsize = 15)

	plt.plot(best_obj_list,linewidth = 2, label = "Best fitness convergence", color = 'b')
	plt.legend()
	plt.show()
	'''
	return best_obj, best_obj_list

''' test
if __name__ == '__main__' :

	print("go ...")
	T, product_size, item_size = (5, 4, 3)
	spsa_fun(T, product_size, item_size)
'''

