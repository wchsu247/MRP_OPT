import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randint
from numpy.random import rand
import replications_of_sim as ros
import sys
MAX_INT=sys.maxsize
import warnings
warnings.filterwarnings('ignore')

# return a integer of the optimization solution (weighted cost) 
def spsa_fun(T, product_size, item_size, bom, Max_measurements, upper_bound, initial_fit, initial_sol, lower_bound = 0):
    
	# random count
	rc = 0

	'''
		Input: initial solution of arrival
		opt_count_limit: # iterations for the SPSA algorithm
	'''
	# -----------------------------------------------------------
	# index setting (1)
	alpha = .602 # .602 from (Spall, 1998)
	gamma = .167 # .167 default
	a = 101 # .101 found empirically using HyperOpt
	A = .193 # .193 default
	c = 2.77 # .0277 default # T * product_size *item_size
	u = initial_sol.reshape(T,item_size).astype('int')
	d_k = 10


	best_solution = u
	best_obj = initial_fit
	best_obj_list = [initial_fit]
 
	spsa_measurment_per_iteration = 3
	k = 0
	measurement = 0
	while measurement < Max_measurements:

		# print(">> Case %d" %(k+1))
		# index setting (2)

		a_k = a / (A + k + 1)**alpha 	# a_k = 1 / (k+1)
		c_k = c / (k + 1)**gamma		# c_k = 1 / ((1 + k) ** (1 / 6))

		# Step 2: Generation of simultaneous perturbation vector
		# choose each component from a bernoulli +-1 distribution with
		# probability of .5 for each +-1 outcome.
		delta_k = np.random.choice([-d_k,d_k], size=(T, item_size), p=[.5, .5])

		# Step 3: Function evaluations
		thetaplus = np.where(u + c_k*delta_k < lower_bound, lower_bound, u + c_k*delta_k)
		thetaplus = np.where(thetaplus > upper_bound, upper_bound, thetaplus).astype('int')
		y_thetaplus, random_count = ros.replications_of_sim(T, product_size, item_size, bom, thetaplus)
		rc += random_count
		
		thetaminus = np.where(u - c_k*delta_k < lower_bound, lower_bound, u - c_k*delta_k)
		thetaminus = np.where(thetaminus > upper_bound, upper_bound, thetaminus).astype('int')
		y_thetaminus, random_count = ros.replications_of_sim(T, product_size, item_size, bom, thetaminus)
		rc += random_count

		# Step 4: Gradient approximation
		g_k = np.dot((y_thetaplus - y_thetaminus) / (2.0*c_k*d_k**2), delta_k)
		# print(c_k*delta_k[0][0], a_k * g_k[0][0])

		# Step 5: Update u estimate
		u = np.where(u - a_k * g_k < lower_bound, lower_bound, u - a_k * g_k)
		u = np.where(u > upper_bound, upper_bound, u).astype('int')
  
		fit, random_count = ros.replications_of_sim(T, product_size, item_size, bom, u)
		rc += random_count
		obj_list = [fit, y_thetaplus, y_thetaminus]
		sol_list = [u, thetaplus, thetaminus]
		obj_value = min(obj_list)
		obj_solution = sol_list[obj_list.index(min(obj_list))]

		# Step 6: Check for convergence
		if obj_value < best_obj:
			best_obj = obj_value
			best_solution = obj_solution
		best_obj_list.append(best_obj)
		measurement += spsa_measurment_per_iteration
		k += 1

	print("The best fitness:   %d" %(best_obj))
	spsa_ans_list = [initial_fit]
	
	for i in range(len(best_obj_list)-1):
		for k in range(spsa_measurment_per_iteration): spsa_ans_list.append(best_obj_list[i+1])
	# -----------------------------------------------------------------------------------

	return best_obj, spsa_ans_list[0:Max_measurements+1], best_solution, rc

'''
# test
if __name__ == '__main__' :
	print("go ...")
	T, product_size, item_size = (5, 4, 3)
	bom = np.random.randint(2, size=(product_size, item_size))
	import time
	time.clock = time.time
	
	# spsa algorithm
	spsa_measurements_per_iteration = 3
	Max_measurements = 9000
	upper_bound = product_size*40
	
	# update initial solution
	initial_sol = np.ones((1, T*item_size))*upper_bound/2
	print(initial_sol)
	# initial_fit = ros.replications_of_sim(T, product_size, item_size, initial_sol.reshape(T,item_size))
	initial_fit = 1222.25
	print(f'initial fitness = {initial_fit}')
 
	tic = time.clock()
	best_spsa, bl_spsa, ans_spsa = spsa_fun(T, product_size, item_size, bom, Max_measurements, upper_bound, initial_fit, initial_sol)
	time_spsa = time.clock()-tic
	print(">> SPSA in %.5f sec." %time_spsa)
	print(len(bl_spsa))
 
	print(ans_spsa)
	# visualization
	plt.figure(figsize = (15,8))
	plt.xlabel("# Measurements",fontsize = 15)
	plt.ylabel("Fitness",fontsize = 15)

	plt.plot(bl_spsa, linewidth = 2, label = "Best fitness convergence", color = 'b')
	plt.legend()
	plt.show()
'''
