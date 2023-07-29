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

'''
# -----------------------------------------------------------
def initial_sol_fun(T, product_size, item_size, upper_bound, lower_bound = 0):
	# return np.random.randint(lower_bound, upper_bound, size=(T, item_size))
	return np.ones((T, item_size))*upper_bound


def normalization(T, product_size, item_size, upper_bound, sample_size = 50):
	sample_list = []
	for i in range(sample_size):
		sample_list.append(ros.replications_of_sim(T, product_size, item_size, initial_sol(T, product_size, item_size, upper_bound)))
	# sample_mean = np.mean(sample_list)
	sample_std = np.std(sample_list)
	return sample_std
'''

# return a integer of the optimization solution (weighted cost) 
def spsa_fun(T, product_size, item_size, bom, Max_measurements, upper_bound, initial_fit, initial_sol, lower_bound = 800):
	'''
		Input: initial solution of arrival
		opt_count_limit: # iterations for the SPSA algorithm
	'''
	# -----------------------------------------------------------
	# index setting (1)
	alpha = .602 # .602 from (Spall, 1998)
	gamma = .167 # .167 default
	a = .00101 # .101 found empirically using HyperOpt
	A = .193 # .193 default
	c = 1 # .0277 default # T * product_size *item_size
	u = initial_sol.reshape(T,item_size).astype('int')
	d_k = 100
	# sample_std = normalization(T, product_size, item_size, upper_bound)
	# print(sample_mean)
	# scalar_u = ros.replications_of_sim(T, product_size, item_size, u)
	# print(u)

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
		# print(c_k*delta_k[0][0])

		# Step 3: Function evaluations
		thetaplus = np.where(u + c_k*delta_k < lower_bound, lower_bound, u + c_k*delta_k)
		thetaplus = np.where(thetaplus > upper_bound, upper_bound, thetaplus).astype('int')
		y_thetaplus = ros.replications_of_sim(T, product_size, item_size, bom, thetaplus)
		
		thetaminus = np.where(u - c_k*delta_k < lower_bound, lower_bound, u - c_k*delta_k)
		thetaminus = np.where(thetaminus > upper_bound, upper_bound, thetaminus).astype('int')
		y_thetaminus = ros.replications_of_sim(T, product_size, item_size, bom, thetaminus)

		# print(thetaplus.min(), thetaplus.max())

		# Step 4: Gradient approximation
		g_k = np.dot((y_thetaplus - y_thetaminus) / (2.0*c_k*d_k**2), delta_k)
		# print(c_k*delta_k[0][0], a_k * g_k[0][0])

		# Step 5: Update u estimate
		# u = np.asarray(np.where((u-a_k*g_k<0, 0, u-a_k*g_k) & (u-a_k*g_k>64, 64, u-a_k*g_k)), dtype = 'int')
		u = np.where(u - a_k * g_k < lower_bound, lower_bound, u - a_k * g_k)
		u = np.where(u > upper_bound, upper_bound, u).astype('int')
  

		obj_list = [ros.replications_of_sim(T, product_size, item_size, bom, u), y_thetaplus, y_thetaminus]
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
	# print(len(best_obj_list),len(spsa_ans_list))
	
	for i in range(len(best_obj_list)-1):
		for k in range(spsa_measurment_per_iteration): spsa_ans_list.append(best_obj_list[i+1])
	# -----------------------------------------------------------------------------------

	return best_obj, spsa_ans_list[0:Max_measurements+1], best_solution


'''# test
if __name__ == '__main__' :
	print("go ...")
	T, product_size, item_size = (5, 4, 3)
	import time
	time.clock = time.time
	
	# spsa algorithm
	spsa_measurements_per_iteration = 3
	Max_measurements = 4500
	upper_bound = product_size*20
	
	# update initial solution
	initial_sol = np.ones((1, T*item_size))*upper_bound
	initial_fit = ros.replications_of_sim(T, product_size, item_size, initial_sol.reshape(T,item_size))
	print(f'initial fitness = {initial_fit}')
 
	tic = time.clock()
	best_spsa, bl_spsa, ans_spsa = spsa_fun(T, product_size, item_size, Max_measurements, upper_bound, initial_fit, initial_sol)
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
