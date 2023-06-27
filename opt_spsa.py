import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randint
from numpy.random import rand
import replications_of_sim as ros
import sys
MAX_INT=sys.maxsize


# -----------------------------------------------------------
def initial_sol(T, product_size, item_size, lower_bound = 0, upper_bound = 1024):
	return np.random.randint(lower_bound, upper_bound, size=(T, item_size))

def normalization(T, product_size, item_size, sample_size = 50):
	sample_list = []
	for i in range(sample_size):
		sample_list.append(ros.replications_of_sim(T, product_size, item_size, initial_sol(T, product_size, item_size)))
	# sample_mean = np.mean(sample_list)
	sample_std = np.std(sample_list)
	return sample_std


# return a integer of the optimization solution (weighted cost) 
def spsa_fun(T, product_size, item_size, opt_count_limit, lower_bound = 0, upper_bound = 1024):
	'''
		Input: initial solution of arrival
		opt_count_limit: # iterations for the SPSA algorithm
	'''
	# -----------------------------------------------------------
	# index setting (1)
	alpha = 0 # .602 from (Spall, 1998)
	gamma = 0.167 # .167 default
	a = 1 # .101 found empirically using HyperOpt
	A = .193 # .193 default
	c = .0277 # .0277 default # T * product_size *item_size
	u = initial_sol(T, product_size, item_size)
	sample_std = normalization(T, product_size, item_size)
	# print(sample_mean)
	# scalar_u = ros.replications_of_sim(T, product_size, item_size, u)
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
		g_k = np.dot((y_thetaplus - y_thetaminus) / (2.0*c_k) / sample_std, delta_k)
		# print(a_k * g_k[0][0])

		# Step 5: Update u estimate
		# u = np.asarray(np.where((u-a_k*g_k<0, 0, u-a_k*g_k) & (u-a_k*g_k>64, 64, u-a_k*g_k)), dtype = 'int')
		u = np.where(u - a_k * g_k < lower_bound, lower_bound, u - a_k * g_k)
		u = np.where(u > upper_bound, upper_bound, u).astype('int')
		obj_value = min(ros.replications_of_sim(T, product_size, item_size, u),y_thetaplus,y_thetaminus)
		# print(u)

		# Step 6: Check for convergence
		if obj_value < best_obj:
			best_obj = obj_value
		best_obj_list.append(best_obj)

	print("The best fitness:   %d" %(best_obj))
	spsa_ans_list = []
	spsa_measurment_per_iteration = 3
	for i in best_obj_list:
		for k in range(spsa_measurment_per_iteration): spsa_ans_list.append(i)
	# -----------------------------------------------------------------------------------
	'''# visualization
	plt.figure(figsize = (15,8))
	plt.xlabel("# Measurements",fontsize = 15)
	plt.ylabel("Fitness",fontsize = 15)

	plt.plot(best_obj_list,linewidth = 2, label = "Best fitness convergence", color = 'b')
	plt.legend()
	plt.show()
	'''
	return best_obj, spsa_ans_list


'''# test
if __name__ == '__main__' :
	print("go ...")
	T, product_size, item_size = (200, 40, 30)
	import time
	time.clock = time.time
	
	tic = time.clock()
	spsa_fun(T, product_size, item_size, 100)
	time_spsa = time.clock()-tic
	print(">> SPSA in %.5f sec." %time_spsa)
'''
