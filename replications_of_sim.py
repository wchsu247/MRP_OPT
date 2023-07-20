import simulation_model_new as simulation_model
import simulation_model_fix
import numpy as np
import scipy.stats as stats
import math
import statistics as stat
import sys
MAX_INT=sys.maxsize

# return the mean value from replications of simulation
def replications_of_sim(T, product_size, item_size, arrival, d_required = 0.015, N_min = 10, alpha = 0.05, y_lim = 20):
	'''
		Input: T, product_size, item_size, arrival (one solution)
		Parameters: 
			d_required: the set desired precision (user defined)
			N_min: the min times of replications
			alpha: significance = 1 - alpha/2
			y_lim: the max times of replications
	'''
	# index setting 
	sim_obj = []
	y = N_min
	d_y = MAX_INT
	
	while d_y >= d_required and y < y_lim:
		# print("aaaaaaa")
		if y == N_min:
			for i in range(N_min):
				obj_value = simulation_model.ans_fun(arrival, T, product_size, item_size)
				sim_obj.append(obj_value)
		else:
			obj_value = simulation_model.ans_fun(arrival, T, product_size, item_size)
			sim_obj.append(obj_value)

		# find t value of degrees = y-1, significance = 1 - alpha/2

		t_value = stats.t.ppf(1 - alpha/2, y-1)
		d_y = (t_value*(np.std(sim_obj, ddof=1)/math.sqrt(y)))/(stat.fmean(sim_obj))
		y += 1
	
	return stat.fmean(sim_obj) #, y-1

'''
# testing
print("test")
T, product_size, item_size =  (52, 4, 3)
arrival = np.random.randint(2, 50, size=(T, item_size))

ans, y = replications_of_sim(T, product_size, item_size, arrival)
print(ans, y)
'''





