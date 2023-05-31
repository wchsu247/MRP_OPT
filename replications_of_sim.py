import simulation_model
import numpy as np
import scipy.stats as stats
import math
import statistics as stat

# return 
def replications_of_sim(T, product_size, item_size, arrival, d_required = 0.01, N_check = 5, N_min = 10, alpha = 0.05):
	
	# index setting 
	# sim_obj = [3, 5, 5, 6, 7, 8, 13, 14, 14, 17, 18]
	sim_obj = []
	y = N_min
	d_y = 99999
	
	while d_y >= d_required and y < 100:
		# print("aaaaaaa")
		if y == N_min:
			for i in range(N_min):
				obj_value = simulation_model.ans_fun(arrival, T, product_size, item_size)
				sim_obj.append(obj_value)
		else:
			obj_value = simulation_model.ans_fun(arrival, T, product_size, item_size)
			sim_obj.append(obj_value)

		# find t value of degrees = y-1, significance = 1 - alpha/2

		# print(obj_value)
		# print(np.std(sim_obj, ddof=1))

		t_value = stats.t.ppf(1 - alpha/2, y-1)
		d_y = (t_value*(np.std(sim_obj, ddof=1)/math.sqrt(y)))/(stat.fmean(sim_obj))
		
		print(y, d_y, np.std(sim_obj, ddof=1))
    
		y += 1
	return stat.fmean(sim_obj), y-1

#--------------------------------------------
# testing
#--------------------------------------------
print("test")
T, product_size, item_size =  (52, 4, 3)
arrival = np.random.randint(2, 50, size=(T, item_size))

ans, y = replications_of_sim(T, product_size, item_size, arrival)
print(ans, y)
#--------------------------------------------






