import random
import numpy as np
import matplotlib.pyplot as plt
import replications_of_sim as ros
# import sys
# MAX_INT=sys.maxsize

# fully random arrival
def random_fun(T, product_size, item_size, opt_count_limit, upper_bound, initial_sol, lower_bound = 0):

	best_obj = initial_sol
	best_obj_list = [initial_sol]
	
	for k in range(opt_count_limit):
		# initialize variables
		# print(">> Case %d" %(k+1))

		# generate an solution randomly
		arrival = np.random.randint(lower_bound, upper_bound, size=(T, item_size))
		# print(arrival)

		# get the cost of the decision
		obj_value = ros.replications_of_sim(T, product_size, item_size, arrival)

		if obj_value < best_obj:
			best_obj = obj_value
			# best_arrival_set = arrival
		best_obj_list.append(best_obj)

	print("The best fitness:   %d" %(best_obj))

	return best_obj, best_obj_list #, best_arrival_set

'''# test
if __name__ == '__main__' :

	print("go ...")
	T, product_size, item_size = (200, 40, 30)
	import time
	time.clock = time.time

	Max_measurements = 4500*20
	upper_bound = product_size*1000
	initial_sol = 732384426
 
	# fully random search
	tic = time.clock()
	best_random, bl_random = random_fun(T, product_size, item_size, Max_measurements, upper_bound, initial_sol)
	time_random = time.clock()-tic

	print(">> Random in %.5f sec." %time_random)

	# visualization
	plt.figure(figsize = (15,8))
	plt.xlabel("# Measurements",fontsize = 15)
	plt.ylabel("Fitness",fontsize = 15)

	plt.plot(bl_random, linewidth = 2, label = "Best fitness convergence", color = 'b')
	plt.legend()
	plt.show()
'''