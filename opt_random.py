import random
import numpy as np
import matplotlib.pyplot as plt
import replications_of_sim as ros
import sys
MAX_INT=sys.maxsize

# fully random arrival
def random_fun(T, product_size, item_size, opt_count_limit, upper_bound, lower_bound = 0):

	best_obj = MAX_INT
	best_obj_list = []
	
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
	# -----------------------------------------------------------------------------------
	'''# visualization
	plt.figure(figsize = (15,8))
	plt.xlabel("Iteration",fontsize = 15)
	plt.ylabel("Fitness",fontsize = 15)

	plt.plot(best_obj_list,linewidth = 2, label = "Best fitness convergence", color = 'b')
	plt.legend()
	plt.show()
	'''

	return best_obj, best_obj_list #, best_arrival_set
'''
# test
if __name__ == '__main__' :

	print("go ...")
	T, product_size, item_size = (200, 40, 30)
	random_fun(T, product_size, item_size)
'''