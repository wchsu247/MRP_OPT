import random
import numpy as np
import simulation_model
import statistics
import matplotlib.pyplot as plt
import math

# fully random gerneration of arrival
def random_fun(T, product_size, item_size, simulation_times, random_count_limit = 10, lower_bound = 0, upper_bound = 64):

	count = 0
	best_obj = 99999999
	best_obj_list = []
	
	while count < random_count_limit:
		
		# initialize variables
		sim_obj = []
		simulation_count = 0
		print(">> Case %d" %(count+1))

		# generate an solution randomly
		arrival = np.random.randint(lower_bound, upper_bound, size=(T, item_size)) # arrival: (time by item nparray) arrival quntity of a part at the beginning of t


		

		# run simulation until desired number of iterations is reached
		while simulation_count < simulation_times:
			# generate random demand and BOM
			demand, bom = simulation_model.data_gen(T, product_size, item_size)
			
			# calculate MRP and objective function
			df_production, df_stock, df_backlog = simulation_model.MRP_abstract(arrival, demand, bom)
			obj_value = simulation_model.obj_function(df_production['production'].sum(),df_stock['stock'].sum(),df_backlog['backlog_qty'].sum())
			
			# add objective function value to list and update simulation count
			sim_obj.append(obj_value)

			simulation_count += 1



		obj_value = statistics.fmean(sim_obj)
		if obj_value < best_obj:
			best_obj = obj_value
			best_arrival_set = arrival
		
		best_obj_list.append(best_obj)
		count += 1
	print("The best fitness:   %d" %(best_obj))
	# print("Arrival list:   ")
	# print(best_arrival_set)

	# visualization
	plt.figure(figsize = (15,8))
	plt.xlabel("Iteration",fontsize = 15)
	plt.ylabel("Fitness",fontsize = 15)

	plt.plot(best_obj_list,linewidth = 2, label = "Best fitness convergence", color = 'b')
	plt.legend()
	plt.show()
	
	return best_obj #, best_arrival_set