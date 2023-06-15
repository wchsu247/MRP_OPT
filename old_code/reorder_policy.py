# no use

import random
import numpy as np
import old_code.simulation_model as simulation_model
import statistics
import matplotlib.pyplot as plt
import math

def reorder_policy(T, product_size, item_size, simulation_times, reorder_point = 8, order_level = 20):
	
	# initial inventory
	arrival = np.ones((1, item_size), dtype=int)*order_level
	
	# generate random demand
	demand, bom = simulation_model.data_gen(T, product_size, item_size)

	for i in range (1):
		df_production, df_stock, df_backlog = simulation_model.MRP_abstract(arrival, np.array([demand[i]]), bom)
		print(df_stock)

		temp = []
		for j in df_stock['stock']:
			if j <= reorder_point:
				temp.append(order_level - j)
			else: temp.append(0)
		print(temp)

	return 0

reorder_policy(52,4,3,20)