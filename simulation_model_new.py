# version with priority

import random
from numpy import random
import numpy as np
import pandas as pd

# get backlog (by Wen-Chih Chen)
# return a dataframe ["time", "item", "backlog_qty"] with backlog > 0, otherwise return None
def get_total_backlog(arrival, demand, bom):
	# arrival: (time by item nparray) arrival quntity of a part at the beginning of t
	# demand: (time by product nparray) demand of a product at t
	# bom: (product by item nparray)

	itemDemand = np.dot(demand, bom) # total demand of item m at t (T by item_size nparray)
	# print(itemDemand)

	net = itemDemand - arrival # net qty (time by item nparray) at a particular period, item
	# net < 0: invetory; net > 0: backlog 
	balance = np.cumsum(net, axis=0) # detemine total backlog
	# print(balance)

	# convert nparray to dataframe (of x-y coordinate)
	df = pd.DataFrame(balance).stack().rename_axis(['time', 'item']).reset_index(name='backlog_qty')
	df['stock_qty'] = df['backlog_qty'] * -1
	df.loc[df['backlog_qty']<0, 'backlog_qty'] = 0
	df.loc[df['backlog_qty']>0, 'stock_qty'] = 0

	df = df.astype('int32')
	# df.sort_values(by=['time', 'item'])

	return None if len(df) == 0 else  df

def obj_function(stock, backlog, cost_s = .5, cost_b = 1):
	return cost_s*stock + cost_b*backlog

def ans_fun(arrival, T, product_size, item_size, ini_backlog = None):
    # calculate MRP and objective function
    demand, bom = data_gen(T, product_size, item_size)
    df = get_total_backlog(arrival, demand, bom)
    return obj_function(df['stock_qty'].sum(), df['backlog_qty'].sum())

# simulation: data generation
def data_gen(T, product_size, item_size, lam = 800):
	# T: time frame
	# product_size: size of product types
	# item_size: size of component types

	bom = np.random.randint(2, size=(product_size, item_size))
	# demand = np.random.randint(demand_lb, demand_ub, size=(T, product_size)) # (time by product array) demand @ t 
	
	# Poisson Distribution
	demand = random.poisson(lam, size=(T, product_size)) # lam — rate or known number of occurences e.g. 2 for above problem
	

	#  ---------- Case 1 -------------
	# demand = np.array([[5,5,5,5],[5,5,5,5],[5,5,5,5],[5,5,5,5],[5,5,5,5]])
	# bom = np.array([[1,1,1],[0,0,0],[1,1,0],[1,0,0]])

	return demand, bom

'''
# ==================================================
if __name__ == '__main__' :
	# np.random.seed(15)
	import time
	time.clock = time.time

	print("go ...")
	case_param = [(5, 4, 3)] # (T, product_size, item_size) !!! product_size need to be an even number

	# result = []
	for c in case_param:
			
		T, product_size, item_size = c

		arrival = np.array([[2,2,2], [22,18,20], [8,16,8], [0,20,12], [20,0,12]])
		# arrival, demand, bom = data_gen(T, product_size, item_size) # data generation 
		
		ans_test = ans_fun(arrival, T, product_size, item_size)
		# tic = time.clock()
		# df_backlog, df_stock = get_total_backlog(arrival, demand, bom)
		print(ans_test)
'''