# version with priority

import random
import numpy as np
import pandas as pd

import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition

from pandas.testing import assert_frame_equal


# get backlog (by Wen-Chih Chen)
# return a dataframe ["time", "item", "backlog_qty"] with backlog > 0, otherwise return None
def get_total_backlog(arrival, demand, bom):
	# arrival: (time by item nparray) arrival quntity of a part at the beginning of t
	# demand: (time by product nparray) demand of a product at t
	# bom: (product by item nparray)

	T, product_size = demand.shape # size of time frame, product types
	item_size = bom.shape[1] # size of component types
	itemDemand = np.dot(demand, bom) # total demand of item m at t (T by item_size nparray)

	net = itemDemand - arrival # net qty (time by item nparray) at a particular period, item
	# net < 0: invetory; net > 0: backlog 
	balance = np.cumsum(net, axis=0) # detemine total backlog
	
	# convert nparray to dataframe (of x-y coordinate)
	df = pd.DataFrame(balance).stack().rename_axis(['time', 'item']).reset_index(name='backlog_qty')
	df = df.loc[df['backlog_qty']>0].reset_index(drop=True) # only keep value > 0, i.e. with backlog

	df['backlog_qty'] = df['backlog_qty'].astype('int32')
	df.sort_values(by=['time', 'item'])

	if len(df) == 0:	return None
	
	return df	


# convert ans list to dataframe and sort it
def df_sorting(ans):
	# df_ans = pd.DataFrame.from_dict(ans)
	df_ans = pd.DataFrame(ans, columns = ['time', 'item', 'product', 'temp_backlog_qty'])
	
	if len(ans) > 0:
		#　combine the backlog record by same [time, item, product]
		df_ans['backlog_qty'] = df_ans.groupby(['time', 'item', 'product'])['temp_backlog_qty'].transform('sum') 
		df_ans = df_ans[['time', 'item', 'product', 'backlog_qty']]
		df_ans.drop_duplicates(subset=None, keep='first', inplace=True)
		df_ans.sort_values(by= ['time', 'item', 'product'], ascending=[True, True, True], inplace=True)
		df_ans = df_ans.reset_index(drop=True)
	return df_ans


# allocate backlogs to each product
# return a dataframe ["time", "item", "product", "backlog_qty"] with backlog > 0, otherwise return None
def alloc_backlog(df_backlog, demand, bom):
	# df_backlog: dataframe with ["time", "item", "backlog total"]
	# demand: (time by product nparray) demand of a product at t
	# bom: (product by item nparray) the quantity of item m needed to produce one unit of product n

	if len(df_backlog) < 1:
		print(">>> All demand are satisfied. No backlog.")
		return None

	# convert df_backlog to a (int) list 
	# backlog_lst = df_backlog[['time', 'item', 'backlog_qty']]
	backlog_lst = df_backlog.values.tolist()
	
	# initialize final list
	# ans = pd.DataFrame(columns = ['time', 'item', 'product', 'temp_backlog_qty'])
	ans = []
	
	# go through all period of backlog_qty
	for e in backlog_lst:
		
		# put backlog from lowest priorty product to highest one
		priorty = len(demand[0]) - 1
		
		# total backlog of this item in this period
		
		backlog_level = e[2]
		t = int(e[0])
		
		# stop the loop when running out all backlogs 
		while(backlog_level > 0):
		
			# calculate how many items are the product require
			qty = bom[priorty][int(e[1])] * demand[t][priorty]
			
			if qty > 0 and backlog_level >= qty:
				
				ans.append([int(e[0]), int(e[1]), priorty, qty])
				# ans.append({'time': int(e[0]), 'item': int(e[1]), 'product': priorty, 'temp_backlog_qty': qty})
				# ans = pd.concat([ans, pd.DataFrame.from_records([{'time': int(e[0]), 'item': int(e[1]), 'product': priorty, 'temp_backlog_qty': qty}])], ignore_index=True)
				# ans.iloc[len(ans.index)] = [int(e[0]), int(e[1]), priorty, qty]
				# ans = pd.DataFrame.from_dict([{'time': int(e[0]), 'item': int(e[1]), 'product': priorty, 'temp_backlog_qty': qty}])
				backlog_level -= qty
				
			elif qty > 0 and backlog_level < qty:
				
				ans.append([int(e[0]), int(e[1]), priorty, backlog_level])
				# ans.append({'time': int(e[0]), 'item': int(e[1]), 'product': priorty, 'temp_backlog_qty': backlog_level})
				# ans = pd.concat([ans, pd.DataFrame.from_records([{'time': int(e[0]), 'item': int(e[1]), 'product': priorty, 'temp_backlog_qty': backlog_level}])], ignore_index=True)
				# ans.iloc[len(ans.index)] = [int(e[0]), int(e[1]), priorty, backlog_level]
				# ans = pd.DataFrame.from_dict([{'time': int(e[0]), 'item': int(e[1]), 'product': priorty, 'temp_backlog_qty': backlog_level}])
				backlog_level = 0
				
			# go to higher priorty product
			priorty -= 1
			if priorty == -1:   
				priorty = len(demand[0]) - 1
				t -= 1
		
	# tic = time.clock()

	df_ans = df_sorting(ans)
	# print(">> sorting in %.5f sec." %(time.clock()-tic))

	return df_ans


# simulation: data generation
def data_gen(T, product_size, item_size):
	# T: time frame
	# product_size: size of product types
	# item_size: size of component types

	#  ---------- Case 0 -------------
	arrival = np.array([[3,3,4]])
	demand = np.array([[5,5,5,5]])
	bom = np.array([[1,1,1],[0,0,1],[1,1,0],[1,0,0]])

	#  ---------- Case 1 -------------
	# arrival = np.array([[2,2,2], [22,18,20], [8,16,8], [0,20,12], [20,0,12]])
	# demand = np.array([[5,5,5,5],[5,5,5,5],[5,5,5,5],[5,5,5,5],[5,5,5,5]])
	# bom = np.array([[1,1,1],[0,0,0],[1,1,0],[1,0,0]])

	return arrival, demand, bom


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
		arrival, demand, bom = data_gen(T, product_size, item_size) # data generation 
		
		
		# tic = time.clock()
		df_for_wc = get_total_backlog(arrival, demand, bom)
		print(df_for_wc)
		df_for_backlog = alloc_backlog(df_for_wc, demand, bom)
		print(df_for_backlog)

