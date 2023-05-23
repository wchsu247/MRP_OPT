import random
import numpy as np
import pandas as pd

# from pyomo.environ import *
# from pyomo.opt import *

import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition

from pandas.testing import assert_frame_equal


# do MRP calculation at the item lavel (by Wen-Chih Chen)
# return a dataframe ['time', 'item', 'product', 'backlog_qty'] with backlog > 0, otherwise return None
def MRP_abstract(arrival, demand, bom, ini_backlog = None):
	# arrival: (time by item nparray) arrival quntity of a part at the beginning of t
	# demand: (time by product nparray) demand of a product at t
	# bom: (product by item nparray)
	# ini_backlog: (item by product nparray) 

	T, product_size = demand.shape # size of time frame, product types
	item_size = bom.shape[1] # size of component types

	cum_demand = demand.cumsum(axis=0) 

	# default ini_backlog is 0
	if ini_backlog is None:
		ini_backlog = np.zeros((item_size, product_size))

	# generate costs for all activities (variables)
	cost = [[(product_size-p)+(T-t)*1000 for p in range(product_size)] for t in range(T)]
	cost = np.array(cost)

	print("Modeling...")
	# tic = time.clock() # for debug
	
	# -----------------------------------------
	#            initializtion
	# -----------------------------------------
	# Create a solver
	opt = pyo.SolverFactory('gurobi')  
	opt.options['OutputFlag'] = 0 # OutputFlag=0
	model = pyo.AbstractModel() 


	# -----------------------------------------
	# 			define SETs 
	# -----------------------------------------
	model.pSet = pyo.RangeSet(0, product_size-1, doc='product index')	# product set
	model.iSet = pyo.RangeSet(0, item_size-1, doc='component index')	# item set
	model.tSet = pyo.RangeSet(0, T-1, doc='time index')	# time set

	# -----------------------------------------
	# 			define PARAMETERs 
	# -----------------------------------------
	# def demand_rule(model, t, i):
	# 	return itemDemand[t, i]
	# model.itemD = pyo.Param(model.tSet, model.iSet, initialize=demand_rule, doc='tol item demand') # total demand of item m at tb

	# def tol_arrival_rule(model, t, i):
	# 	return arrival[t, i]
	# model.arrival = pyo.Param(model.tSet, model.iSet, initialize=tol_arrival_rule, doc='item arrival') # total arrival of item m at tb


	# -----------------------------------------
	# 			define VARIABLEs 
	# ----------------------------------------- NonNegativeIntegers; NonNegativeReals
	model.production = pyo.Var(model.tSet, model.pSet, within=pyo.NonNegativeIntegers) # production level of product p at period t
	model.backlog = pyo.Var(model.tSet, model.pSet, within=pyo.NonNegativeReals) # cumulative shortage level of product p at the end of period t
	model.stock = pyo.Var(model.tSet, model.iSet, within=pyo.NonNegativeReals) # cumulative inventory level of item i at the end of period t

	# -----------------------------------------
	# 			define VCONSTRAINTs
	# -----------------------------------------
	
	# balance equation for product p at time t
	def product_balance_rule(model, t, p):
		if t > 0:
			return model.production[t, p] == model.backlog[t-1, p] + demand[t, p] - model.backlog[t, p] 
		else:
			return model.production[t, p] == demand[t, p] - model.backlog[t, p]
	model.product_balanceCON = pyo.Constraint(model.tSet, model.pSet, rule=product_balance_rule, doc='balance of product p at t') 


	# capacity constraint for item i at time t
	def capacity_rule(model, t, i):
		if t > 0:
			return model.stock[t, i] + sum(bom[p,i]*model.production[t, p] for p in model.pSet) == arrival[t, i] + model.stock[t-1, i]
		else:
			return model.stock[t, i] + sum(bom[p,i]*model.production[t, p] for p in model.pSet) == arrival[t, i]
	model.capacityCON = pyo.Constraint(model.tSet, model.iSet, rule=capacity_rule, doc='balance of product p at t') 


	# -----------------------------------------
	# 		define VOBJECTIVE function 
	# -----------------------------------------
	def ObjRule(model): 
		return sum(model.backlog[t, p]*cost[t, p] for t in model.tSet for p in model.pSet)

	model.obj = pyo.Objective(rule=ObjRule, sense=pyo.minimize)

	# problem solving
    # Create a model instance and optimize
	instance = model.create_instance()
	# print(instance.pprint())

	print("solving ... ")
	results = opt.solve(instance, tee=False)  # solve the instance


	# ======================
	# print("computing in {0} sec.".format(time.clock()-tic)) # for debug

	# post-computing process
	# sol_b = instance.backlog.get_values() # get backog results
	# lst_b = [[key[0], key[1], int(value)] for (key, value) in sol_b.items() if value > 0] # only the backlog > 0; key: (time, item, product)
	# if len(lst_b) > 0:
	# 	df = pd.DataFrame(lst_b, columns = ['time', 'product', 'backlog_qty']) 
	# 	df = df.sort_values(by=['time', 'product'])
	# 	df = df.reset_index(drop=True)

	sol_x = instance.production.get_values() # get production results
	lst_x = [[key[0], key[1], int(value)] for (key, value) in sol_x.items()] # only the backlog > 0; key: (time, item, product)

	df_x = pd.DataFrame(lst_x, columns = ['time', 'product', 'production']) 	
	df_x = df_x.sort_values(by=['product', 'time'])
	df_x = df_x.reset_index(drop=True)

	return df_x
	# return df_x, df
	# 	return df # return a df with columns ['time', 'item', 'product', 'backlog_qty']

	# else: # no backlog
	# 	print("\n>> All demands are satisfied!!")
	# 	return None


# simulation: data generation
def data_gen(T, product_size, item_size):
	# T: time frame
	# product_size: size of product types
	# item_size: size of component types

	#  ---------- Case 0 -------------
	# arrival = np.array([[2,2,2], [22,18,20], [8,16,8], [0,20,12], [20,0,12]])
	# # arrival = np.array([[2,2,2], [22,18,20], [8,16,8], [0,20,12]])
	# demand = np.array([[5,5,5,5],[5,5,5,5],[5,5,5,5],[5,5,5,5],[5,5,5,5]])
	# bom = np.array([[1,1,1],[0,0,0],[1,1,0],[1,0,0]])


	#  ---------- Case 1 -------------
	arrival = np.array([[2,2,2], [22,18,20], [8,16,8], [0,20,12], [20,0,12]])
	# arrival = np.array([[2,2,2], [22,18,20], [8,16,8], [0,20,12]])
	demand = np.array([[5,5,5,5],[5,5,5,5],[5,5,5,5],[5,5,5,5],[5,5,5,5]])
	bom = np.array([[1,1,1],[0,0,0],[1,1,0],[1,0,0]])
	
	#  ---------- Case 2 -------------
	# arrival = np.array([[300, 500], [100, 600], [300, 300], [300, 400], [200, 200]]) 
	# demand = np.array([[10, 20],[15, 10],[10, 20],[5, 20], [15, 5]]) #A>B
	# # demand = np.array([[10, 10],[10, 0],[10, 10],[10, 10], [5, 7]]) #A>B production plan
	# bom = np.array([[10, 5], [20, 30]]) #A>B

	#  ---------- Case 3 -------------
	# demand = np.random.randint(8, 12, size=(T, product_size)) # (time by product array) demand @ t 

	# bom = np.random.randint(10, size=(product_size, item_size)) # product by item array # product by item array
	# # bom = np.ones((product_size, item_size)) # product by item array # product by item array
	# if bom.sum()==0: # no product needed
	# 	bom[0,0] = 1

	# a = int(product_size * 0.5)
	# arrival = a* np.random.randint(0, 100, size=(T, item_size)) # (time by component) arrival @ beginning of t
	# arrival[0,:] = 10 # initial inventory

	return arrival, demand, bom


# ==================================================
if __name__ == '__main__' :

	# np.random.seed(45)
	import time

	print("go ...")

	# case_param = [(52, 10, 10), (52, 10, 200), (52, 10, 500), (52, 10, 500), (52, 10, 500)] # (T, product_size, item_size)
	case_param = [(5, 4, 3)] # (T, product_size, item_size)
	count = 1

	for c in case_param:
		
		T, product_size, item_size = c
	
		arrival, demand, bom = data_gen(T, product_size, item_size) # data generation 
		# arrival: (time by item nparray) arrival quntity of a part at the beginning of t
		# demand: (time by product nparray) demand of a product at t
		# bom: (product by item nparray)

		
		# tic = time.clock()
		df_x = MRP_abstract(arrival, demand, bom) # return a df with columns ['item', 'product', 'time', 'backlog'] with non-zero backlog
		# print(">> (T, product_size, item_size)=%s: MRP_abstract() finds solution in %.3f sec." %(str(c), time.clock()-tic))
		
		# print(df_x)
		
		
		count +=1
		



