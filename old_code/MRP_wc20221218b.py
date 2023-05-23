
import random
import numpy as np
import pandas as pd

import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition


# do MRP calculation by Wen-Chih Chen
# return a dataframe ["time", item", "product", "backlog_qty"] with backlog > 0, otherwise return None
def MRP(arrival, demand, bom):
	# arrival: (time by component nparray) arrival quntity of a part at the beginning of t
	# demand: (time by product nparray) demand of a product at t
	# bom: (product by item nparray)

	T, product_size = demand.shape # size of time frame, product types
	item_size = bom.shape[1] # size of component types
	itemDemand = np.dot(demand, bom) # total demand of item m at t (T by item_size nparray)

	# print("arrival: ", arrival) # for debug
	# print("itemDemand: ", itemDemand) # for debug
	# print('demand: ', demand) # for debug
	# print("bom: ", bom) # for debug

	h_cost, b_cost = weight_gen(T, bom) # weights for holding (product by item array) and backlog (product by item array)

	print("Modeling...")
# 	tic = time.clock() # for debug

	
	# initializtion
    # Create a solver
	opt = pyo.SolverFactory('gurobi')  
	opt.options['OutputFlag'] = 0 # OutputFlag=0
	model = pyo.AbstractModel() 


	# define SETs
	model.nSet = pyo.RangeSet(0, product_size-1, doc='product index')	# product set
	model.mSet = pyo.RangeSet(0, item_size-1, doc='component index')	# item set
	model.tSet = pyo.RangeSet(0, T-1, doc='time index')	# time set

	# define PARAMETERs
	def demand_rule(model, t, m):
		return itemDemand[t, m]
	model.itemD = pyo.Param(model.tSet, model.mSet, initialize=demand_rule, doc='tol item demand') # total demand of item m at tb

	def tol_arrival_rule(model, t, m):
		return arrival[t, m]
	model.arrival = pyo.Param(model.tSet, model.mSet, initialize=tol_arrival_rule, doc='item arrival') # total arrival of item m at tb

	# print(model.arrival.pprint()) # for debug

	# define VARIABLEs 
	model.slack = pyo.Var(model.tSet, model.mSet, within=pyo.NonNegativeReals) # slack level of component n at the end of period t
	model.backlog = pyo.Var(model.tSet, model.mSet, model.nSet, within=pyo.NonNegativeReals) # shortage level of component n at the end of period t
	
	# define CONSTRAINTs 
	# the balance equation for part m at period t
	def balance_rule(model, t, m):
		if t > 0:
			return model.arrival[t, m] + model.slack[t-1, m] - sum(model.backlog[t-1, m, n] for n in model.nSet) - model.itemD[t, m] \
				== model.slack[t, m] - sum(model.backlog[t, m, n] for n in model.nSet)
		else: # t = 0 (boundary)	
			return float(model.arrival[t, m] - model.itemD[t, m]) == model.slack[t, m] - sum(model.backlog[t, m, n] for n in model.nSet) 	

	model.con = pyo.Constraint(model.tSet, model.mSet, rule=balance_rule, doc='balance at t') 

	# upper bound of the backlogs
	def ub_rule(model, t, m, n):
		if t > 0:
			return model.backlog[t, m, n] - model.backlog[t-1, m, n] <= 99999999 
		else: # t = 0 (boundary)	
			return model.backlog[t, m, n] <= demand[t, n] * bom[n, m]  	

	model.con_backlog = pyo.Constraint(model.tSet, model.mSet, model.nSet, rule=ub_rule, doc='ub of backlog')


	# define OBJECTIVE function
	def ObjRule(model): 
		return sum(h_cost[m]*model.slack[t, m] for m in model.mSet for t in model.tSet) \
				+ sum(b_cost[n,m]*model.backlog[t, m, n] \
				for n in model.nSet for m in model.mSet for t in model.tSet)

	model.obj = pyo.Objective(rule=ObjRule, sense=pyo.minimize)
# 	print("modeling in {0} sec.".format(time.clock()-tic))	# for debug
# 	tic = time.clock() # for debug

	# problem solving
    # Create a model instance and optimize
	instance = model.create_instance()

# 	print("create_instance in {0} sec.".format(time.clock()-tic)) # for debug
# 	tic = time.clock() # for debug

	print("solving ... ")
	results = opt.solve(instance, tee=False)  # solve the instance

# 	print("computing in {0} sec.".format(time.clock()-tic)) # for debug

	# post-computing process
	sol_b = instance.backlog.get_values() # get backog results
	lst = [[key[0], key[1], key[2], value] for (key, value) in sol_b.items() if value > 0] # only the backlog > 0; key: (time, item, product)

	sol_s = instance.slack.get_values() # get backog results
	lst_s = [[key[0], key[1], value] for (key, value) in sol_s.items() if value > 0] # only the backlog > 0; key: (time, item, product)

	
	if len(lst) > 0:
		df = pd.DataFrame(lst) 
		df.columns = ['time', 'item', 'product', 'backlog_qty'] # add column labels
		df_s = pd.DataFrame(lst_s) 
		df_s.columns = ['time', 'item', 'inventory_qty'] # add column labels

		return df, df_s # return a df with columns ['time', 'item', 'product', 'backlog_qty']
	else: # no backlog
		print("\n>> All demands are satisfied!!")
		return None	




def weight_gen(T, bom):
	# T: time frame
	# bom: product by item array

	M = 9999

	product_size, item_size = bom.shape


	h_cost = np.ones(item_size)/100000 # product by item by time array, unit holding cost
	
	# initialzation for backlog cost for each period
	# smaller index has higher priority
	cost_t = np.array([[product_size-n]*item_size for n in range(product_size)]) 


	# A = bom == 0
	cost_t[bom == 0] = M # assign big M to element in cost_t where item m is not used for product n

	return h_cost, cost_t


# simulation: data generation
def data_gen(T, product_size, item_size):
	# T: time frame
	# product_size: size of product types
	# item_size: size of component types

	arrival = np.array([[2,2,2], [22,18,20], [8,16,8], [0,20,12], [20,0,12]])
	demand = np.array([[5,5,5,5],[5,5,5,5],[5,5,5,5],[5,5,5,5]])
	bom = np.array([[1,1,1],[0,0,0],[1,1,0],[1,0,0]])

	a = product_size * 0.5
	arrival = a* np.random.randint(0, 12, size=(T, item_size)) # (time by component) arrival @ beginning of t
	arrival[0,:] = 200 # initial inventory
	demand = np.random.randint(5, 6, size=(T, product_size)) # (time by product array) demand @ t 

	# create bom
	if item_size > 2:
		bom = np.random.randint(2, size=(product_size, item_size)) # product by item array # product by item array
	else:	
		bom = np.array([[1, 0], [0, 1]]) # product by item array

	return arrival, demand, bom


# ==================================================
if __name__ == '__main__' :

	# np.random.seed(15)
	import time
	time.clock = time.time
    
	print("go ...")




	case_param = [(10, 2, 2), (52, 10, 200)] # (T, product_size, item_size)

	count = 1

	for c in case_param:
		
		T, product_size, item_size = c
	
		arrival, demand, bom = data_gen(T, product_size, item_size) # data generation 
		print("\n>> Case %d" %count)
		tic = time.clock()
		df_ans, df_inv = MRP(arrival, demand, bom) # return a df with columns ['item', 'product', 'time', 'backlog'] with non-zero backlog
# 		print(">> Case %d finds solution in %.5f sec." %(count, time.clock()-tic))
		
		print(df_inv)
		# if df_ans is not None:
			# backlogAnaysis(df_ans) # do some anaylsis
		
		count +=1
		


