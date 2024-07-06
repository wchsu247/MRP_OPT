# version without priority
import random
import numpy as np
import pandas as pd
# from pyomo.environ import *
# from pyomo.opt import *
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition
from pandas.testing import assert_frame_equal
import statistics
import math
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import warnings

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

	# print("Modeling...")
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

	# print("solving ... ")
	results = opt.solve(instance, tee=False)  # solve the instance


	# ======================
	# print("computing in {0} sec.".format(time.clock()-tic)) # for debug

	# post-computing process
	sol_b = instance.backlog.get_values() # get backog results
	lst_b = [[key[0], key[1], int(value)] for (key, value) in sol_b.items() if value > 0] # only the backlog > 0; key: (time, item, product)
	if len(lst_b) > 0:
		df = pd.DataFrame(lst_b, columns = ['time', 'product', 'backlog_qty']) 
		df = df.sort_values(by=['time', 'product'])
		df = df.reset_index(drop=True)
	else:
		df = pd.DataFrame(lst_b, columns = ['time', 'product', 'backlog_qty'])

	sol_x = instance.production.get_values() # get production results
	lst_x = [[key[0], key[1], int(value)] for (key, value) in sol_x.items()] # only the backlog > 0; key: (time, item, product)
	
	df_x = pd.DataFrame(lst_x, columns = ['time', 'product', 'production'])
	df_x = df_x.sort_values(by=['product', 'time'])
	df_x = df_x.reset_index(drop=True)


	sol_y = instance.stock.get_values() # get production results
	lst_y = [[key[0], key[1], int(value)] for (key, value) in sol_y.items()] # only the backlog > 0; key: (time, item, product)
	
	df_y = pd.DataFrame(lst_y, columns = ['time', 'item', 'stock'])
	df_y = df_y.sort_values(by=['item', 'time'])
	df_y = df_y.reset_index(drop=True)


	return df_x, df_y, df
	# return df_x, df
	# 	return df # return a df with columns ['time', 'item', 'product', 'backlog_qty']

	# else: # no backlog
	# 	print("\n>> All demands are satisfied!!")
	# 	return None

# simulation: data generation
def data_gen(T, product_size, item_size, demand_lb = 8, demand_ub = 20):
	# product_size: size of product types
	# item_size: size of component types

	bom = np.ones([product_size, item_size], dtype = int)
	demand = np.random.randint(demand_lb, demand_ub, size=(T, product_size)) # (time by product array) demand @ t 

	return demand, bom

def obj_function(production, stock, backlog):
	cost_p = 1
	cost_s = 5
	cost_b = 10
	return cost_p*production + cost_s*stock + cost_b*backlog

def ans_fun(arrival, T, product_size, item_size, ini_backlog = None):
    # calculate MRP and objective function
    demand, bom = data_gen(T, product_size, item_size)
    df_production, df_stock, df_backlog = MRP_abstract(arrival, demand, bom)
    return obj_function(df_production['production'].sum(), df_stock['stock'].sum(), df_backlog['backlog_qty'].sum())
