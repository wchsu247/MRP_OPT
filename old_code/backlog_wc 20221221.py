import random
import numpy as np
import pandas as pd

import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition

from pandas.testing import assert_frame_equal


# -----------------------------------------------------------------------------------------------------
# do MRP calculation at the item lavel by straight forward for-loop checking
# as a benchmark
# return a dataframe ["time", "item", "backlog_qty"] with backlog > 0, otherwise return None
def total_backlog(arrival, demand, bom):
    # arrival: the incoming on-hand inventory level of item m at the beginning of period 0,...,T-1
    # demand: the demand for product in period 0,...,T-1
    # bom: the quantity of item m needed to produce one unit of product n
    
    # backlog = ["time", "item", "product", "backlog_qty", "processed (1) or not (0)"]
    backlog = []
    total_backlog_qty = []
    
    
    # (item) inventory level begin at 0
    inventory_level = [0] * len(bom[0])
        
    # for all time
    for i in range(len(demand)):
        # print(f'-------------- t = {i} --------------')
        
        # for all item types
        for j in range(len(bom[0])):
            # print(f'-------- item = {j} --------')
            
            #　accmulate the total backlog of j item in i period
            ttl_backlog = 0
            
            # for all product types
            for k in range(len(bom)):
                # print(f'-- product = {k} --')
                
                
                
                if k == 0:
                    # item arrival
                    inventory_level[j] += arrival[i][j]
                    
                    # backlog processing
                    temp = []
                    if len(backlog) > 0:
                        for b in backlog:
                            
                            # same item without processing
                            if b[1] == j and b[4] == 0:
                                
                                # can meet the demand
                                if inventory_level[j] >= b[3]:  
                                    inventory_level[j] -= b[3]
                                    b[4] = 1
                                    
                                # cannot meet the demand
                                else:
                                    # item level > 0
                                    if inventory_level[j] > 0:
                                        # temp.append([b[0],b[1],b[2],b[3]-inventory_level[j],0])
                                        temp.append([i,b[1],b[2],b[3]-inventory_level[j],0])
                                        ttl_backlog += b[3]-inventory_level[j]
                                        inventory_level[j] = 0
                                        b[4] = 1
            
                                    # item level = 0
                                    else:
                                        # temp.append([b[0],b[1],b[2],b[3],0])
                                        temp.append([i,b[1],b[2],b[3],0])
                                        ttl_backlog += b[3]
                                        inventory_level[j] = 0
                                        b[4] = 1
                        for t in temp:  backlog.append(t)
                    
                # product need the item
                if bom[k][j] * demand[i][k] > 0:
                    
                    # can meet the demand
                    if inventory_level[j] >= bom[k][j] * demand[i][k]:  inventory_level[j] -= bom[k][j] * demand[i][k]                       
                    
                    # cannot meet the demand
                    else:
                        
                        # item level  > 0
                        if inventory_level[j] > 0:
                            backlog.append([i,j,k,bom[k][j] * demand[i][k]-inventory_level[j],0])
                            ttl_backlog += bom[k][j] * demand[i][k]-inventory_level[j]
                            inventory_level[j] = 0

                        # item level = 0
                        else:
                            backlog.append([i,j,k,bom[k][j] * demand[i][k],0])
                            ttl_backlog += bom[k][j] * demand[i][k]
                            inventory_level[j] = 0
            if ttl_backlog > 0:
                total_backlog_qty.append([i, j, ttl_backlog])
    
    # # print(inventory_level)
    # for j in backlog: del j[-1]
    # backlog.sort()
    
    # list convert to dataframe
    total_backlog_qty = pd.DataFrame(total_backlog_qty)
    if len(total_backlog_qty) > 0:
        total_backlog_qty.columns = ['time', 'item', 'backlog_qty']
        
        # #　combine the backlog record by same [time, item, product]
        # backlog['backlog_qty'] = backlog.groupby(['time', 'item', 'product'])['temp_backlog_qty'].transform('sum') 
        # backlog = backlog[['time', 'item', 'product', 'backlog_qty']]
        # backlog.drop_duplicates(subset=None, keep='first', inplace=True)
        
    return total_backlog_qty


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

	if len(df) == 0:
		return None
	
	return df	



# do MRP calculation at the item lavel (by Wen-Chih Chen)
# return a dataframe ["time", item", "backlog_qty"] with backlog > 0, otherwise return None
def MRP_item(arrival, demand, bom):
	# arrival: (time by component nparray) arrival quntity of a part at the beginning of t
	# demand: (time by product nparray) demand of a product at t
	# bom: (product by item nparray)

	T, product_size = demand.shape # size of time frame, product types
	item_size = bom.shape[1] # size of component types
	itemDemand = np.dot(demand, bom) # total demand of item m at t (T by item_size nparray)

	h_cost, b_cost = weight_gen_item(T, bom) # weights for holding (item array) and backlog (item array)

	print("Modeling...")
	tic = time.clock() # for debug

	
	# initializtion
    # Create a solver
	opt = pyo.SolverFactory('gurobi')  
	opt.options['OutputFlag'] = 0 # OutputFlag=0
	model = pyo.AbstractModel() 


	# define SETs
	# model.nSet = pyo.RangeSet(0, product_size-1, doc='product index')	# product set
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
	model.backlog = pyo.Var(model.tSet, model.mSet, within=pyo.NonNegativeReals) # shortage level of component n at the end of period t
	# model.slack = pyo.Var(model.tSet, model.mSet, within=pyo.NonNegativeIntegers) # slack level of component n at the end of period t
	# model.backlog = pyo.Var(model.tSet, model.mSet, within=pyo.NonNegativeReals) # shortage level of component n at the end of period t
	
	# define CONSTRAINTs 
	# the balance equation for part m at period t
	def balance_rule(model, t, m):
		if t > 0:
			return model.arrival[t, m] + model.slack[t-1, m] - model.backlog[t-1, m] - model.itemD[t, m] \
				== model.slack[t, m] - model.backlog[t, m] 
		else: # t = 0 (boundary)	
			return float(model.arrival[t, m] - model.itemD[t, m]) == model.slack[t, m] - model.backlog[t, m] 	

	model.con = pyo.Constraint(model.tSet, model.mSet, rule=balance_rule, doc='balance at t') 

	# define OBJECTIVE function
	def ObjRule(model): 
		return sum(h_cost[m]*model.slack[t, m] for m in model.mSet for t in model.tSet) \
				+ sum(b_cost[m]*model.backlog[t, m] for m in model.mSet for t in model.tSet)

	model.obj = pyo.Objective(rule=ObjRule, sense=pyo.minimize)
	print("modeling in {0} sec.".format(time.clock()-tic))	# for debug
	tic = time.clock() # for debug

	# problem solving
    # Create a model instance and optimize
	instance = model.create_instance()

	print("create_instance in {0} sec.".format(time.clock()-tic)) # for debug
	tic = time.clock() # for debug

	print("solving ... ")
	results = opt.solve(instance, tee=False)  # solve the instance

	print("computing in {0} sec.".format(time.clock()-tic)) # for debug

	# post-computing process
	sol_b = instance.backlog.get_values() # get backog results
	lst = [[key[0], key[1], value] for (key, value) in sol_b.items() if value > 0] # only the backlog > 0; key: (time, item, product)

	sol_s = instance.slack.get_values() # get backog results
	lst_s = [[key[0], key[1], value] for (key, value) in sol_s.items() if value > 0] # only the backlog > 0; key: (time, item, product)

	
	if len(lst) > 0:
		df = pd.DataFrame(lst) 
		df.columns = ['time', 'item', 'backlog_qty'] # add column labels
		# df_s = pd.DataFrame(lst_s) # for debug
		# df_s.columns = ['time', 'item', 'inventory_qty'] # add column labels # for debug

		return df # return a df with columns ['time', 'item', 'backlog_qty']
		# return df, df_s # for debug
	else: # no backlog
		print("\n>> All demands are satisfied!!")
		return None
		# return None, None # for debug

def weight_gen_item(T, bom):
	# T: time frame
	# bom: product by item array

	M = 999

	product_size, item_size = bom.shape

	h_cost = np.ones(item_size) # item array, unit holding cost
	b_cost = np.ones(item_size)*10  # (item array), unit backlog cost
	# cost_t = np.array([[product_size-n]*item_size for n in range(product_size)]) 

	return h_cost, b_cost


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
	tic = time.clock() # for debug

	
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

	# # upper bound of the backlogs
	# def ub_rule(model, t, m, n):
	# 	if t > 0:
	# 		return model.backlog[t, m, n] - model.backlog[t-1, m, n] <= demand[t, n] * bom[n, m]
	# 		# return model.backlog[t, m, n] - model.backlog[t-1, m, n] <= 999999  
	# 	else: # t = 0 (boundary)	
	# 		return model.backlog[t, m, n] <= demand[t, n] * bom[n, m]  	

	# model.con_backlog = pyo.Constraint(model.tSet, model.mSet, model.nSet, rule=ub_rule, doc='ub of backlog')


	# define OBJECTIVE function
	def ObjRule(model): 
		return sum(h_cost[m]*model.slack[t, m] for m in model.mSet for t in model.tSet) \
				+ sum(b_cost[n,m]*model.backlog[t, m, n] \
				for n in model.nSet for m in model.mSet for t in model.tSet)

	model.obj = pyo.Objective(rule=ObjRule, sense=pyo.minimize)
	print("modeling in {0} sec.".format(time.clock()-tic))	# for debug
	tic = time.clock() # for debug

	# problem solving
    # Create a model instance and optimize
	instance = model.create_instance()

	print("create_instance in {0} sec.".format(time.clock()-tic)) # for debug
	tic = time.clock() # for debug

	print("solving ... ")
	results = opt.solve(instance, tee=False)  # solve the instance

	print("computing in {0} sec.".format(time.clock()-tic)) # for debug

	# post-computing process
	sol_b = instance.backlog.get_values() # get backog results
	lst = [[key[0], key[1], key[2], value] for (key, value) in sol_b.items() if value > 0] # only the backlog > 0; key: (time, item, product)

	# sol_s = instance.slack.get_values() # get backog results
	# lst_s = [[key[0], key[1], value] for (key, value) in sol_s.items() if value > 0] # only the backlog > 0; key: (time, item, product)

	
	if len(lst) > 0:
		df = pd.DataFrame(lst) 
		df.columns = ['time', 'item', 'product', 'backlog_qty'] # add column labels
		# df_s = pd.DataFrame(lst_s) 
		# df_s.columns = ['time', 'item', 'inventory_qty'] # add column labels

		return df # return a df with columns ['time', 'item', 'product', 'backlog_qty']
	else: # no backlog
		print("\n>> All demands are satisfied!!")
		return None	


def weight_gen(T, bom):
	# T: time frame
	# bom: product by item array

	M = 9999

	product_size, item_size = bom.shape


	h_cost = np.ones(item_size) # product by item by time array, unit holding cost
	
	# initialzation for backlog cost for each period
	# smaller index has higher priority
	cost_t = np.array([[product_size-n]*item_size for n in range(product_size)]) 


	# A = bom == 0
	cost_t[bom == 0] = M # assign big M to element in cost_t where item m is not used for product n

	return h_cost, cost_t


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
    backlog_lst = df_backlog[['time', 'item', 'backlog_qty']]
    # a = a.astype({'time':'int'})
    # a = a.astype({'item':'int'})
    # print(backlog_lst)
    backlog_lst = backlog_lst.values.tolist()
    
    # initialize final list
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
                            
                ans.append([int(e[0]), int(e[1]), priorty,qty])
                backlog_level -= qty
                
            elif qty > 0 and backlog_level < qty:
                
                ans.append([int(e[0]), int(e[1]), priorty, backlog_level])
                backlog_level = 0
            
            # go to higher priorty product
            priorty -= 1

            if priorty == -1:   
                priorty = len(demand[0]) - 1
                t -= 1
    
    tic = time.clock()
    ans.sort()
    df_ans = pd.DataFrame(ans) # convert list to dataframe
    
    if len(df_ans) > 0:
        df_ans.columns = ['time', 'item', 'product', 'temp_backlog_qty']
        
        #　combine the backlog record by same [time, item, product]
        df_ans['backlog_qty'] = df_ans.groupby(['time', 'item', 'product'])['temp_backlog_qty'].transform('sum') 
        df_ans = df_ans[['time', 'item', 'product', 'backlog_qty']]
        df_ans.drop_duplicates(subset=None, keep='first', inplace=True)
        # df_ans.sort_values(by= ['time', 'item', 'product'], ascending=[True, True, True])
        
    print(">> sorting in %.5f sec." %(time.clock()-tic))
    return df_ans


def naive_evaluator(arrival, demand, bom):
    # arrival: the incoming on-hand inventory level of item m at the beginning of period 0,...,T-1
    # demand: the demand for product in period 0,...,T-1
    # bom: the quantity of item m needed to produce one unit of product n
    
    # backlog = ["time", "item", "product", "backlog_qty", "processed (1) or not (0)"]
    backlog = []
    
    # (item) inventory level begin at 0
    inventory_level = [0] * len(bom[0])
        
    # for all time
    for i in range(len(demand)):
        # print(f'-------------- t = {i} --------------')
        
        # for all item types
        for j in range(len(bom[0])):
            # print(f'-------- item = {j} --------')
            
            # for all product types
            for k in range(len(bom)):
                # print(f'-- product = {k} --')
                
                if k == 0:
                    # item arrival
                    inventory_level[j] += arrival[i][j]
                    
                    # backlog processing
                    temp = []
                    if len(backlog) > 0:
                        for b in backlog:
                            
                            # same item without processing
                            if b[1] == j and b[4] == 0:
                                
                                # can meet the demand
                                if inventory_level[j] >= b[3]:  
                                    inventory_level[j] -= b[3]
                                    b[4] = 1
                                    
                                # cannot meet the demand
                                else:
                                    # item level > 0
                                    if inventory_level[j] > 0:
                                        # temp.append([b[0],b[1],b[2],b[3]-inventory_level[j],0])
                                        temp.append([i,b[1],b[2],b[3]-inventory_level[j],0])
                                        inventory_level[j] = 0
                                        b[4] = 1
            
                                    # item level = 0
                                    else:
                                        # temp.append([b[0],b[1],b[2],b[3],0])
                                        temp.append([i,b[1],b[2],b[3],0])
                                        inventory_level[j] = 0
                                        b[4] = 1
                        for t in temp:  backlog.append(t)
                    
                # product need the item
                if bom[k][j] * demand[i][k] > 0:
                    
                    # can meet the demand
                    if inventory_level[j] >= bom[k][j] * demand[i][k]:  inventory_level[j] -= bom[k][j] * demand[i][k]
                        
                    
                    # cannot meet the demand
                    else:
                        
                        # item level  > 0
                        if inventory_level[j] > 0:

                            backlog.append([i,j,k,bom[k][j] * demand[i][k]-inventory_level[j],0])
                            inventory_level[j] = 0

                        # item level = 0
                        else:

                            backlog.append([i,j,k,bom[k][j] * demand[i][k],0])
                            inventory_level[j] = 0

    # print(inventory_level)
    for j in backlog: del j[-1]
    backlog.sort()
    
    # list convert to dataframe
    backlog = pd.DataFrame(backlog)
    if len(backlog) > 0:
        backlog.columns = ['time', 'item', 'product', 'temp_backlog_qty']
        
        #　combine the backlog record by same [time, item, product]
        backlog['backlog_qty'] = backlog.groupby(['time', 'item', 'product'])['temp_backlog_qty'].transform('sum') 
        backlog = backlog[['time', 'item', 'product', 'backlog_qty']]
        backlog.drop_duplicates(subset=None, keep='first', inplace=True)
        
    return backlog


# simulation: data generation
def data_gen(T, product_size, item_size):
	# T: time frame
	# product_size: size of product types
	# item_size: size of component types

	# arrival = np.array([[2,2,2], [22,18,20], [8,16,8], [0,20,12], [20,0,12]])
	# demand = np.array([[5,5,5,5],[5,5,5,5],[5,5,5,5],[5,5,5,5]])
	# bom = np.array([[1,1,1],[0,0,0],[1,1,0],[1,0,0]])

	# arrival  = np.array([[50.0, 50.0, 50.0], [20.0, 40.0, 50.0], [0.0, 20.0, 55.0], [0.0, 35.0, 50.0]])
	# demand = np.array([[13, 10, 13, 8, 19, 12, 16, 16, 18, 2], [6, 9, 4, 6, 4, 13, 11, 6, 11, 13], [0, 17, 2, 16, 18, 10, 10, 5, 11, 13], [9, 0, 16, 15, 14, 7, 8, 14, 18, 3]])
	# bom = np.array([[0, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1], [0, 0, 1], [1, 0, 1], [0, 1, 0]])


	a = product_size * 0.5
	arrival = a* np.random.randint(0, 1, size=(T, item_size)) # (time by component) arrival @ beginning of t
	arrival[0,:] = 0 # initial inventory
	demand = np.random.randint(0, 50, size=(T, product_size)) # (time by product array) demand @ t 

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

	print("go ...")


	case_param = [(52, 10, 500)] # (T, product_size, item_size) !!! product_size need to be an even number

	count = 1

	result = []
		
	for r in range(10):
		for c in case_param:
		
			T, product_size, item_size = c
			arrival, demand, bom = data_gen(T, product_size, item_size) # data generation 
    
    
			print("\n>> Case %d" %count)
            
    # 		tic = time.clock()
    # 		df_MRP_item = MRP_item(arrival, demand, bom) # return a df with columns ['time', 'time', 'backlog_qty'] with non-zero backlog
    # 		# df_ans, df_inv = MRP_item(arrival, demand, bom) # for debug
    # 		print(">> MRP_item() in %.5f sec." %(time.clock()-tic))
    
    # 		tic = time.clock()
    # 		df_MRP = MRP(arrival, demand, bom)[['time', 'item', 'backlog_qty']] # return a df with columns ['time', 'time', 'backlog_qty'] with non-zero backlog
    # 		# df_ans, df_inv = MRP_item(arrival, demand, bom) # for debug
    # 		print(">> MRP in %.5f sec." %(time.clock()-tic))		
    
    
    # 		tic = time.clock()
    # 		df_for_hsu = total_backlog(arrival, demand, bom)
    # 		
    # 		print(">> total_backlog in %.5f sec." %(time.clock()-tic))
    
    
    
			tic = time.clock()
			df_for_wc = get_total_backlog(arrival, demand, bom)
			print(">> get_total_backlog in %.5f sec." %(time.clock()-tic))
            
			print(">> length of total_backlog = %.5f." %(len(df_for_wc)))
            
			tic = time.clock()
			df_for_backlog = alloc_backlog(df_for_wc, demand, bom)
			print(">> alloc_backlog() in %.5f sec." %(time.clock()-tic))
            
# 			print("\n run naive_evaluator()...")
# 			tic = time.clock()
# 			df_backlog3 = naive_evaluator(arrival, demand, bom)
# 			print(">> naive_evaluator() in %.5f sec." %(time.clock()-tic))
            
# 			assert_frame_equal(df_backlog3, df_for_backlog, check_dtype=False)
    # 		assert_frame_equal(df_MRP_item, df_for_wc, check_dtype=False)
    # 		assert_frame_equal(df_MRP, df_for_wc, check_dtype=False)
    
			result.append([count, ])
			count +=1
		


