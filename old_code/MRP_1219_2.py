
import random
import numpy as np
import pandas as pd

from pandas.testing import assert_frame_equal

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
	
	if len(lst) > 0:
		df = pd.DataFrame(lst) 
		df.columns = ['time', 'item', 'product', 'backlog_qty'] # add column labels
		return df # return a df with columns ['time', 'item', 'product', 'backlog_qty']
	else: # no backlog
		print("\n>> All demands are satisfied!!")
		return None	


def weight_gen(T, bom):
	# T: time frame
	# bom: product by item array

	M = 999
	
	h_cost = np.ones(bom.shape[1]) # product by item by time array, unit holding cost
	
	# initialzation for backlog cost for each period
	# smaller index has higher priority
	cost_t = np.array([[product_size-n]*item_size for n in range(product_size)]) 

# 	print(cost_t)

	# A = bom == 0
	cost_t[bom == 0] = M # assign big M to element in cost_t where item m is not used for product n

	return h_cost, cost_t


# simulation: data generation
def data_gen(T, product_size, item_size):
	# T: time frame
	# product_size: size of product types
	# item_size: size of component types

 	arrival = np.array([[2,2,2], [22,18,20], [8,16,8], [0,20,12], [20,0,12]])
 	demand = np.array([[5,5,5,5],[5,5,5,5],[5,5,5,5],[5,5,5,5],[5,5,5,5]])
 	bom = np.array([[1,1,1],[0,0,0],[1,1,0],[1,0,0]])
    
 	# a = product_size * 0.5
 	# arrival = a* np.random.randint(0, 12, size=(T, item_size)) # (time by component) arrival @ beginning of t
 	# arrival[0,:] = 2 # initial inventory
 	# demand = np.random.randint(5, 6, size=(T, product_size)) # (time by product array) demand @ t 

 	# # create bom
 	# if item_size > 2:
 	#  	bom = np.random.randint(2, size=(product_size, item_size)) # product by item array # product by item array
 	# else:	
 	#  	bom = np.array([[1, 0], [0, 1]]) # product by item array

 	return arrival, demand, bom

#------------------------------------------------------------------------------------------------------
# Note
# return a dataframe ["time", "item", "product", "backlog_qty"] with backlog > 0, otherwise return None

# evaluator: backlog records
def evaluator(arrival, demand, bom):
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
                                        temp.append([b[0],b[1],b[2],b[3]-inventory_level[j],0])
                                        # temp.append([i,b[1],b[2],b[3]-inventory_level[j],0])
                                        inventory_level[j] = 0
                                        b[4] = 1
            
                                    # item level = 0
                                    else:
                                        temp.append([b[0],b[1],b[2],b[3],0])
                                        # temp.append([i,b[1],b[2],b[3],0])
                                        inventory_level[j] = 0
                                        b[4] = 1
                        for t in temp:  backlog.append(t)
                    
                # product need the item
                if bom[k][j] > 0 and demand[i][k] > 0:
                    
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

                            backlog.append([i,j,k,bom[k][j] * demand[i][k]-inventory_level[j],0])
                            inventory_level[j] = 0

    # print(inventory_level)
    for j in backlog: del j[-1]
    backlog.sort()
    
    # list convert to dataframe
    backlog = pd.DataFrame(backlog)
    if len(backlog) > 0:
        backlog.columns = ['time', 'item', 'product', 'backlog_qty']
        
    return backlog
#------------------------------------------------------------------------------------------------------

# return a dataframe ["time", "item", "product", "backlog_qty"] with backlog > 0, otherwise return None
def allocation(backlog_qty, demand, bom):
    # backlog_qty: ["time", "item", "backlog total"]
    # demand: the demand for product in period 0,...,T-1
    # bom: the quantity of item m needed to produce one unit of product n
    
    
    # backlog_qty = df_ans
    
    # a is c. convert to int list to calculate
    a = backlog_qty[['time','item','backlog_qty']]
    # a = a.astype({'time':'int'})
    # a = a.astype({'item':'int'})
    a = a.values.tolist()
    
    
    # initialize final list
    alloc = []
    
    # go through all period of backlog_qty
    for i in a:
                
        # put backlog from lowest priorty product to highest one
        priorty = len(demand[0]) - 1
        
        # total backlog of this item in this period
        backlog_level = i[2]
        
        t = int(i[0])
        
        # stop the loop when run out all of total backlog 
        while(backlog_level > 0):
            
            # calculate how many items are the product require
            qty = bom[priorty][int(i[1])] * demand[t][priorty]
            
            if qty > 0 and backlog_level >= qty:
                            
                alloc.append([t, int(i[1]), priorty,qty])                        
                backlog_level -= qty
                
            elif qty > 0 and backlog_level < qty:
                
                alloc.append([t, int(i[1]), priorty,backlog_level]) 
                backlog_level = 0
            
            # go to higher priorty product
            priorty -= 1
            
            if priorty == -1:   
                priorty = len(demand[0]) - 1
                t -= 1
    
    alloc.sort()
    alloc = pd.DataFrame(alloc)
    if len(alloc) > 0:
        alloc.columns = ['time', 'item', 'product', 'backlog_qty']
        
    return alloc
#------------------------------------------------------------------------------------------------------
# ==================================================
if __name__ == '__main__' :

	# np.random.seed(15)
	import time

	print("go ...")

    # (test_arrival, test_demand, test_bom)
    
	# case_param = [(10, 10, 6),(15, 6, 70)] # (T, product_size, item_size)

	count = 1
	round_data = 1
	test_record =[]
    
	for i in range(round_data):
    
		test_case = [(random.randint(5,5), random.randint(4,4), random.randint(3,3))]
        
		for c in test_case:
    		
			T, product_size, item_size = c
    	
			arrival, demand, bom = data_gen(T, product_size, item_size) # data generation 
			print("\n>> Case %d" %count)

			# tic = time.clock()
			time_start = time.time()		
			df_ans = MRP(arrival, demand, bom) # return a df with columns ['item', 'product', 'time', 'backlog'] with non-zero backlog       
			time_end = time.time()
			# print(">> Case %d finds solution in %.5f sec." %(count, time.clock()-tic))
			a1 = time_end - time_start
			print(">> Case %d finds solution in %.5f sec." %(count, time_end - time_start))

			time_start = time.time()
			alloc = allocation(df_ans, demand, bom)
			time_end = time.time()
			a2 = time_end - time_start
			print(">> Case %d allocation in %.5f sec." %(count, time_end - time_start))


			time_start = time.time()
			backlog = evaluator(arrival, demand, bom)
			time_end = time.time()
			b = time_end - time_start
			print(">> Case %d evaluator in %.5f sec." %(count, time_end - time_start))

			# print(df_ans)
			# if df_ans is not None:

			# backlogAnaysis(df_ans) # do some anaylsis
			assert_frame_equal(alloc, backlog, check_dtype=False)

			test_record.append([test_case, a1, a2, b])
			count +=1

    # ==================================================

    
		


