
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

