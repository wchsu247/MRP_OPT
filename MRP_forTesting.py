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
import simulation_model
import time
import replications_of_sim
time.clock = time.time

# ==================================================
if __name__ == '__main__' :

	print("go ...")
	T, product_size, item_size = (52, 20, 500)
	demand, bom = simulation_model.data_gen(T, product_size, item_size) # data generation

	# generate arrival
	arrival = np.random.randint(0, 16, size=(T, item_size))

	tic = time.clock()

	# df_production, df_stock, df_backlog = simulation_model.MRP_abstract(arrival, demand, bom)
	obj_value =  replications_of_sim.replications_of_sim(T, product_size, item_size, arrival)
	# simulation_model.obj_function(df_production['production'].sum(),df_stock['stock'].sum(),df_backlog['backlog_qty'].sum())
	print(">> One time simulation in %.5f sec." %(time.clock()-tic))

	print(obj_value)
	# print(df_production)
	# print(df_stock)
	# print(df_backlog)
