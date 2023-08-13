# version without priority

import random
import numpy as np
import pandas as pd
# from pyomo.environ import *
# from pyomo.opt import *
# import pyomo.environ as pyo
# from pyomo.opt import SolverFactory
# from pyomo.opt import SolverStatus, TerminationCondition
# from pandas.testing import assert_frame_equal
import simulation_model_new as simulation_model
import time
import replications_of_sim as ros
import warnings
warnings.filterwarnings('ignore')
time.clock = time.time

# ==================================================
if __name__ == '__main__' :

	print("go ...")
	T, product_size, item_size = (5, 4, 3)
	bom = np.random.randint(2, size=(product_size, item_size))
	demand = simulation_model.data_gen(T, product_size, item_size) # data generation
	# arrival = np.random.randint(0, 10000, size=(T, item_size))
	arrival = np.ones((T, item_size))*product_size*20
	
	tic = time.clock()
	# obj_value_1 = simulation_model.ans_fun(arrival, T, product_size, item_size)
	obj_value_1 = ros.replications_of_sim(T, product_size, item_size, bom, arrival)
	print("NEW model >> One time simulation in %.5f sec." %(time.clock()-tic))

	'''
	T, product_size, item_size = (5, 4, 3)
	demand, bom = simulation_model.data_gen(T, product_size, item_size) # data generation
	arrival = np.random.randint(0, 1024, size=(T, item_size))
	tic = time.clock()
	# obj_value_2 =  simulation_model.ans_fun(arrival, T, product_size, item_size)
	obj_value_2 = ros.replications_of_sim(T, product_size, item_size, arrival)
	print("NEW model >> One time simulation in %.5f sec." %(time.clock()-tic))
	'''
	print(obj_value_1)
