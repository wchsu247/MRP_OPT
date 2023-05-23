import numpy as np
import pandas as pd
import time

#------------------------------------------------------------------------------------------------------
# simulation: data generation
def data_gen(T, product_size, item_size):
	# T: time frame
	# product_size: size of product types
	# item_size: size of component types

	a = product_size * 0.5
	arrival = a* np.random.randint(0, 3, size=(T, item_size)) # (time by component) arrival @ beginning of t
	arrival[0,:] = 20 # initial inventory
	demand = np.random.randint(5, 6, size=(T, product_size)) # (time by product array) demand @ t 

	# create bom
	if item_size > 2:
		bom = np.random.randint(2, size=(product_size, item_size)) # product by item array # product by item array
	else:	
		bom = np.array([[1, 0], [0, 1]]) # product by item array

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
                                        inventory_level[j] = 0
                                        b[4] = 1
            
                                    # item level = 0
                                    else:
                                        temp.append([b[0],b[1],b[2],b[3],0])
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
def func(df):
    return ','.join(df.values)

#------------------------------------------------------------------------------------------------------



def fullfillment(backlog):
    
    #----------------------------   
    ans1 = backlog[['time', 'product']]
    ans1.drop_duplicates(subset=None, keep='first', inplace=True)
    
    #----------------------------  
    backlog['period'] = backlog.groupby(['time', 'item', 'product'])['backlog_qty'].transform('count') 
    ans2 = pd.merge(ans1, backlog[['time', 'product','period']], how='left', on=['time', 'product'])
    ans2['max_period'] = ans2.groupby(['time', 'product'])['period'].transform('max')
    ans2 = ans2[['time', 'product','max_period']]
    ans2.drop_duplicates(subset=None, keep='first', inplace=True)
    
    
    #----------------------------  
    backlog['max_backlog_qty'] = backlog.groupby(['time', 'item', 'product'])['backlog_qty'].transform('max')
    ans3 = pd.merge(ans1, backlog[['time', 'item', 'product','max_backlog_qty']], how='left', on=['time', 'product'])
    ans3.drop_duplicates(subset=None, keep='first', inplace=True)
    ans3['item']= ans3['item'].map(str)
    ans3 = ans3.groupby(by= ['time', 'product', 'max_backlog_qty']).agg(func).reset_index()
    ans3['backlog_qty'] = ans3.groupby(['time', 'product'])['max_backlog_qty'].transform('max')
    ans3 = ans3.where(ans3.backlog_qty == ans3.max_backlog_qty).dropna(axis = 0)
    ans3 = ans3[['time', 'product', 'item', 'backlog_qty']]
    ans3 = ans3.rename(columns={'item':'blocking_factor'})
    
    
    return ans1, ans2, ans3
#------------------------------------------------------------------------------------------------------
def main():
    
    # index setting
    T = 3
    product_size = 5
    item_size = 5
    
    
    # simulation
    arrival, demand, bom = data_gen(T, product_size, item_size)
    
    # print(bom)
    
    time_start = time.time()
    
    # performance evaluator
    backlog = evaluator(arrival, demand, bom)
    
    time_end = time.time()
    
    print('evaluator time cost = ', time_end - time_start, 's')
    print(backlog)
    
    time_start = time.time()
    ans1, ans2, ans3 = fullfillment(backlog)
    time_end = time.time()
    
    print('fullfillment calculator time cost = ', time_end - time_start, 's')
    
if __name__ == '__main__':
    main()