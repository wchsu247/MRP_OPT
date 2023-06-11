import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import seaborn as sns
import simulation_model
import replications_of_sim

def generateXvector(X):
    """ Taking the original independent variables matrix and add a row of 1 which corresponds to x_0
        Parameters:
            X:  independent variables matrix
        Return value: the matrix that contains all the values in the dataset, not include the outcomes values 
    """
    vectorX = np.c_[np.ones((len(X), 1)), X]
    return vectorX

def theta_init(X):
    """ Generate an initial value of vector θ from the original independent variables matrix
        Parameters:
            X:  independent variables matrix
        Return value: a vector of theta filled with initial guess
    """
    theta = np.random.randn(len(X[0])+1, 1)
    return theta

def Multivariable_Linear_Regression(X, y, learningrate, iterations):
    """ Find the multivarite regression model for the data set
        Parameters:
            X:  independent variables matrix
            y: dependent variables matrix
        learningrate: learningrate of Gradient Descent
        iterations: the number of iterations
        Return value: the final theta vector and the plot of cost function
    """
    y_new = np.reshape(y, (len(y), 1)) 
    cost_lst = []
    vectorX = generateXvector(X)
    print(vectorX)
    theta = theta_init(X)
    m = len(X)
    for i in range(iterations):
        gradients = 2/m * vectorX.T.dot(vectorX.dot(theta) - y_new)
        theta = theta - learningrate * gradients
        y_pred = vectorX.dot(theta)
        
        cost_value = 1/(2*len(y))*((y_pred - y)**2) #Calculate the loss for each training instance
        
        total = 0
        for i in range(len(y)):
            total += cost_value[i][0] #Calculate the cost function for each iteration
        cost_lst.append(total)
    
    plt.plot(np.arange(1,iterations),cost_lst[1:], color = 'red')
    plt.title('Cost function Graph')
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost')
    plt.show()
    return theta


# ------------------------------------------------------------------------------------------------------------------------
def random_fun(T, product_size, item_size, random_count_limit = 30, lower_bound = 0, upper_bound = 64):

	count = 0
	best_obj = 99999999
	best_obj_list = []
	arr_1 = []
	arr_2 = []
    
	while count < random_count_limit:
		
		# initialize variables
		print(">> Case %d" %(count+1))

		# generate an solution randomly
		arrival = np.random.randint(lower_bound, upper_bound, size=(T, item_size))
		arr_1.append(arrival)
		# get the cost of the decision
		obj_value = replications_of_sim.replications_of_sim(T, product_size, item_size, arrival)
		arr_2.append(obj_value)
		'''
		# run simulation until desired number of iterations is reached
		while simulation_count < simulation_times:
			# generate random demand and BOM
			demand, bom = simulation_model.data_gen(T, product_size, item_size)
			
			# calculate MRP and objective function
			df_production, df_stock, df_backlog = simulation_model.MRP_abstract(arrival, demand, bom)
			obj_value = simulation_model.obj_function(df_production['production'].sum(),df_stock['stock'].sum(),df_backlog['backlog_qty'].sum())
			
			# add objective function value to list and update simulation count
			sim_obj.append(obj_value)

			simulation_count += 1
		obj_value = statistics.fmean(sim_obj)
		'''

		if obj_value < best_obj:
			best_obj = obj_value
			# best_arrival_set = arrival
		best_obj_list.append(best_obj)
		count += 1

	# print("The best fitness:   %d" %(best_obj))
	# print("Arrival list:   ")
	# print(best_arrival_set)

	return np.array(arr_1), np.array(arr_2)
# ------------------------------------------------------------------------------------------------------------------------





from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
print(type(X), X.shape)
print(type(y), y.shape)


# -------------------------------------------------------------------------------
T, product_size, item_size =  (5, 4, 3)
'''
X_ = np.random.randint(2, 50, size=(T, item_size))
y_ = replications_of_sim.replications_of_sim(T, product_size, item_size, X_)
'''

X_, y_ = random_fun(T, product_size, item_size)
X_2 = X_.reshape(30, T*item_size)

print(X_, X_2)
print(y_)

print(type(X_2), X_2.shape)
print(type(y_), y_.shape)
# -------------------------------------------------------------------------------

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_transform=sc.fit_transform(X_2)

print(X_transform)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_transform, y_)
print(lin_reg.intercept_, lin_reg.coef_)

Multivariable_Linear_Regression(X_2,y_, 0.03, 30000)