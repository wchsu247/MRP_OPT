import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from bayes_opt import BayesianOptimization, UtilityFunction
import warnings
warnings.filterwarnings("ignore")

# Prepare the data.
cancer = load_breast_cancer()
X = cancer["data"]
y = cancer["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                            stratify = y,
                                        random_state = 42)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Define the black box function to optimize.

def black_box_function(x1, x2, ):
    # C: SVC hyper parameter to optimize for.
    print(C)
    model = SVC(C = C)
    model.fit(X_train_scaled, y_train)
    y_score = model.decision_function(X_test_scaled)
    return roc_auc_score(y_test, y_score)






T, product_size, item_size = (5, 4, 3)
# n = T*item_size # Number of elements in the list
x = [f'x{str(i)}' for i in range(1, T*item_size+1)]
y = [[0, 99] for i in range(T*item_size)]
p = zip(x,y)


# print(x, y, dict(p), pbounds)


# Set range of C to optimize for.
# bayes_opt requires this to be a dictionary.
pbounds = dict(p)
# Create a BayesianOptimization optimizer,
# and optimize the given black_box_function.
optimizer = BayesianOptimization(f = black_box_function,
                                pbounds = pbounds, 
                                verbose = 2,
                                random_state = 4)
optimizer.maximize(init_points = T*item_size, n_iter = 10)
print(f'Best result: {optimizer.max["params"]}; f(x) = {optimizer.max["target"]}.')