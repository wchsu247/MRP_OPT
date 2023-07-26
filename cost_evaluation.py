import simulation_model_new as simulation_model
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def cost_evaluation(T, product_size, item_size, arrival, sample_lb = 30, cl = 0.99):
    sample_size = sample_lb
    sample_list = []
    for i in range(sample_lb):
        sample_list.append(simulation_model.ans_fun(arrival, T, product_size, item_size))

    error_ratio = 1
    while error_ratio >= 0.005:
        sample_size += 1
        sample_list.append(simulation_model.ans_fun(arrival, T, product_size, item_size))

        sample_mean = np.mean(sample_list)
        sample_std = np.std(sample_list)
        ci = stats.norm.interval(cl, sample_mean, sample_std/ np.sqrt(sample_size))
        error_ratio = (ci[1]-sample_mean)/sample_mean
    print(sample_mean, sample_std, ci, error_ratio, sample_size)
    
    # visulization
    plt.hist(sample_list, bins=100)
    plt.axvline(sample_mean, color='red', linestyle='dashed', linewidth=2)
    plt.axvline(ci[0], color='magenta', linestyle='dashed', linewidth=2)
    plt.axvline(ci[1], color='magenta', linestyle='dashed', linewidth=2)
    plt.show()
    
    
    return sample_mean, sample_std, sample_size

'''# test
arrival = np.array([48, 40, 21, 18, 4, 33, 31, 15, 15,  5, 24, 50, 27 , 1, 2])
T, product_size, item_size = (200, 40, 30)
a, b, c = cost_evaluation(T, product_size, item_size, np.ones((T, item_size))*38400) # arrival.reshape(T,item_size)
'''
