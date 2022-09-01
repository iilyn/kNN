# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 19:13:33 2022

@author: neural.net_
"""

import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
    
## ------------------------------------------- Simulate data 
def simulate_data():
    # data points
    cluster_1 = np.random.multivariate_normal(mean=[2, 2], cov=[[3, 0], [0, 3]], size=25).T
    cluster_2 = np.random.multivariate_normal(mean=[7, 2], cov=[[4, 1], [1, 4]], size=20).T
    cluster_3 = np.random.multivariate_normal(mean=[8, 7], cov=[[2, 0.5], [0.5, 2]], size=30).T
    cluster_4 = np.random.multivariate_normal(mean=[5, 8], cov=[[1, 0.5], [0.5, 1]], size=20).T
    
    # add label information
    data1 = np.vstack((cluster_1, np.ones((1, cluster_1.shape[1])))).T
    data2 = np.vstack((cluster_2, 2*np.ones((1,cluster_2.shape[1])))).T
    data3 = np.vstack((cluster_3, 3*np.ones((1,cluster_3.shape[1])))).T
    data4 = np.vstack((cluster_4, 4*np.ones((1,cluster_4.shape[1])))).T
    
    # create data
    data = np.concatenate((data1, data2, data3, data4))
    df = pd.DataFrame(data=data , columns = ['X', 'Y', 'label'])
    return df

## ------------------------------------------- Plot data
def plot_decision(x, k_nearest, data):
    for i in range(len(k_nearest)):
        plt.plot([x[0], k_nearest.iloc[i]['X']], 
                  [x[1] , k_nearest.iloc[i]['Y']], '--', color='grey')
    plt.plot(data[df['label'] == 1]['X'].values, data[df['label'] == 1]['Y'].values, 'x', color=CLASSES[0], label='class1')
    plt.plot(data[df['label'] == 2]['X'].values, data[df['label'] == 2]['Y'].values, 'x', color=CLASSES[1], label='class2')
    plt.plot(data[df['label'] == 3]['X'].values, data[df['label'] == 3]['Y'].values, 'x', color=CLASSES[2], label='class3')
    plt.plot(data[df['label'] == 4]['X'].values, data[df['label'] == 4]['Y'].values, 'x', color=CLASSES[3], label='class4')
    plt.plot(x[0], x[1], 'o', color='black', label='sample')
    plt.axis('equal')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('k-Nearest-Neighbors')
    plt.show()
    
## ------------------------------------------- kNN-algorithm
def kNN(x, data, k=5, distance_metric='mse'):
    def calc_distance(x, data, distance_metric):
        if distance_metric == 'mse':
            X = data['X']
            Y = data['Y']
            distance = np.sqrt((x[0]-X)**2 + (x[1] - Y)**2)
        else:
            print('Invalid: Please implement the defined distance metric.')
            sys.exit()
        return distance.values
    
    dist = calc_distance(x, data, distance_metric)
    idx_smallest_values = np.argpartition(dist, k)[:k]
    idx_labels = data.iloc[idx_smallest_values]['label'].values
    k_nearest = data.iloc[idx_smallest_values]
    label = np.bincount(idx_labels.astype(int)).argmax()
    num_votes = sum(idx_labels==label)
    return label, num_votes, k_nearest
    

## ------------------------------------------- 
# Simulate data
CLASSES = ['blue', 'red', 'orange', 'green']
df = simulate_data()

# Define the number of k and the distance metric for the kNN-algorithm
k = 5
distance_metric = 'mse'

# Sample new datapoint
x_i = np.random.randint(0,10,2)

# Predict class of new datapoint
predicted_label, num_votes, k_nearest = kNN(x_i, df, k, distance_metric)

# Print and plot prediction
print('Predicted class ' + CLASSES[predicted_label-1] + ' with ' + str(num_votes) + '/' + str(k) + ' votes.')
plot_decision(x_i, k_nearest, df)
