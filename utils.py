import scipy.stats

import copy
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data.sampler import Sampler
import random
#from sklearn.model_selection import train_test_split

def train_test_split(X, Y, split_rate):
    
    train_idx_overall = np.array([])
    test_idx_overall = np.array([])

    for l in np.unique(Y):

        idx = np.where(Y == l)[0]

        test_size = int(len(idx) * split_rate)

        test_choice = np.random.choice(len(idx), size=test_size, replace=False)

        train_idx = np.delete(idx, test_choice)

        test_idx = idx[test_choice]
        
        train_idx_overall = np.append(train_idx_overall, train_idx)
        
        test_idx_overall = np.append(test_idx_overall, test_idx)
        
        
    return (X[train_idx_overall.astype(int)], Y[train_idx_overall.astype(int)],
            X[test_idx_overall.astype(int)], Y[test_idx_overall.astype(int)],
            train_idx_overall, test_idx_overall)


def running_mean (data):
    
    out = np.zeros(len(data))
    for i in range(int(np.ceil(len(data) / 10))):
        second_mean = data[10*i:10*(i+1)].mean()
        out[10*i:10*(i+1)] = second_mean
        
    return out

def amplitude (data):
    
    diff_mask = np.zeros(len(data))
    for i in range(int(np.ceil(len(data) / 10))):
        second_diff = data[10*i:10*(i+1)].max() - data[10*i:10*(i+1)].min()
        diff_mask[10*i:10*(i+1)] = second_diff
        
    return diff_mask.mean()

def cross_count (a, b): 
    count = 0
    diff = a > b   
    for i in range(len(diff) - 1):
        if diff[i + 1] != diff[i]:
            count+=1
    return count     

def sum_stats (data, repeat=False, repeat_size=None):
    
    out = []
    
    x_axis = data[0::3]
    y_axis = data[1::3]
    z_axis = data[2::3]
    
    if repeat:
        x_axis = repeat_crop_data (x_axis, repeat_size)
        y_axis = repeat_crop_data (y_axis, repeat_size)
        z_axis = repeat_crop_data (z_axis, repeat_size)
        
    mean_x = x_axis.mean()
    std_x = x_axis.std()
    skew_x = scipy.stats.skew(x_axis)
    kurtosis_x = scipy.stats.kurtosis(x_axis)
    max_val_x = x_axis.max()
    min_val_x = x_axis.min()
    norm_x = np.linalg.norm(x_axis)/len(x_axis)
    
    out.append(mean_x)
    out.append(std_x)
    out.append(skew_x)
    out.append(kurtosis_x)
    out.append(max_val_x)
    out.append(min_val_x)
    out.append(norm_x)
    
    mean_y = y_axis.mean()
    std_y = y_axis.std()
    skew_y = scipy.stats.skew(y_axis)
    kurtosis_y = scipy.stats.kurtosis(y_axis)
    max_val_y = y_axis.max()
    min_val_y = y_axis.min()
    norm_y = np.linalg.norm(y_axis)/len(y_axis)
    
    out.append(mean_y)
    out.append(std_y)
    out.append(skew_y)
    out.append(kurtosis_y)
    out.append(max_val_y)
    out.append(min_val_y)
    out.append(norm_y)

    mean_z = z_axis.mean()
    std_z = z_axis.std()
    skew_z = scipy.stats.skew(z_axis)
    kurtosis_z = scipy.stats.kurtosis(z_axis)
    max_val_z = z_axis.max()
    min_val_z = z_axis.min()
    norm_z = np.linalg.norm(z_axis)/len(z_axis)
    
    out.append(mean_z)
    out.append(std_z)
    out.append(skew_z)
    out.append(kurtosis_z)
    out.append(max_val_z)
    out.append(min_val_z)
    out.append(norm_z)

    cov_xy = np.cov(x_axis, y_axis)[0][1]
    cov_xz = np.cov(x_axis, z_axis)[0][1]
    cov_yz = np.cov(y_axis, z_axis)[0][1]
    
    out.append(cov_xy) 
    out.append(cov_xz)
    out.append(cov_yz)

    cor_xy = np.corrcoef(x_axis, y_axis)[0][1]
    cor_xz = np.corrcoef(x_axis, z_axis)[0][1]
    cor_yz = np.corrcoef(y_axis, z_axis)[0][1]
    
    out.append(cor_xy) 
    out.append(cor_xz)
    out.append(cor_yz)

    mean_diff_xy = abs(mean_x-mean_y)
    mean_diff_xz = abs(mean_x-mean_z)
    mean_diff_yz = abs(mean_y-mean_z)
    
    out.append(mean_diff_xy) 
    out.append(mean_diff_xz)
    out.append(mean_diff_yz)

    std_diff_xy = abs(std_x-std_y)
    std_diff_xz = abs(std_x-std_z)
    std_diff_yz = abs(std_y-std_z)
    
    out.append(std_diff_xy) 
    out.append(std_diff_xz)
    out.append(std_diff_yz)

    DBA_x = abs(x_axis - running_mean(x_axis)).sum()
    DBA_y = abs(y_axis - running_mean(y_axis)).sum()
    DBA_z = abs(z_axis - running_mean(z_axis)).sum()
    
    out.append(DBA_x)
    out.append(DBA_y)
    out.append(DBA_z)

    ODBA = DBA_x + DBA_y + DBA_z
    
    out.append(ODBA)

    amp_x = amplitude(x_axis)
    amp_y = amplitude(y_axis)
    amp_z = amplitude(z_axis)
    
    out.append(amp_x)
    out.append(amp_y)
    out.append(amp_z)

    cross_xy = cross_count(x_axis, y_axis)
    cross_xz = cross_count(x_axis, z_axis)
    cross_yz = cross_count(y_axis, z_axis)
    
    out.append(cross_xy) 
    out.append(cross_xz)
    out.append(cross_yz)

    x_25 = np.percentile(x_axis, 25)
    x_50 = np.percentile(x_axis, 50)
    x_75 = np.percentile(x_axis, 75)
    
    out.append(x_25)
    out.append(x_50)
    out.append(x_75)

    y_25 = np.percentile(y_axis, 25)
    y_50 = np.percentile(y_axis, 50)
    y_75 = np.percentile(y_axis, 75)
    
    out.append(y_25)
    out.append(y_50)
    out.append(y_75)

    z_25 = np.percentile(z_axis, 25)
    z_50 = np.percentile(z_axis, 50)
    z_75 = np.percentile(z_axis, 75)
    
    out.append(z_25)
    out.append(z_50)
    out.append(z_75)
    
    return out


def round_cv (clf, X, Y, cv):
    
    best_score = 0.
    overall_score = []

    for i in range(10):

        scores = []

        for j in range(10):

            train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=1/cv)

            clf = clf.fit(train_x, train_y)

            scores.append(clf.score(test_x, test_y))

        if best_score < np.mean(scores):
            best_score = np.mean(scores)
            
        overall_score.append(np.mean(scores))

    return best_score, np.mean(overall_score)


def repeat_crop_data (x, size):
    
    while len(x) < size:
        x = np.tile(x, 2)
        
    x = x[:size]
    
    return x

##################################
## Class-aware sampling, partly implemented by frombeijingwithlove

class RandomCycleIter:
    
    def __init__ (self, data):
        self.data_list = list(data)
        self.length = len(self.data_list)
        self.i = self.length - 1
        
    def __iter__ (self):
        return self
    
    def __next__ (self):
        self.i += 1
        
        if self.i == self.length:
            self.i = 0
            random.shuffle(self.data_list)
            
        return self.data_list[self.i]
    
def class_aware_sample_generator (cls_iter, data_iter_list, n):
    
    i = 0
    
    while i < n:
        yield next(data_iter_list[next(cls_iter)])
        i += 1
        
class ClassAwareSampler (Sampler):
    
    def __init__ (self, data_source, num_classes, num_samples=0):
        
        self.data_source = data_source
        self.class_iter = RandomCycleIter(range(num_classes))
        class_data_list = [[] for _ in range(num_classes)]
        
        for idx, label in enumerate(self.data_source.Y.astype(np.int)):
            class_data_list[label].append(idx) 
            
        self.data_iter_list = [RandomCycleIter(x) for x in class_data_list]
        
        self.num_samples = max([len(x) for x in class_data_list]) * len(class_data_list)
        
    def __iter__ (self):
        return class_aware_sample_generator(self.class_iter, self.data_iter_list, self.num_samples)
    
    def __len__ (self):
        return self.num_samples