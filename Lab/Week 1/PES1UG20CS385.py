#This week code focuses on understading basic functions of pandas and numpy

import numpy as np
import pandas as pd

#input: tuple (x,y)   x,y: int
def create_numpy_ones_array(shape):
    #return a numpy array with one at all index
    array=None
    array = np.ones (shape=shape)
    return array

#input: tuple (x,y)   X,y: int
def create_numpy_zeros_array(shape) :
    #return a numpy array with zeros at all index
    array=None
    array = np.zeros (shape=shape)
    return array

#input: int
def create_identity_numpy_array(order) :
    #return a identity numpy array of the defined order
    array=None
    array = np.identity(order)
    return array

#Input: numpy array
def matrix_cofactor (array):
    #return coractor matrix or the given array
    det = np.linalg.det (array)
    if (det!=0):
        cofact = None
        cofact = np.linalg.inv(array).T * det
        #return cofactor matrix of the given matrix
        return array

#input : (numpy array, int, numpy array, int, int, int, int, tuple, tuple)        
#tuple (x,y)  x,y: int
def f1 (X1, coef1, X2, coef2, seed1, seed2, seed3, shape1, shape2):
    np.random.seed(seed1)
    m1=np.random.rand(shape1[0], shape1[1])
    np.random.seed(seed2)
    m2=np.random.rand(shape2[0], shape2[1])
    if(shape1 == shape2 and shape1[1] == X1.shape[0] == X1.shape[1] and shape2[1] == X2.shape[0] == X2.shape[1]):
        shape3 = (shape1[0], shape2[1])
        np.random.seed(seed3)
        S = np.random.rand(shape3[0], shape3[1])
        x = np.matmul(m1, np.linalg.matrix_power(X1, coef1))
        y = np.matmul(m2, np.linalg.matrix_power(X2, coef2))
        ans = x+y+S
        return ans
    return -1


def fill_with_mode(filename, column):
    df = pd.read_csv(filename)
    df[column].fillna(df[column].mode()[0], inplace = True)
    return df

def fill_with_group_average(df, group, column):
    df[column].fillna(df.groupby(group)[column].transform('mean'), inplace=True)
    return df

def get_rows_greater_than_avg(df, column):
    df=df[df[column] > df[column].mean()]
    return df