import numpy as np

def square(size):
  return np.ones((size, size))
  
def cross(size):
  cross_ = np.zeros((size, size))
  cross_[int(size/2),: ] = 1 
  cross_[:,int(size/2)] = 1 
  return cross_

def circle(size):
  circle = np.ones((size, size))
  for x in range(size):
    for y in range(size):
      if (x - int(size/2))**2 + (y - int(size/2))**2 > (size/2)**2:
        circle[x,y] = 0
  return circle      

def hline(size):
  
  hline = np.zeros((size, size))
  hline[int(size/2),:] = 1
  return hline
  
def vline(size):
  
  vline = np.zeros((size, size))
  vline[:,int(size/2)] = 1
  return hline  