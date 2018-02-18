import numpy as np

def square(size):
    """
        square:
            returns square binary structuring element
        parameters:
            size - int, dimension of the structuring element
        returns:
            numpy array - binary structuring element with square shape
    """

    return np.ones((size, size))

def cross(size):
    """
        cross:
            returns cross-like binary structuring element
        parameters:
            size - int, dimension of the structuring element
        returns:
            numpy array - binary structuring element with cross shape
    """
    cross_ = np.zeros((size, size))
    cross_[int(size/2),: ] = 1
    cross_[:,int(size/2)] = 1
    return cross_

def circle(size):
    """
        circle:
            returns circular binary structuring element
        parameters:
            size - int, dimension of the structuring element
        returns:
            numpy array - binary structuring element with circular shape
    """

    circle = np.ones((size, size))
    for x in range(size):
        for y in range(size):
          if (x - int(size/2))**2 + (y - int(size/2))**2 > (size/2)**2:
            circle[x,y] = 0
    return circle

def hline(size):
    """
        hline:
            returns binary structuring element representing a horizontal line
        parameters:
            size - int, dimension of the structuring element
        returns:
            numpy array - binary structuring element with horizontal line
    """
    hline = np.zeros((size, size))
    hline[int(size/2),:] = 1
    return hline

def vline(size):
    """
        hline:
            returns binary structuring element representing a vertical line
        parameters:
            size - int, dimension of the structuring element
        returns:
            numpy array - binary structuring element with vertical line
    """

    vline = np.zeros((size, size))
    vline[:,int(size/2)] = 1
    return hline
