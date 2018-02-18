# morphological operations
#
#
#

import numpy as np

def to_structuring_array(SE):

    """
        to_structuring_array:
            structuring element is given as a matrix, this function turns it
            into an index array. Each 1 value in the structuring array
            is turned into its corresponding position in the structuring
            element.
        parameters:
            SE - structuring element as numpy array
        returns:
            numpy array of indices
    """
    # todo: correct error handeling
    if SE.shape.length != 2:
        print ( "in to_structuring_array: structuring element should have two dimensions")


    struc_array = []
    m_x = int(SE.shape[0]/2)
    m_y = int(SE.shape[1]/2)

    for x in range(SE.shape[0]):
        for y in range(SE.shape[1]):
            if SE[x,y] == 1:
                struc_array.append([x - m_x,y - m_y])
    return struc_array


def dilation(I, SE):
    # todo: maybe add option for boundary mode
    """
        dilation:
            performs morphological dilation of image I with structuring
            element SE. Returns separate array I and SE are unchanged.
        parameters:
            I  - image to perform dilation on as 2D numpy array
            SE - structuring element for dilation as 2D numpy array
        returns
            dilated image as numpy array
    """

    # todo: error handeling

    struct_array = to_structuring_array(SE)
    I_dil = I.copy()

    mx = int(SE.shape[0]/2)
    my = int(SE.shape[1]/2)

    I_ = np.zeros([I.shape[0] + mx*2, I.shape[1] + my*2 ])
    I_[mx:-mx, my:-my] = I.copy()

    for x in range(I_dil.shape[0]):
        for y in range(I_dil.shape[1]-1, -1, -1):
            I_dil[x,y] = np.max([I_[x + i + mx ,y + j + my] for [i,j] in struct_array])

    return I_dil

def erosion(I, SE):

    # todo: option for boundary mode

    """
        erosion:
            performs morphological erosion of image I with structuring
            element SE. Returns separate array I and SE are unchanged.
        parameters:
            I  - image to perform erosion on as 2D numpy array
            SE - structuring element for erosion as 2D numpy array
        returns
            eroded image as numpy array
    """


    # todo: test for correctness of input

    struct_array = to_structuring_array(SE)

    # zero padding:
    mx = int(SE.shape[0]/2)
    my = int(SE.shape[1]/2)

    I_ero = I.copy()

    I_ = np.zeros([I.shape[0] + mx*2, I.shape[1] + my*2 ])
    I_[mx:-mx, my:-my] = I.copy()

    for x in range(I_ero.shape[0]):
        for y in range(I_ero.shape[1]-1, -1, -1):
          I_ero[x,y] = np.min([I_[x + i + mx ,y + j + my] for [i,j] in struct_array])

    return I_ero


def morphological_gradient(I, SE, rescale= True):

    """
        morphological_gradient:
            Performs morphological gradient operation on input image I. This
            operation is defined as the pointwise difference of dilated and
            eroded image. Usually the intensities are rescalled afterwards
        arguments:
            I  - image to perform morphological gradient on as numpy array
            SE - structuring element for erosion and dilation as numpy array
            rescale - True (default) if greyscale values should be rescaled
    """

    mg = dilation(I, SE) -  erosion(I, SE)
    if rescale:
        mg = mg - np.min(mg)
        mg = mg * (255/np.max(mg))

    return   mg.astype(np.uint8)

def closing(I,SE):
    """
        closing:
            performs morphological closing (erosion followed by dilation) of
            image I with structuring element SE. Returns separate array I and
            SE are unchanged.
        parameters:
            I  - image to perform closing on as 2D numpy array
            SE - structuring element for closing as 2D numpy array
        returns
            closed image as numpy array
    """
    # todo: boundary mode + error handeling
    return dilation(erosion(I,SE), SE)


def opening(I,SE):
    """
        closing:
            performs morphological opening (dilation followed by erosion) of
            image I with structuring element SE. Returns separate array I and
            SE are unchanged.
        parameters:
            I  - image to perform opening on as 2D numpy array
            SE - structuring element for opening as 2D numpy array
        returns
            opened image as numpy array
    """
        # todo: boundary mode + error handeling

    return erosion(dilation(I,SE), SE)
