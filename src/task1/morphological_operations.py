# morphological operations


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

    #check the validity of the SE input
    if type(SE) is not np.ndarray:
        raise ValueError("The SE has to be a 2D numpy array")
    if len(SE.shape) != 2:
        raise ValueError("The SE has to be a 2D numpy array")

    struc_array = []
    m_x = int(SE.shape[0]/2)
    m_y = int(SE.shape[1]/2)

    for x in range(SE.shape[0]):
        for y in range(SE.shape[1]):
            if SE[x,y] == 1:
                struc_array.append([x - m_x,y - m_y])
    return struc_array


def dilation(I, SE, boundary_mode = 'padding'):
    """
        dilation:
            performs morphological dilation of image I with structuring
            element SE. Returns separate array I and SE are unchanged.
        parameters:
            I  - image to perform dilation on as 2D numpy array
            SE - structuring element for dilation as 2D numpy array
            boundary_mode - 'padding' or  rescaling ('crop') size of output
        returns
            dilated image as numpy array with size either same as input or cropped
            according to structuing element
    """

    #check the validity of the SE input
    if type(SE) is not np.ndarray:
        raise ValueError("The SE has to be a 2D numpy array")
    if len(SE.shape) != 2:
        raise ValueError("The SE has to be a 2D numpy array")

    #check the validity of the SI input
    if type(I) is not np.ndarray:
        raise ValueError("The image I has to be a 2D numpy array")
    if len(I.shape) != 2:
        raise ValueError("The image I has to be a 2D numpy array")

    if boundary_mode != 'padding' and boundary_mode != 'crop':
        raise ValueError("only 'padding' and 'crop' allowed as boundary mode")

    struct_array = to_structuring_array(SE)
    mx = int(SE.shape[0]/2)
    my = int(SE.shape[1]/2)

    if boundary_mode == 'padding':
        I_dil = I.copy()
        I_ = np.zeros([I.shape[0] +2* mx, I.shape[1] + 2*my ])
        I_[mx:-mx, my:-my] = I.copy()

    elif boundary_mode == 'crop':
        I_dil = np.zeros((I.shape[0] -2* mx, I.shape[1] - 2*my ))
        I_ = I.copy()

    for x in range(I_dil.shape[0]):
        for y in range(I_dil.shape[1]):
            I_dil[x,y] = np.max([I_[x + i + mx ,y + j +my] for [i,j] in struct_array])

    return I_dil

def erosion(I, SE, boundary_mode = 'padding'):
    """
        erosion:
            performs morphological erosion of image I with structuring
            element SE. Returns separate array I and SE are unchanged.
        parameters:
            I  - image to perform erosion on as 2D numpy array
            SE - structuring element for erosion as 2D numpy array
            boundary_mode - 'padding' or  rescaling ('crop') size of output
        returns
            eroded image as numpy array with size either same as input or cropped 
            according to structuing element
    """

    #check the validity of the SE input
    if type(SE) is not np.ndarray:
        raise ValueError("The SE has to be a 2D numpy array")
    if len(SE.shape) != 2:
        raise ValueError("The SE has to be a 2D numpy array")

    #check the validity of the SI input
    if type(I) is not np.ndarray:
        raise ValueError("The image I has to be a 2D numpy array")
    if len(I.shape) != 2:
        raise ValueError("The image I has to be a 2D numpy array")

    if boundary_mode != 'padding' and boundary_mode != 'crop':
        raise ValueError("only 'padding' and 'crop' allowed as boundary mode")

    struct_array = to_structuring_array(SE)
    mx = int(SE.shape[0]/2)
    my = int(SE.shape[1]/2)

    if boundary_mode == 'padding':
        I_ero = I.copy()
        I_ = np.zeros([I.shape[0] + mx*2, I.shape[1] + my*2 ])
        I_[mx:-mx, my:-my] = I.copy()

    elif boundary_mode == 'crop':
        I_ero = np.zeros((I.shape[0] - 2*mx, I.shape[1] - 2*my ))
        I_ = I.copy()


    for x in range(I_ero.shape[0]):
        for y in range(I_ero.shape[1]):
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

    #check the validity of the SE input
    if type(SE) is not np.ndarray:
        raise ValueError("The SE has to be a 2D numpy array")
    if len(SE.shape) != 2:
        raise ValueError("The SE has to be a 2D numpy array")

    #check the validity of the SI input
    if type(I) is not np.ndarray:
        raise ValueError("The image I has to be a 2D numpy array")
    if len(I.shape) != 2:
        raise ValueError("The image I has to be a 2D numpy array")

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
    #check the validity of the SE input
    if type(SE) is not np.ndarray:
        raise ValueError("The SE has to be a 2D numpy array")
    if len(SE.shape) != 2:
        raise ValueError("The SE has to be a 2D numpy array")

    #check the validity of the SI input
    if type(I) is not np.ndarray:
        raise ValueError("The image I has to be a 2D numpy array")
    if len(I.shape) != 2:
        raise ValueError("The image I has to be a 2D numpy array")

    return dilation(erosion(I,SE, 'crop'), SE, 'crop')


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

    #check the validity of the SE input
    if type(SE) is not np.ndarray:
        raise ValueError("The SE has to be a 2D numpy array")
    if len(SE.shape) != 2:
        raise ValueError("The SE has to be a 2D numpy array")

    #check the validity of the SI input
    if type(I) is not np.ndarray:
        raise ValueError("The image I has to be a 2D numpy array")
    if len(I.shape) != 2:
        raise ValueError("The image I has to be a 2D numpy array")

    return erosion(dilation(I,SE, 'crop'), SE, 'crop')
