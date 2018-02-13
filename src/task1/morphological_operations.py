# morphological operations
#
#
#

import numpy as np 

# to_structuring_array: 
# input: 
#   SE - structuring element (numpy array)
# returns: 
#   struc_array 
     
def to_structuring_array(SE):
  struc_array = []
  m_x = int(SE.shape[0]/2)
  m_y = int(SE.shape[1]/2)
  
  for x in range(SE.shape[0]):
    for y in range(SE.shape[1]):
      if SE[x,y] == 1:
        struc_array.append([x - m_x,y - m_y])
  return struc_array
    
# dilation algorithm
# input: I - image as numpy array
#        SE - Structuring element as numpy array

def dilation(I, SE):
  # todo: test for correctness of input 
  
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
  # todo: test for correctness of input 
  
  struct_array = to_structuring_array(SE)  
  
  
  #for x, y  in np.indices((I.shape[0] - int(SE.shape[0] // 1) , I.shape[1] - int(SE.shape[1] // 1) )).reshape(-1,2): 
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


def morphological_gradient(I, SE):
  mg = dilation(I, SE) -  erosion(I, SE)
  mg = mg - np.min(mg)
  mg = mg * (255/np.max(mg))  
  return   mg.astype(np.uint8)

def closing(I,SE):
  return dilation(erosion(I,SE), SE) 
  

def opening(I,SE):
  return erosion(dilation(I,SE), SE)   
  