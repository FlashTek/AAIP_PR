import numpy as np
import matplotlib.pyplot as plt
import morphological_operations as mo
import structuring_elements as se 
from PIL import Image
import matplotlib
from cycler import cycler

# plotting parameters

matplotlib.rc('text', usetex = True)
matplotlib.rc('font', **{'size':9})
params = {'text.latex.preamble': [r'\usepackage{siunitx}', 
    r'\usepackage{sfmath}', r'\sisetup{detect-family = true}',
    r'\usepackage{amsmath}']}   
plt.rcParams.update(params)   
plt.rc('axes', prop_cycle=(cycler('color', ['b', 'r', 'k', 'c', 'm', 'y']) ))
figsize = (6.29 , 3.54) # = 16*9 cm
dpi = 150


#to_denoise = np.zeros((100, 100))
#to_denoise[50:] = 255

#mask = np.random.random((100, 100))
#mask = [mask > 0.7]

#mask_ = np.random.random((100, 100))
#mask_ = [mask_ > 0.7]

#to_denoise[mask_] = 255 
#to_denoise[mask] = 0

to_denoise = np.random.randn(100,100) 
to_denoise = to_denoise - np.min(to_denoise)
SE = se.square(5)

denoised = mo.closing(to_denoise, SE)

SE1 = se.square(9)
denoised1 = mo.closing(to_denoise, SE1)


fig, ax = plt.subplots(1,3, figsize = figsize, dpi  = dpi)
ax[0].imshow(to_denoise, cmap = 'Greys_r')
ax[1].imshow(denoised, cmap = 'Greys_r')
ax[2].imshow(denoised1 , cmap = 'Greys_r')
ax[0].set_title("Gaussian Noise")
ax[1].set_title("Closing by 5x5 Square")
ax[2].set_title("Closing by 9x9 Square")
for a in ax:
  a.axis('off')
plt.show() 