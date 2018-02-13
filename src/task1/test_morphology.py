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


im = np.asarray(Image.open("brain.jpg"))[:,50:-50,0]

square = se.square(5)
cross = se.cross(5)
circle = se.circle(5)


fig_mg, ax_mg = plt.subplots(1,4, figsize = figsize, dpi = dpi)
fig_cl, ax_cl = plt.subplots(1,4, figsize = figsize, dpi = dpi)
i = 1
ax_mg[0].imshow(im, cmap = 'Greys_r')
ax_cl[0].imshow(im, cmap = 'Greys_r')
ax_mg[0].axis('off')
ax_cl[0].axis('off')
ax_mg[0].set_title('original')
ax_cl[0].set_title('original')
for se, name  in zip([square, cross, circle], ["square", "cross", "circle"]):
  
  mg = mo.morphological_gradient(im ,se)
  cl = mo.opening(im, se)
  ax_mg[i].imshow(mg, cmap = 'Greys_r')
  ax_mg[i].set_title(name)
  ax_mg[i].axis('off')
  
  ax_cl[i].imshow(cl, cmap = 'Greys_r')
  ax_cl[i].set_title(name)
  ax_cl[i].axis('off')
  
  i+=1
  
fig_mg.subplots_adjust(wspace = 0.01, left = 0.01, right= 0.99)
fig_mg.savefig("morphological_gradient.pdf")

fig_cl.subplots_adjust(wspace = 0.01, left = 0.01, right= 0.99)
fig_cl.savefig("morphological_opening.pdf")


plt.show()