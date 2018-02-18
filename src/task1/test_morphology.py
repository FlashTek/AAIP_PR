"""
    Tests the effects of the morphological operation (erison, dilation, opening, closing, gradient)
    on a sample image and plots the results
"""

import numpy as np
import matplotlib.pyplot as plt
import morphological_operations as mo
import structuring_elements as se
from PIL import Image
import matplotlib
from cycler import cycler

#plotting parameters
matplotlib.rc('text', usetex = True)
font = {'family':'serif','size':11}
plt.rc('font',**font)
plt.rc('axes', prop_cycle=(cycler('color', ['b', 'r', 'k', 'c', 'm', 'y']) ))
figsize = (6.29 , 3.54) # = 16*9 cm
dpi = 150

#load input image (MRI scan of a human brain)
im = np.asarray(Image.open("brain.jpg"))[:,50:-50,0]

#create structuring elements
square = se.square(5)
cross = se.cross(5)
circle = se.circle(5)

#prepare plots (set axis settings, etc.)
fig_mg, ax_mg = plt.subplots(1,4, figsize = (figsize[0], 2.5), dpi = dpi)
fig_cl, ax_cl = plt.subplots(1,4, figsize = (figsize[0], 2.5), dpi = dpi)
i = 1
ax_mg[0].imshow(im, cmap = 'Greys_r')
ax_cl[0].imshow(im, cmap = 'Greys_r')
ax_mg[0].axis('off')
ax_cl[0].axis('off')
ax_mg[0].set_title('original')
ax_cl[0].set_title('original')

#apply the gradient and opening operation for different structuring elements and plot the results
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
#save the image
fig_mg.savefig("morphological_gradient.pdf")

fig_cl.subplots_adjust(wspace = 0.01, left = 0.01, right= 0.99)
#save the image
fig_cl.savefig("morphological_closing.pdf")

fig_ed, ax_ed = plt.subplots(3,1, figsize = (figsize[0]/3 , figsize[1] * 1.7))

#plot erosion and dilation
ax_ed[0].imshow(im, cmap = "Greys_r",  vmin = 0, vmax = 255)
ax_ed[1].imshow(mo.erosion(im, square), cmap = "Greys_r",  vmin = 0, vmax = 255)
ax_ed[2].imshow(mo.dilation(im, square), cmap = "Greys_r", vmin = 0, vmax = 255)
for a in ax_ed:
    a.axis("off")
fig_ed.subplots_adjust(hspace = 0.02, top = 0.99, bottom = 0.01, right = 0.98, left = 0.02)

#save the resulting image
fig_ed.savefig("er_dil.pdf")

plt.show()
