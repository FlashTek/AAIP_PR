import numpy as np
import matplotlib.pyplot as plt
import morphological_operations as mo
import structuring_elements as se
from PIL import Image
import matplotlib
from cycler import cycler

# plotting parameters

matplotlib.rc('text', usetex = True)
font = {'family':'serif','size':11}
plt.rc('font',**font)
figsize = (6.29 , 3.54) # = 16*9 cm
dpi = 150

# gaussian noise example

to_denoise = np.zeros((100, 100))  + np.random.randn(100,100)

to_denoise[50:] = to_denoise[50:] + 10

to_denoise = to_denoise - np.min(to_denoise)
SE = se.square(5)

denoised = mo.closing(to_denoise, SE)

SE1 = se.square(9)
denoised1 = mo.opening(to_denoise, SE)


fig, ax = plt.subplots(1,3, figsize = (figsize[0] , 2.2), dpi  = dpi)

ax[0].imshow(to_denoise, cmap = 'Greys_r')
ax[1].imshow(denoised, cmap = 'Greys_r')
ax[2].imshow(denoised1 , cmap = 'Greys_r')
ax[0].set_title("Gaussian Noise", fontsize = 11)
ax[1].set_title("Closing", fontsize = 11)
ax[2].set_title("Opening", fontsize = 11)

fig.subplots_adjust(left = 0.02, right = 0.98, hspace = 0.01)

for a in ax:
     a.axis('off')

#fig.savefig("closing_gauss.pdf")

to_denoise = np.zeros((100, 100))
to_denoise[50:] = 255

mask = np.random.random((100, 100))
mask = [mask > 0.8]

mask_ = np.random.random((100, 100))
mask_ = [mask_ > 0.8]

to_denoise[mask_] = 255 - np.random.rand(to_denoise[mask_].shape[0]) * 50
to_denoise[mask] = np.random.rand(to_denoise[mask].shape[0]) * 50

fig1, ax1 = plt.subplots(1,3, figsize = (figsize[0], 2.2), dpi = dpi)
fig1.subplots_adjust(left = 0.02, right = 0.98, hspace = 0.01)

ax1[0].imshow(to_denoise, cmap = 'Greys_r')
ax1[0].set_title("Original", fontsize = 11)
ax1[1].set_title("Closing", fontsize = 11)
ax1[1].imshow(mo.closing(to_denoise, SE), cmap = 'Greys_r')
ax1[2].set_title("Opening", fontsize = 11)
ax1[2].imshow(mo.opening(to_denoise, SE), cmap = 'Greys_r')

for a in ax1:
    a.axis('off')

fig1.savefig("sp_noise_example.pdf")

plt.show()
