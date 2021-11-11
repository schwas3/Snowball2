import numpy as np
import numpy.random
import matplotlib.pyplot as plt

# Generate some test data
newImage = np.zeros((100,50))
y = np.rot90([np.arange(len(newImage))]*len(newImage[0]),3)
x = np.array([np.arange(len(newImage[0]))]*len(newImage))

heatmap, xedges, yedges = np.histogram2d(x.flatten(), y.flatten(), bins=(512,384),weights=newImage.flatten())
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.clf()
plt.imshow(heatmap.T, extent=extent, origin='lower')
plt.show()