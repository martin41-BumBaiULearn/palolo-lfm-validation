import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy import ndimage

print("Loading Oahu DEM for Pālolo Stream...")
with rasterio.open('USGS_13_n22w158.tif') as src:
    dem = src.read(1)

dem[dem < 0] = 0
print(f"DEM loaded. Shape: {dem.shape}")

print("Calculating Pālolo bottlenecks...")
filled = ndimage.grey_closing(dem, size=(3,3))
slope = ndimage.sobel(filled)
flow_accum = ndimage.gaussian_filter(slope, sigma=2)

bottlenecks = (flow_accum > np.percentile(flow_accum, 98)) & (slope > np.percentile(slope, 95))

print("Plotting results...")
plt.figure(figsize=(10, 8))
plt.imshow(dem, cmap='terrain')
plt.imshow(bottlenecks, cmap='Reds', alpha=0.6)
plt.title('Pālolo Stream LFM3 Bottlenecks - SF=12.58')
plt.savefig('palolo_bottlenecks.png', dpi=300)
print("Saved palolo_bottlenecks.png")
