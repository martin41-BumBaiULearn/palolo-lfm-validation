import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

print("Loading Oahu DEM for Pālolo Stream...")

# This reads the TIF without rasterio
img = Image.open('USGS_13_n22w158.tif')
dem = np.array(img)
dem[dem < 0] = 0
print(f"DEM loaded. Shape: {dem.shape}")

print("Calculating Pālolo bottlenecks...")

# 3x3 max filter instead of grey_closing
from numpy.lib.stride_tricks import sliding_window_view
padded = np.pad(dem, 1, mode='edge')
windows = sliding_window_view(padded, (3,3))
filled = windows.max(axis=(2,3))

# Simple slope instead of sobel
gy, gx = np.gradient(filled)
slope = np.sqrt(gx**2 + gy**2)

# 3x3 average instead of gaussian_filter  
padded_slope = np.pad(slope, 1, mode='edge')
windows_slope = sliding_window_view(padded_slope, (3,3))
flow_accum = windows_slope.mean(axis=(2,3))

bottlenecks = (flow_accum > np.percentile(flow_accum, 98)) & (slope > np.percentile(slope, 95))

print("Plotting results...")
plt.figure(figsize=(10, 8))
plt.imshow(dem, cmap='terrain')
plt.imshow(bottlenecks, cmap='Reds', alpha=0.6)
plt.title('Pālolo Stream LFM3 Bottlenecks - SF=12.58')
plt.savefig('palolo_bottlenecks.png', dpi=300)
print("Saved palolo_bottlenecks.png")
