# Visualizer
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# """
# This function creates a heatmap using a 3D numpy array with labelled axis
# """
# def heatMapper(training_data, test_data, depth, width, epochs):
#     m = wdTuner(training_data, test_data, depth, width, epochs)
#     plt.imshow(m, cmap='hot', interpolation='nearest')
#     plt.show()
#     return "heheh"
def heatMapper(m):
    df = pd.DataFrame(m)
    plt.figure(figsize=(300, 100))
    sns.set(font_scale=0.25)
    sns.heatmap(df, vmax=300)
    plt.xlabel('Width',fontsize=22)
    plt.ylabel('Depth',fontsize=22)
    plt.show()

m = np.load("errorArray.npy")

m[np.isnan(m)] = 10^200
print(np.min(m))


#heatMapper(m)

