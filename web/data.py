import pandas as pd
import numpy as np

test_data = pd.read_csv('data/test.csv')
test_pixels = test_data.values
test_pixels = test_pixels / 255.0
test_pixels = test_pixels.reshape(-1, 28, 28, 1)
np.save('data/X_test_new.npy', test_pixels)
