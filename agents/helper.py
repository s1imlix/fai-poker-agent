import numpy as np

def sigmoid(x, sharpness=5, midpoint=0.5):
    return 1 / (1 + np.exp(-sharpness * (x - midpoint)))