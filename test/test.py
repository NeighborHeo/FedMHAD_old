import numpy as np
import torch
import random
# 20 * 5 array random (0 ~ 1)
arr = torch.rand(20, 5)
print(arr)
_, predicted = torch.max(arr, 1)
print(predicted)