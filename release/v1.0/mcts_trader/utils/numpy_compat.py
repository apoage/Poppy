import warnings
import os

# Suppress noisy NumPy warnings (e.g. "The value of the smallest subnormal for <class 'numpy.float64'> type is zero")
warnings.filterwarnings("ignore", message="The value of the smallest subnormal")
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")

# Suppress Torch UserWarnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")