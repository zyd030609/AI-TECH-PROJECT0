try:
    import pygame
    print("Pygame successfully imported!")
except ImportError as e:
    print(f"Error importing pygame: {e}")

import random
import sys
import numpy as np
import matplotlib.pyplot as plt

# 打印Python环境信息
import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

# 打印已安装的包
import pkg_resources
print("\nInstalled packages:")
for package in pkg_resources.working_set:
    print(f"{package.key} - Version: {package.version}")

print("\nNumPy test:")
a = np.zeros((1,2))
print(a)