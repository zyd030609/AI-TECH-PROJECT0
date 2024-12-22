import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

# 定义数据
X = [10.31, 10.32, 10.33, 10.34, 10.35, 10.36, 10.37, 10.38, 10.39, 10.40]
UO = [0.005, 0.312, 1.183, 2.65, 4.57, 6.36, 8.64, 10.85, 12.37, 14.34]

# 进行二次多项式拟合
coefficients = np.polyfit(X, UO, 2)
poly = np.poly1d(coefficients)

# 生成更密集的点以绘制平滑的拟合曲线
X_fit = np.linspace(min(X), max(X), 100)
UO_fit = poly(X_fit)

# 创建图表
plt.figure(figsize=(10, 6))
plt.plot(X, UO, 'bo', label='实验数据')  # 原始数据点
plt.plot(X_fit, UO_fit, 'r-', label='拟合曲线')  # 拟合曲线
plt.xlabel('位移 X (mm)')
plt.ylabel('输出电压 UO (V)')
plt.title('光纤位移传感器的位移特性曲线')
plt.grid(True)
plt.legend()

# 打印拟合方程
print("拟合方程：")
print(f"UO = {coefficients[0]:.2f}x^2 + {coefficients[1]:.2f}x + {coefficients[2]:.2f}")

plt.show()