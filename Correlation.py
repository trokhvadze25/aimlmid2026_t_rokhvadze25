import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

data = pd.read_csv('data.csv')
x = data['x']
y = data['y']

r, p_value = pearsonr(x, y)
print(f'Pearson correlation coefficient (r): {r:.4f}')
print(f'P-value: {p_value:.6f}')

plt.scatter(x, y)
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Scatter Plot of Blue Dot Data')
plt.grid(True)
plt.show()
