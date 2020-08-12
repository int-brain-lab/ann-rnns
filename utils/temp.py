import matplotlib.pyplot as plt
import numpy as np


x = np.random.normal(size=1000)
y = np.random.normal(size=1000)


# for color in ['#d73027', '#fc8d59', '#fee090', '#91bfdb', '#4575b4']:
#     plt.scatter(x, y, color=color)
#     plt.show()

for color in ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']:

    plt.scatter(x, y, color=color)
    plt.title(color)
    plt.show()
