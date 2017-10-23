# # -*- coding: utf-8 -*-
# import matplotlib.pyplot as plt
#
# if __name__ == "__main__":
#     data = {'Logistic Regression': 0.85, 'Naïve Bayes': 0.87, 'KNN': 0.82, 'RF': 0.91, 'DT': 0.85}
#     # x, height, *, align = 'center'
#     names = list(data.keys())
#     values = list(data.values())
#     plt.plot(names, values)
#
#     plt.show()

import matplotlib.pyplot as plt

data = {'apples': 10, 'oranges': 15, 'lemons': 5, 'limes': 20}
names = list(data.keys())
values = list(data.values())

fig, ax = plt.subplots()
ax.bar((1,2), names, 0,4)
fig.suptitle('Categorical Plotting')