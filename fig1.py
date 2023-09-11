import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter


fig, ax = plt.subplots()

model_name = ['ELMo', 'GPT', 'BERT', 'XLM', "GPT-2", "Megatron-LM", "T5", "Turing-NLG", "GPT-3", "PaLM",  ]
model_size = [0.009,   0.011, 0.340,  0.655,   1.5,     8.3,             11, 17.000,      175,   540 ]
generate_year = [2018, 2018, 2018, 2019,    2019,    2019,          2019,   2020,        2020,    2022]

bar_colors = ["#014C42", "#3B7A75", "#72B0A7", "#B4DBD6", "#E7EEEE", "#F4EBD5", "#DECC94", "#B98F4C", "#8B5C24", "#533411" ]

# ax.bar(model_name, model_size, color=bar_colors)

# ax.set_ylabel('Model Size (B)')
# plt.show()




for i, (generate_year_i, model_name_i, model_size_i, color) in enumerate(zip(generate_year, model_name, model_size, bar_colors)):
    ax.add_patch(plt.Circle((generate_year_i, model_size_i), radius=0.3, color=color))
ax.axis('equal')
ax.margins(0)

plt.show()