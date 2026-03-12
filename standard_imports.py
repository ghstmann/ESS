import pandas as pd
from pandas.tseries.offsets import DateOffset
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import sys
import os
plt.style.use('fivethirtyeight')
sys.path.append('/Users/ghost/Desktop/Credit/code')

from pathlib import Path
code_path = Path('/', 'Users', 'ghost', 'Desktop', 'Credit', 'code')
data_path = Path('/', 'Users', 'ghost', 'Desktop', 'Credit', 'data')

print('\033[93mImporting pandas as pd, numpy as np, datetime as dt\033[0m')
print('\033[93mImporting matplotlib.pyplot as plt, plotly.graph_objects as go\033[0m')
print('\033[93mImporting sys, os\033[0m')

print('\033[93mImporting DateOffset from pandas.tseries.offsets\033[0m\n')

print('\033[93mSetting Matplotlib style to Five Thirty Eight\033[0m\n')

print('\033[93mAdding Code Snippet folder to sys.path\033[0m')

print(f'\033[93mAdding {code_path} as code_path using pathlib.\033[0m')
print(f'\033[93mAdding {data_path} as data_path using pathlib.\033[0m')