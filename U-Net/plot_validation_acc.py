import matplotlib.pyplot as plt 
import csv 
import pandas as pd
import numpy as np

fig = plt.figure()
fig.set_size_inches(5, 5)

for seed in range(3):
    
    x = np.arange(101) * 10
    y = np.arange(101)/100
    acc = pd.read_csv(f'L2ss_1_Seed_{seed}_Chan_8_ReLU_Smth_0.05_Set_4_LR_0.0001_BZ_1.csv')
    y[:len(acc['Dice'].tolist())] = acc['Dice'].tolist()
    
    plt.plot(x, y, label = f'seed={seed}') 
  
    plt.xticks(rotation = 25) 
    plt.xlabel('Epoch', fontsize = 10) 
    plt.ylabel('Dice', fontsize = 10) 
    plt.title('U-Net', fontsize = 16) 
    plt.grid() 
    plt.legend() 
    plt.show() 
plt.savefig('U-Net.png', bbox_inches="tight")