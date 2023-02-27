import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
history= pd.read_csv('./AttentDenseUNet/2022-07-21_1924_1000/history_2000.csv')
N = len(history)
plt.figure(dpi= 300)

'''plt.plot(np.arange(0, N), history["loss"], label="Training Loss", linewidth= 1)
plt.plot(np.arange(0, N), history["val_loss"], label="Validation Loss", linewidth= 1)
#plt.plot(np.arange(0, N), history["val_loss"], label="Validation Loss", color='red', linewidth= 0.7)
plt.title('')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks([0, 400, 800, 1200, 1600, 2000])
plt.ylim(0, 0.015)
plt.yticks([0.000, 0.003, 0.006, 0.009, 0.012, 0.015])
plt.legend(loc='upper right')
plt.show()
'''
'''matplotlib.pyplot.figure(num=None, figsize=None (default: [6.4, 4.8], dpi=None (default: 100.0), facecolor=None, edgecolor=None, frameon=True, 
FigureClass=<class 'matplotlib.figure.Figure'>, clear=False, **kwargs)'''

def loss():
    plt.plot(np.arange(0, N), history["loss"], label="Training Loss", linewidth= 1)
    plt.plot(np.arange(0, N), history["val_loss"], label="Validation Loss", linewidth= 1)
    #plt.title('RA-Unet Model Loss', fontsize=10)
    #plt.xlabel('Epoch', fontsize=10)
    #plt.ylabel('Loss', fontsize=10)
    plt.xticks([0, 500, 1000, 1500, 2000])
    #plt.ylim(0, 0.03)
    plt.yticks([0, 0.010, 0.020, 0.030])
    plt.legend(loc='upper right')
    plt.show()

loss()