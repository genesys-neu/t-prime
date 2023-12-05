import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

os.makedirs('plots', exist_ok=True)
test_path = './test_results/'

dir_list = os.listdir(test_path)

cnn_path = test_path + [s for s in dir_list if 'cnn' in s][0]
lg_path = test_path + [s for s in dir_list if 'lg' in s][0]
sm_path = test_path + [s for s in dir_list if 'sm' in s][0]
amcnet_path = test_path + [s for s in dir_list if 'amcnet' in s][0]
resnet_path = test_path + [s for s in dir_list if 'resnet' in s][0]
mcformer_path = test_path + [s for s in dir_list if 'mcformer' in s][0]

# Load the results
with open(lg_path, 'rb') as f:
    lg = pickle.load(f)
with open(sm_path, 'rb') as f:
    sm = pickle.load(f)
with open(cnn_path, 'rb') as f:
    cnn = pickle.load(f)
with open(resnet_path, 'rb') as f:
    resnet = pickle.load(f)
with open(amcnet_path, 'rb') as f:
    amcnet = pickle.load(f)
with open(mcformer_path, 'rb') as f:
    mcformer = pickle.load(f)

total_lg = np.sum(lg, axis=1)/len(lg.columns)
total_sm = np.sum(sm, axis=1)/len(sm.columns)
total_cnn = np.sum(cnn, axis=1)/len(cnn.columns)
total_resnet = np.sum(resnet, axis=1)/len(resnet.columns)
total_amcnet = np.sum(amcnet, axis=1)/len(amcnet.columns)
total_mcformer = np.sum(mcformer, axis=1)/len(mcformer.columns)

# Plot the results

for channel in lg.columns:
    plt.figure(figsize=(20, 8.5))
    plt.plot(lg[channel], label='LG Trans.', color='crimson', marker='^', markersize=20)
    plt.plot(sm[channel], label='SM Trans.', color='dodgerblue', marker='v', markersize=20)
    plt.plot(cnn[channel], label='CNN', color='goldenrod', marker='o', markersize=20)
    plt.plot(amcnet[channel], label='AMCNet', color='darkgreen', marker='P', linestyle='--', markersize=20)
    plt.plot(resnet[channel], label='ResNet', color='purple', marker='d', linestyle='--', markersize=20)
    plt.plot(mcformer[channel], label='MCFormer', color='mediumspringgreen', marker='X', linestyle='--', markersize=20)
    plt.title(channel, fontsize=54, fontweight='bold')
    plt.grid()
    plt.xticks(fontsize=46)
    plt.yticks(np.arange(0, 101, 50), fontsize=46)
    plt.xlabel('SNR(dBs)', fontsize=46, fontweight='bold')
    plt.ylabel('Accuracy(%)', fontsize=46, fontweight='bold')
    plt.legend(loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0), fontsize=14)
    plt.savefig(f'plots/{channel}.pdf', format="pdf", bbox_inches="tight")


# Plot the total results
plt.figure(figsize=(20, 8.5))
plt.plot(total_lg, label='LG Trans.', color='crimson', marker='^', markersize=20)
plt.plot(total_sm, label='SM Trans.', color='dodgerblue', marker='v', markersize=20)
plt.plot(total_cnn, label='CNN', color='goldenrod', marker='o', markersize=20)
plt.plot(total_amcnet, label='AMCNet', color='darkgreen', marker='P', linestyle='--', markersize=20)
plt.plot(total_resnet, label='ResNet', color='purple', marker='d', linestyle='--', markersize=20)
plt.plot(total_mcformer, label='MCFormer', color='mediumspringgreen', marker='X', linestyle='--', markersize=20)
plt.title('Total', fontsize=54, fontweight='bold')
plt.grid()
plt.xticks(fontsize=46)
plt.yticks(np.arange(0, 101, 20), fontsize=46)
plt.xlabel('SNR(dBs)', fontsize=46, fontweight='bold')
plt.ylabel('Accuracy(%)', fontsize=46, fontweight='bold')
plt.legend(loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0), fontsize=14)
plt.savefig(f'plots/total.pdf', format="pdf", bbox_inches="tight")


