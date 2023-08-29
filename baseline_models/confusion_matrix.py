from glob import glob
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np

def plot_confmatrix(logdir, pkl_file, labels, figname):
    tot_samples = []
    correct_samples = []
    conf_mat = None
    cm = pickle.load(open(os.path.join(logdir, pkl_file), 'rb'))
    # print(cm)
    tot_s = np.sum(cm)
    correct_s = 0
    for r in range(cm.shape[0]):
        correct_s += cm[r, r]
    tot_samples.append(tot_s)
    correct_samples.append(correct_s)
    for r in range(cm.shape[0]):  # for each row in the confusion matrix
        sum_row = np.sum(cm[r, :])
        cm[r, :] = cm[r, :] / sum_row  # compute in percentage
        # also compute accuracy for each row
    if conf_mat is None:
        conf_mat = cm
    else:
        conf_mat += cm
    plt.clf()
    df_cm = pd.DataFrame(conf_mat, labels, labels)
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
    # plt.show()
    plt.savefig(os.path.join(logdir, figname))
    return df_cm

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True, help="Path (can use wildcard *) containing Ray's logs with custom confusion matrix validation output.")
    parser.add_argument("--pkl", default='conf_matrix.best.pkl', help="Pickle file name containing confusion matrix data")
    parser.add_argument("--figname", default='confusion_matrix.png', help="Name of confusion matrix image to be saved. NOTE: it will be saved in logdir")
    parser.add_argument("--labels", default=['802_11ax', '802_11b', '802_11n', '802_11g'], help="Labels for each class (in the order of ML model output)")
    args, _ = parser.parse_known_args()
     
    logdir = args.logdir
    pkl_file = args.pkl
    figname = args.figname
    labels = args.labels
    
    plot_confmatrix(logdir, pkl_file, labels, figname)
    
