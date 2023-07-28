import numpy as np
from glob import glob
import scipy.io as sio
import os


if __name__ == "__main__":


    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--protocols", nargs='+', default=['802.11ax', '802.11b', '802.11n', '802.11g'],
                        help="Specify the protocols/classes to be included in the dataset")
    parser.add_argument("--fs", required=True, type=float, choices=[20e6, 62.5e6])
    parser.add_argument("--fc", required=True, type=float, choices=[2.442e9, 2.45e9])
    parser.add_argument('--dbm', nargs='+', default=['0', '10', '20', '30'])
    parser.add_argument('--raw_path', default='/home/miquelsirera/Desktop/dstl/data/RM_573C_power/', help='Path where raw signals are stored.')
    parser.add_argument('--postfix', default='', help='Postfix to append to dataset file.')
    args, _ = parser.parse_known_args()

    from scipy import signal
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

    max_plots = 20
    plt.rcParams.update({'font.size': 17})
    plt.rcParams['figure.figsize'] = [7.5, 5.5]
    for p in args.protocols:
        for dbm in args.dbm:
            filename = p + '_' + dbm if dbm != "" else p
            path = os.path.join(args.raw_path, filename)
            for ix, bin in enumerate(glob(os.path.join(path, '*.bin'))):
                if ix < max_plots:
                    s = np.fromfile(bin, dtype=np.complex128)
                    fs = args.fs

                    #f, t, Sxx = signal.spectrogram(s, fs=fs)
                    #th = Sxx.max() - 1e-10
                    #Sxx[np.where(Sxx > th)] = th    # clip values for visualization po

                    Sxx, freqs, t, im = plt.specgram(s, Fs=fs)
                    plt.xlabel("Time")
                    plt.ylabel("Frequency")

                    #plt.pcolormesh(f, t, Sxx.T, shading='auto', cmap='seismic')
                    fc = args.fc
                    ch7 = 2.442e9 - fc
                    ch8 = 2.447e9 - fc
                    ch9 = 2.452e9 - fc
                    ch10 = 2.457e9 - fc

                    ch7_col = ch8_col = ch9_col = ch10_col = 'purple'
                    if '25' in os.path.basename(args.raw_path):
                        ch7_col = 'yellow'
                        ch10_col = 'yellow'
                    elif '50' in os.path.basename(args.raw_path):
                        ch7_col = 'yellow'
                        ch9_col = 'yellow'

                    #plt.title('Protocols: '+str(p)+' '+os.path.basename(args.raw_path))

                    plt.colorbar()
                    #plt.show()
                    plt.savefig(os.path.join(path, p+'_'+os.path.basename(args.raw_path)+'_'+str(ix)+'.png'))

                    xmin = 1e-5
                    plt.hlines(y=0, xmin=xmin, xmax=max(t),
                               colors='red',
                               label='fc = '+"{:e}".format(fc))

                    if ch7 != 0:
                        plt.hlines(y=ch7, xmin=xmin, xmax=max(t),
                                   colors=ch7_col, linestyles='dashed',
                                   label='fc = 2.442e9 [Ch 7]')

                    plt.hlines(y=ch8, xmin=xmin, xmax=max(t),
                               colors=ch8_col, linestyles='solid',
                               label='fc = 2.447e9 [Ch 8]')
                    plt.hlines(y=ch9, xmin=xmin, xmax=max(t),
                               colors=ch9_col, linestyles='dashdot',
                               label='fc = 2.452e9[Ch 9]')
                    if fs > 20e6:
                        plt.hlines(y=ch10, xmin=xmin, xmax=max(t),
                                   colors=ch10_col, linestyles='dotted',
                                   label='fc = 2.457e9 [Ch 10]')


                    plt.legend(prop={'size': 12})

                    #plt.show()
                    plt.savefig(os.path.join(path, p + '_' + os.path.basename(args.raw_path) + '_' + str(ix) + '__lines.png'))
                    plt.clf()
                else:
                    break





