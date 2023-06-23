import numpy as np
from glob import glob
import scipy.io as sio
import os


if __name__ == "__main__":


    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--protocols", nargs='+', default=['802.11ax', '802.11b', '802.11n', '802.11g'],
                        help="Specify the protocols/classes to be included in the dataset")
    parser.add_argument('--dbm', nargs='+', default=['0', '10', '20', '30'])
    parser.add_argument('--raw_path', default='/home/miquelsirera/Desktop/dstl/data/RM_573C_power/', help='Path where raw signals are stored.')
    parser.add_argument('--postfix', default='', help='Postfix to append to dataset file.')
    args, _ = parser.parse_known_args()

    from scipy import signal
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

    for p in args.protocols:
        for dbm in args.dbm:
            filename = p + '_' + dbm if dbm != "" else p
            path = os.path.join(args.raw_path, filename)
            for ix, bin in enumerate(glob(os.path.join(path, '*.bin'))):
                if ix < 3:
                    s = np.fromfile(bin, dtype=np.complex128)
                    fs = 62.5e6 #20e6 we use a wider bandwidth here to collect data
                    f, t, Sxx = signal.spectrogram(s, fs=fs, nfft=1024)
                    th = Sxx.max() - 1e-10
                    Sxx[np.where(Sxx > th)] = th    # clip values for visualization po
                    """
                    plt.pcolormesh(t, f, Sxx, shading='gouraud')
                    plt.xlabel('Time [sec]')
                    plt.ylabel('Frequency [Hz]')
                    """
                    plt.pcolormesh(f, t, Sxx.T, shading='auto', cmap='seismic')
                    fc = 2.45e9
                    ch7 = 2.443e9 - fc
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


                    plt.ylabel('Time [sec]')
                    plt.xlabel('Frequency [Hz]')
                    plt.title('Protocols: '+str(p)+' '+os.path.basename(args.raw_path))
                    plt.colorbar()
                    #plt.show()
                    plt.savefig(os.path.join(path, p+'_'+os.path.basename(args.raw_path)+'_'+str(ix)+'.png'))

                    plt.vlines(x=0, ymin=0, ymax=max(t),
                               colors='red',
                               label='fc = 2.45e9')

                    plt.vlines(x=ch7, ymin=0, ymax=max(t),
                               colors=ch7_col, linestyles='dashed',
                               label='fc = 2.443e9 [Ch 7]')

                    plt.vlines(x=ch8, ymin=0, ymax=max(t),
                               colors=ch8_col, linestyles='solid',
                               label='fc = 2.447e9 [Ch 8]')
                    plt.vlines(x=ch9, ymin=0, ymax=max(t),
                               colors=ch9_col, linestyles='dashdot',
                               label='fc = 2.452e9[Ch 9]')

                    plt.vlines(x=ch10, ymin=0, ymax=max(t),
                               colors=ch10_col, linestyles='dotted',
                               label='fc = 2.457e9 [Ch 10]')

                    plt.legend()
                    plt.show()
                    plt.savefig(os.path.join(path, p + '_' + os.path.basename(args.raw_path) + '_' + str(ix) + '__lines.png'))
                    plt.clf()
                else:
                    break





