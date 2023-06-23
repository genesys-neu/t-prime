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

    os.makedirs(os.path.join(args.raw_path, 'mat'), exist_ok=True)
    for p in args.protocols:
        for dbm in args.dbm:
            path = os.path.join(args.raw_path, p+'_'+dbm)
            for f in glob(os.path.join(path, '*.bin')):
                s = np.fromfile(f, dtype=np.complex128)
                mat_path = os.path.join(args.raw_path, 'mat', p + '_' + dbm)
                if not os.path.isdir(mat_path):
                    os.makedirs(mat_path)
                sio.savemat(os.path.join(args.raw_path, 'mat', p+'_'+dbm, os.path.basename(f)[:-4]+'.mat'), {'signal': s})







