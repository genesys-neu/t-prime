import PIL
from PIL import Image, ImageChops
import os
import glob
import numpy as np
from tqdm import tqdm
import argparse


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = '',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dataset_folder_path', default='', type=str, help='')
    args = parser.parse_args()

    source_folder = 'images'
    destination_folder = 'cropped_images'

    source_path = os.path.join(os.path.abspath(args.dataset_folder_path), source_folder)
    save_path = os.path.join(os.path.abspath(args.dataset_folder_path), destination_folder)

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    file_list = glob.glob(source_path+'/*')

    print('********** Cropping the white margines in spectrograms **********')
    for source_path in tqdm(file_list):
        filename = source_path.split('/')[-1]
        im = Image.open(source_path)
        cropped_im = trim(im)
        image_array = np.array(cropped_im)[:,0:-1,:]
        cropped_im = Image.fromarray(image_array)
        # rotating a image 90 deg clockwise
        cropped_im = cropped_im.rotate(90, expand = 1)
        cropped_path = save_path + '/' + filename
        cropped_im.save(cropped_path)
        cropped_im.close()