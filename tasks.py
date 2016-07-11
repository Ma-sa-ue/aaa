from invoke import task
from glob import glob
from data_util import HDF5Writer
import train4inv
from util import crop_resize
from tqdm import tqdm

MSCOCO_TRAIN_PATH = '/share/MSCOCO/32x32/train2014/'
IMAGE_SHAPE = (32, 32, 3)
HDF_FILE = './data/mscoco.hdf5'

@task
def prepare():
    lsun_images = glob(MSCOCO_TRAIN_PATH + '*.jpg')
    writer = HDF5Writer([len(lsun_images)] + list(IMAGE_SHAPE), HDF_FILE)
    for i, image_path in enumerate(tqdm(lsun_images)):
        cropped_img = crop_resize(image_path)
        assert cropped_img.shape == IMAGE_SHAPE, "IMG_SHAPE not matched %d: %s, %s" % (i, cropped_img.shape, image_path)
        writer.write(i, cropped_img)
        
    print "Done, bye"

@task
def train():
    train4inv.train()
    print "Done, bye"

## for debugging
if __name__ == '__main__':
#     train()
    prepare()
    
    