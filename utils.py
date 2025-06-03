import os
import glob
import shutil
import random
from tqdm import tqdm

def create_validation_data(train_dir, valid_dir, split= 0.2, ext='jpg'):
    """ Create validation dataset by splitting from training set
    """
    if not os.path.exists(valid_dir):
        os.mkdir(valid_dir)

    train_ds = glob.glob(train_dir + f'/*/*.{ext}')
    valid_sz = int(split * len(train_ds)) if split < 1.0 else split
    valid_ds = random.sample(train_ds, valid_sz)

    for fname in tqdm(valid_ds):
        basename = os.path.basename(fname)
        label = fname.split('\\')[-2]
        src_folder = os.path.join(train_dir, label)
        tgt_folder = os.path.join(valid_dir, label)
        if not os.path.exists(tgt_folder):
            os.mkdir(tgt_folder)
            
        shutil.move(os.path.join(src_folder, basename), os.path.join(tgt_folder, basename))

def imshow(inp, title=None):
    """Imshow for Tensor.
    """
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.axis('off')
    if title is not None:
        plt.title(title)