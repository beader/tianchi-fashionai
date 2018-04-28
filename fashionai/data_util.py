import os
import numpy as np
from skimage.io import imread
from skimage.util import pad as pad_image
from skimage.transform import resize as resize_image
from keras.utils.data_utils import Sequence


def extract_landmarks(landmark_data):
    """
    input is a list of list of landmark strings:
        [['1_1_1', '2_2_1', '3_3_0'],
         ['1_1_0', '2_2_1', '-1_-1_-1']]
    output is landmark coordinates and landmark visibilities:
        (
         np.array([[1, 1, 2, 2, 3, 3],
                   [1, 1, 2, 2, -1, -1]])
         ,
         np.array([[1, 1, 0],
                   [0, 1, -1]])
         )
    """
    xys, vs = [], []
    for record in landmark_data:
        r_xy, r_v = [], []
        for lm_str in record:
            x, y, v = lm_str.split('_')
            r_xy.append(int(x))
            r_xy.append(int(y))
            r_v.append(int(v))
        xys.append(r_xy)
        vs.append(r_v)
    return np.array(xys, dtype='float'), np.array(vs, dtype='float')


def scale_coordinates(data, source_shape, target_shape):
    """
    data is the form of:
    np.array([[x11,y11,x12,y12,...,x1d,y1d],
              [x21,y21,x22,y22,...,x2d,y2d],
              ...
              [xn1,yn1,xn2,yn2,...,xnd,ynd]])
    """
    assert isinstance(data, np.ndarray) and data.ndim == 2
    cord_x_idx = np.arange(data.shape[1], step=2)
    cord_y_idx = cord_x_idx + 1
    data[:, cord_x_idx] = data[:, cord_x_idx] * target_shape[0] / source_shape[0]
    data[:, cord_y_idx] = data[:, cord_y_idx] * target_shape[0] / source_shape[1]
    return data


class ImageData:
    """
    """

    def __init__(self, img_ids, img_dir,
                 source_shape=None,
                 target_shape=None,
                 padding='edge',
                 as_grey=False,
                 normalize=False):
        assert isinstance(source_shape, tuple), 'source_shape shoulb be a tuple'
        assert isinstance(target_shape, tuple), 'target_shape should be a tuple'
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.source_shape = source_shape[:2]
        self.target_shape = target_shape[:2]
        self.padding = padding
        self.as_grey = as_grey
        self.normalize = normalize

    def __len__(self):
        return len(self.img_ids)

    @property
    def shape(self):
        num_channels = 1 if self.as_grey else 3
        return (len(self),) + self.target_shape + (num_channels,)

    def _pad_image(self, img):
        pad_width = (
            (0, max(0, self.source_shape[0] - img.shape[0])),
            (0, max(0, self.source_shape[1] - img.shape[1])),
            (0, 0)
        )
        img = pad_image(img, pad_width, mode=self.padding)
        img = img[:self.source_shape[0], :self.source_shape[1]]
        return img

    def _resize_image(self, img, mode='constant'):
        return resize_image(img, self.target_shape, mode=mode)

    def _normalize_image(self, img):
        img = (img - np.mean(img, axis=(0, 1), keepdims=True)) / np.std(img, axis=(0, 1), keepdims=True)
        return img

    def _read_image(self, img_id):
        filename = os.path.join(self.img_dir, img_id)
        img = imread(filename, as_grey=self.as_grey)
        if len(img.shape) < 3:
            img = np.expand_dims(img, axis=-1)
        img = self._pad_image(img)
        if self.source_shape != self.target_shape:
            img = self._resize_image(img)
        if self.normalize:
            img = self._normalize_image(img)
        return img

    def __getitem__(self, indices):
        if isinstance(indices, np.ndarray) or isinstance(indices, slice):
            imgs = [self._read_image(img_id) for img_id in self.img_ids[indices]]
            imgs = np.array(imgs)
            return imgs
        else:
            return self._read_image(self.img_ids[indices])


class DataSequence(Sequence):
    def __init__(self, X, y=None, num_samples=None, batch_size=64, shuffle=True, global_indices=None):
        assert isinstance(num_samples, int), 'num_samples should be given'
        self.X = X
        self.y = y
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        if global_indices is None:
            self.global_indices = np.arange(num_samples)
        else:
            self.global_indices = global_indices
        if shuffle:
            np.random.shuffle(self.global_indices)

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.global_indices)

    def __getitem__(self, batch_id):
        batch_data_index = \
            self.global_indices[batch_id * self.batch_size: (batch_id + 1) * self.batch_size]
        if isinstance(self.X, tuple) or isinstance(self.X, list):
            batch_x = [x[batch_data_index] for x in self.X]
        else:
            batch_x = self.X[batch_data_index]

        if self.y is None:
            return batch_x

        if isinstance(self.y, tuple) or isinstance(self.y, list):
            batch_y = [y[batch_data_index] for y in self.y]
        else:
            batch_y = self.y[batch_data_index]

        return batch_x, batch_y


def train_test_split(X, y, batch_size=64, num_samples=None, validation_split=0.1):
    assert isinstance(num_samples, int), 'num_samples should be given'
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    num_val_samples = int(num_samples * validation_split)
    num_train_samples = num_samples - num_val_samples
    train_sample_indices = indices[num_val_samples:].copy()
    val_sample_indices = indices[:num_val_samples].copy()

    train_data = DataSequence(X, y,
                              num_samples=num_train_samples,
                              batch_size=batch_size,
                              global_indices=train_sample_indices,
                              shuffle=True)
    val_data = DataSequence(X, y,
                            num_samples=num_val_samples,
                            batch_size=batch_size,
                            global_indices=val_sample_indices,
                            shuffle=False)
    return train_data, val_data