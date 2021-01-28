# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

def prepare_dataset(args):    
    # prepare dataset
    if args.dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif args.dataset == 'fashion-mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    input_shape = x_train[0].shape[0] * x_train[0].shape[1]
    label_shape = 10
    
    # reshape
    x_train = x_train.reshape(-1, input_shape)
    x_test = x_test.reshape(-1, input_shape)
    
    # one-hot encoding
    y_train_onehot = np.zeros((y_train.shape[0], 10))
    y_test_onehot = np.zeros((y_test.shape[0], 10))
    
    y_train_onehot[np.arange(y_train.shape[0]), y_train] = 1
    y_test_onehot[np.arange(y_test.shape[0]), y_test] = 1
    #y_train = tf.one_hot(y_train, depth=10)
    #y_test = tf.one_hot(y_test, depth=10)
    
    # normalization
    x_train = x_train / 255.
    x_test = x_test / 255.
    
    return x_train, y_train_onehot, x_test, y_test_onehot, input_shape, label_shape

class Plot_Reproduce_Result():
    def __init__(self, output_dir, n_img_x=10, n_img_y=10, img_w=28, img_h=28):
        self.dir = output_dir
        
        assert n_img_x > 0 and n_img_y > 0
        
        self.n_img_x = n_img_x
        self.n_img_y = n_img_y
        self.n_total = n_img_x * n_img_y
        
        assert img_w > 0 and img_h > 0
        
        self.img_w = img_w
        self.img_h = img_h
    
    def save_image(self, images, name='result.jpg'):
        images = images.reshape((-1, self.img_h, self.img_w))
        merged_img = self._merge(images)
        merged_img *= 255
        cv2.imwrite(self.dir + '/' + name, merged_img)
    
    def _merge(self, images):
        img = np.zeros((self.img_h * self.n_img_y, self.img_w * self.n_img_x))
        
        for idx, image in enumerate(images):
            i = int(idx / self.n_img_x)
            j = int(idx % self.n_img_x)
            
            img[i * self.img_h:i * self.img_h + self.img_h, 
                j * self.img_w:j * self.img_w + self.img_w] = image
        
        return img

class Plot_Manifold_Learning_Result():
    def __init__(self, output_dir, n_img_x=10, n_img_y=10, img_w=28, img_h=28, z_range=4):
        self.dir = output_dir
        
        assert n_img_x > 0 and n_img_y > 0
        
        self.n_img_x = n_img_x
        self.n_img_y = n_img_y
        self.n_total = n_img_x * n_img_y
        
        assert img_w > 0 and img_h > 0
        
        self.img_w = img_w
        self.img_h = img_h
        
        assert z_range > 0
        self.z_range = z_range
        
        self._set_latent_vectors()
        self.y = np.zeros(shape=[self.z.shape[0], 1])
        
    def _set_latent_vectors(self):
        z = np.rollaxis(np.mgrid[self.z_range:-self.z_range:self.n_img_y * 1j, self.z_range:-self.z_range:self.n_img_x * 1j], 0, 3)
        self.z = z.reshape([-1, 2])
    
    def save_image(self, images, name='result.jpg'):
        images = images.reshape((-1, self.img_h, self.img_w))
        merged_img = self._merge(images)
        merged_img *= 255
        cv2.imwrite(self.dir + '/' + name, merged_img)
    
    def _merge(self, images):
        img = np.zeros((self.img_h * self.n_img_y, self.img_w * self.n_img_x))
        
        for idx, image in enumerate(images):
            i = int(idx / self.n_img_x)
            j = int(idx % self.n_img_x)
            
            img[i * self.img_h:i * self.img_h + self.img_h, 
                j * self.img_w:j * self.img_w + self.img_w] = image
        
        return img
    
    def save_scattered_image(self, z, id, name='scattered_image.jpg'):
        N = 10
        plt.figure(figsize=(8, 6))
        plt.scatter(z[:, 0], z[:, 1], c=np.argmax(id, 1), marker='o', edgecolor='none', cmap=discrete_cmap(N, 'jet'))
        plt.colorbar(ticks=range(N))
        axes = plt.gca()
        axes.set_xlim([-self.z_range-2, self.z_range+2])
        axes.set_ylim([-self.z_range-2, self.z_range+2])
        plt.grid(True)
        plt.savefig(self.dir + "/" + name)

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

class Plot_Analogical_Reasoning_Result():
    def __init__(self, output_dir, dim_z, img_w=28, img_h=28, z_range=2):
        n_img_x = 11
        n_img_y = 4
        
        self.dir = output_dir
        
        assert n_img_x > 0 and n_img_y > 0 and dim_z > 0
        self.dim_z = dim_z
        self.n_img_x = n_img_x
        self.n_img_y = n_img_y
        self.n_total = n_img_x * n_img_y
        
        assert img_w > 0 and img_h > 0
        
        self.img_w = img_w
        self.img_h = img_h
        
        assert z_range > 0
        self.z_range = z_range
        
        self._set_latent_vectors()
    
    def _set_latent_vectors(self):
        # In order to maximize style-difference, let's peak 4 points
        assert self.n_img_y == 4 and self.dim_z == 2
        v = self.z_range * 0.7
        z = [[v, v],[-v, v], [v, -v], [-v, -v]]

        repeat_shape = list(np.int32(np.ones(self.n_img_y) * self.n_img_x))
        z = np.repeat(z,repeat_shape,axis=0)
        z = np.clip(z, -self.z_range, self.z_range)

        self.z = z
    
    def save_image(self, images, name='result.jpg'):
        images = images.reshape((-1, self.img_h, self.img_w))
        merged_img = self._merge(images)
        merged_img *= 255
        cv2.imwrite(self.dir + '/' + name, merged_img)
    
    def _merge(self, images):
        img = np.zeros((self.img_h * self.n_img_y, self.img_w * self.n_img_x))
        
        for idx, image in enumerate(images):
            i = int(idx / self.n_img_x)
            j = int(idx % self.n_img_x)
            
            img[i * self.img_h:i * self.img_h + self.img_h, 
                j * self.img_w:j * self.img_w + self.img_w] = image
        
        return img