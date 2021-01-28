# -*- coding: utf-8 -*-
import os
import glob

from cvae import CVAE
from utils import *

import tensorflow as tf
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--output', type=str, default='results', help='File path of output images')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'fashion-mnist'], help='The name of dataset')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    
    parser.add_argument('--add_noise', type=bool, default=False, help='boolean for adding noise to input image')
    parser.add_argument('--noise_factor', type=float, default=0.7, help='Factor of noise')
    parser.add_argument('--dim_z', type=int, default=2, help='Dimension of latent vector')#, required=True)
    parser.add_argument('--n_hidden', type=int, default=500, help='Number of hidden units in MLP')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate of adam optimizer')
    parser.add_argument('--num_epochs', type=int, default=20, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    
    parser.add_argument('--PRR_n_img_x', type=int, default=10, help='Number of images along x-axis')
    parser.add_argument('--PRR_n_img_y', type=int, default=10, help='Number of images along y-axis')
    parser.add_argument('--PMLR_n_img_x', type=int, default=20, help='Number of images along x-axis')
    parser.add_argument('--PMLR_n_img_y', type=int, default=20, help='Number of images along y-axis')
    parser.add_argument('--PMLR_z_range', type=float, default=2.0, help='Range for uniformly distributed latent vector')
    parser.add_argument('--PMLR_n_samples', type=int, default=5000, help='Number of samples in order to get distribution of labeled data')
    
    return check_args(parser.parse_args())
    
def check_args(args):
    # --output
    try:
        os.mkdir(args.output)
    except(FileExistsError):
        pass
    # delete all output files
    files = glob.glob(args.output + '/*')
    for file in files:
        os.remove(file)
    
    # --add_noise
    try:
        assert args.add_noise == True or args.add_noise == False
    except:
        print('add_noise must be boolean type')
        return None
    
    # --dim-z
    try:
        assert args.dim_z > 0
    except:
        print('dim_z must be positive interger')
    
    # --n_hidden
    try:
        assert args.n_hidden >= 1
    except:
        print('number of hidden units must be larger than or equal to one')
        
    # --learning_rate
    try:
        assert args.learning_rate > 0
    except:
        print('learning_rate must be positive')
    
    # --num_epochs
    try:
        assert args.num_epochs >= 1
    except:
        print('number of epochs must be larger than or equal to one')
    
    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    
    return args

def main(args):
    # dataset    
    x_train, y_train, x_test, y_test, input_shape, label_shape = prepare_dataset(args)
    
    n_samples = x_train.shape[0]
    
    # build model
    cvae = CVAE(input_shape, label_shape, args)
    
    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    
    # loss metric
    loss_metric = tf.keras.metrics.Mean()
        
    # train
    total_batch = int(n_samples / args.batch_size)
    
    # plot reproduce result
    PRR = Plot_Reproduce_Result(args.output, args.PRR_n_img_x, args.PRR_n_img_y)
    
    # add noise to input and save input images
    x_PRR = x_test[:PRR.n_total]
    y_PRR = y_test[:PRR.n_total]
    PRR.save_image(x_PRR, 'input.jpg')
    
    if args.add_noise:
        x_PRR = x_PRR + args.noise_factor * np.random.randn(*x_PRR.shape)
        
        PRR.save_image(x_PRR, name='input_noise.jpg')
    
    # plot manifold learning result
    if args.dim_z == 2:
        PMLR = Plot_Manifold_Learning_Result(args.output, args.PMLR_n_img_x, args.PMLR_n_img_y, z_range=args.PMLR_z_range)
        PARR = Plot_Analogical_Reasoning_Result(args.output, args.dim_z)
     
        x_PMLR = x_test[:args.PMLR_n_samples]
        id_PMLR = y_test[:args.PMLR_n_samples]
        
        # add noise
        if args.add_noise:
            x_PMLR = x_PMLR + args.noise_factor * np.random.randn(*x_PMLR.shape)
    
    for epoch in range(args.num_epochs):
        print(f'Start of epoch {epoch+1}')
        
        # random shuffling
        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices)
        x_train = x_train[indices]
        y_train = y_train[indices]
        
        for step in range(total_batch):
            # Compute the offset of the current minibatch in the data
            offset = (step * args.batch_size) % n_samples
            batch_x_train = x_train[offset:(offset + args.batch_size)]
            batch_y_train = y_train[offset:(offset + args.batch_size)]
            
            with tf.GradientTape() as tape:
                if args.add_noise:
                    pass
                
                x_hat, bce_loss, kld_loss = cvae([batch_x_train, batch_y_train])
                total_loss = bce_loss + kld_loss
            
            grads = tape.gradient(total_loss, cvae.trainable_weights)
            optimizer.apply_gradients(zip(grads, cvae.trainable_weights))
            
            loss_metric(total_loss.numpy())
            
            if step % 100 == 0:
                print(f'Epoch: {epoch+1} step: {step} mean loss = {loss_metric.result().numpy()}')
                print(f'BCE loss: {bce_loss}, KL loss: {kld_loss}')
        
        # save plot of reproduce result
        x_PRR_hat, _, _ = cvae([x_PRR, y_PRR])
        PRR.save_image(x_PRR_hat.numpy(), f'PRR_epoch_{epoch+1:02d}.jpg')
        
        if args.dim_z == 2:
            target_labels = [epoch % 10]
            
            # save plot of manifold learning result
            if epoch + 1 == args.num_epochs:
                target_labels = [0,1,2,3,4,5,6,7,8,9]
                
                for label in target_labels:
                    fake_id_PMLR = np.zeros(shape=[PMLR.z.shape[0],mnist_data.NUM_LABELS])
                    fake_id_PMLR[:, label] = 1.0
                    
                    x_hat_PMLR = cvae.decoder([PMLR.z, fake_id_PMLR])
                    PMLR.save_image(x_hat_PMLR.numpy(), f'PMLR_epoch_{epoch+1:02d}_{label:02d}.jpg')
            
            mu, sigma = cvae.encoder([x_PMLR, id_PMLR])
            z_PMLR = cvae.sampling_layer([mu, sigma])
            PMLR.save_scattered_image(z_PMLR, id_PMLR, name=f'PMLR_map_epoch_{epoch+1:02d}.jpg')
            
            # save plot of analogical reasoning result
            if epoch + 1 == args.num_epochs:
                fake_id_PARR = np.zeros(shape=[PARR.z.shape[0],mnist_data.NUM_LABELS])
                for i in range(PARR.z.shape[0]):
                    if i%11 == 0:
                        label = 3
                    else:
                        label = (i % 11) - 1
                    fake_id_PARR[i, label] = 1.0
                y_PAAR = cvae.decoder([PARR.z, fake_id_PARR])
                PARR.save_image(y_PAAR.numpy(), 'PAAR.jpg')

if __name__ == '__main__':
    # parse arguments
    args = parse_args()
    
    if args is None:
        exit()
    
    main(args)