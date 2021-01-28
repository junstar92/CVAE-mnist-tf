# -*- coding: utf-8 -*-
import tensorflow as tf

class CVAE(tf.keras.Model):
    def __init__(self, input_shape, label_shape, args):
        super(CVAE, self).__init__()
                
        # encoder
        self.encoder = gaussian_encoder(input_shape, label_shape, args)
        
        # z sampling layer
        self.sampling_layer = Sampling()
        
        # decoder
        self.decoder = bernoulli_decoder(args, label_shape, input_shape)
    
    def KLD_loss(self, mu, sigma):
        loss = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.math.log(1e-8 + tf.square(sigma)) - 1, 1)
        loss = tf.reduce_mean(loss)
        #loss = tf.square(mu) + tf.math.exp(sigma) - sigma - 1
        #loss = 0.5 * tf.reduce_mean(loss)
        
        return loss
    
    def BCE_loss(self, x, y):
        loss = tf.reduce_sum(x*tf.math.log(y) + (1-x)*tf.math.log(1-y), 1)
        
        return -tf.reduce_mean(loss)
    
    def call(self, inputs):
        x, y = inputs
        mu, sigma = self.encoder((x, y))
        z = self.sampling_layer((mu, sigma))        
        x_hat = self.decoder((z, y))
        x_hat = tf.clip_by_value(x_hat, 1e-8, 1-1e-8)
        
        # loss
        bce_loss = self.BCE_loss(x, x_hat)
        kld_loss = self.KLD_loss(mu, sigma)
        
        return x_hat, bce_loss, kld_loss

def gaussian_encoder(input_shape, label_shape, args):
    # initializer
    w_init = tf.keras.initializers.glorot_normal(args.seed)
    
    # input
    inputs_x = tf.keras.layers.Input(shape=input_shape)
    inputs_y = tf.keras.layers.Input(shape=label_shape)
    dim_y = label_shape
    
    #concat
    x = tf.keras.layers.concatenate([inputs_x, inputs_y], axis=1)
    
    # hidden layer
    x = tf.keras.layers.Dense(args.n_hidden+dim_y, activation='elu', 
                              kernel_initializer=w_init)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(args.n_hidden, activation='tanh',
                              kernel_initializer=w_init)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    
    # output
    mu = tf.keras.layers.Dense(args.dim_z, kernel_initializer=w_init)(x)
    sigma = tf.keras.layers.Dense(args.dim_z, kernel_initializer=w_init)(x)
    sigma = 1e-6 + tf.nn.softplus(sigma)
    
    encoder = tf.keras.Model([inputs_x, inputs_y], [mu, sigma])
    encoder.summary()
    
    return encoder

def bernoulli_decoder(args, label_shape, n_output):
    # initializer
    w_init = tf.keras.initializers.glorot_normal(args.seed)
    
    # input
    inputs_z = tf.keras.layers.Input(shape=args.dim_z)
    inputs_y = tf.keras.layers.Input(shape=label_shape)
    
    # concat
    x = tf.keras.layers.concatenate([inputs_z, inputs_y], axis=1)
    
    # hidden layer
    x = tf.keras.layers.Dense(args.n_hidden, activation='tanh',
                              kernel_initializer=w_init)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(args.n_hidden, activation='elu',
                              kernel_initializer=w_init)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    
    # output
    x = tf.keras.layers.Dense(n_output, activation='sigmoid',
                              kernel_initializer=w_init)(x)
    
    decoder = tf.keras.Model([inputs_z, inputs_y], x)
    decoder.summary()
    
    return decoder

class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        mu, sigma = inputs
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]

        z = mu + sigma * tf.random.normal((batch, dim), 0, 1)
        #z = mu + tf.math.exp(0.5*sigma) * tf.random.normal((batch, dim), 0, 1)
        return z