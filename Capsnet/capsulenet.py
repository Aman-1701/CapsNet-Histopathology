# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 02:40:36 2021

@author: AMAN VERMA
"""
import numpy as np
import pandas as pd
from keras import layers, models, optimizers
from keras import backend as K
from sklearn.model_selection import train_test_split
from utils import plot_log
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from datagenerator import DataGenerator

K.set_image_data_format('channels_last')


def CapsNet(input_shape, n_class, routings):
    
    x = layers.Input(shape=input_shape)

    # Layers 1: Convolution Layers to extract low level features
    conv1 = layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='relu', name='Conv1')(x)
    conv2 = layers.Conv2D(filters=128, kernel_size=3, strides=2, activation='relu', name='Conv2')(conv1)
    conv3 = layers.Conv2D(filters=256, kernel_size=6, strides=2, activation='relu', name='Conv3')(conv2)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv3, dim_capsule=8, n_channels=32, kernel_size=6, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    classcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                             name='classcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    out_caps = Length(name='capsnet')(classcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([classcaps, y])  
    masked = Mask()(classcaps)  

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    return train_model, eval_model


def margin_loss(y_true, y_pred):
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
    return K.sum(L, axis=-1)


def train(model, train_gen, val_gen, args):
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})

    model.fit_generator(generator=train_gen,
                        epochs=args.epochs,
                        validation_data=val_gen,
                        callbacks=[log, tb, checkpoint, lr_decay],
                        use_multiprocessing=True,
                        workers=6,
                        verbose=1)

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    
    plot_log(args.save_dir + '/log.csv', show=True)

    return model


def load_generators(data_dir):
    # Parameters
    params = {'dim': (96,96),
              'batch_size': 100,
              'n_classes': 2,
              'n_channels': 3,
              'shuffle': True}
    
    # Data
    data = pd.read_csv(data_dir + 'train_labels.csv')
    train, val = train_test_split(data, test_size = 0.1, random_state=42)
    partition = {"train":list(train['id']), "validation":list(val['id'])}
    labels = dict(zip(data['id'], data['label']))

    train_dir = data_dir + "train/"

    # Generators
    train_gen = DataGenerator(partition['train'], labels, train_dir, **params)
    val_gen = DataGenerator(partition['validation'], labels, train_dir, **params)
    
    return train_gen, val_gen


if __name__ == "__main__":
    import os
    import argparse
    from keras import callbacks

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on Histopathologic Cancer Detection Dataset.")
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='results')
    parser.add_argument('--data_dir', default='../input/histopathologic-cancer-detection/')
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args([])
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train_gen, val_gen = load_generators(args.data_dir)

    # define model
    model, eval_model = CapsNet(input_shape=train_gen.shape[1:],
                                n_class=train_gen.n_classes,
                                routings=args.routings)
    model.summary()

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    train(model=model, train_gen=train_gen, val_gen=val_gen, args=args)