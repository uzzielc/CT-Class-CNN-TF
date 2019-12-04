from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import random
import os
import sys, os, getopt

img_height = 256
img_width = 256
total_imgs = 200

def usage():
    print('\nusage: test.py')
    print('   -h : help')
    print('   --te : train model & evaluate')
    print('   --le : load (weights & params) & evaluate')
    print('   --lte : load (weights & params), train model, & evaluate\n')
    
def data_preprocess():
    # Load the labels 
    labels = pd.read_csv('labels.csv').to_numpy()

    # Create the data structure that will hold the images
    img_height = 256
    img_width = 256

    total_imgs = 200

    # imgDS will is initialized and will be used to store the image data 
    imgDS = np.empty((img_height,img_width,total_imgs))

    # load the image data in order & resize to make uniform
    for indx in range(total_imgs):
        if indx < 10:
            filename = 'head_ct/00'+str(indx)+'.png'
        elif indx < 100:
            filename = 'head_ct/0'+str(indx)+'.png'
        else:
            filename = 'head_ct/'+str(indx)+'.png'
        
        # Read and scale the image into img
        img = 1. * cv2.imread(filename,0) / 255
        # Make images a uniform size
        img2 = cv2.resize(img, dsize=(img_height, img_width))
        # Store the images in imgDS
        imgDS[:,:,indx] = img2

    # imgDS is now of size (img_height,img_width,200)

    # Combine the images and labels then shuffle data
    data = []
    for indx in range(total_imgs):
        data.append([imgDS[:,:,indx],labels[indx,1]])

    random.seed(1)
    random.shuffle(data)

    # ******************
    # Create the properly formatted DataStructure for use in the Tensorflow Model
    # ******************

    X = []
    Y = []
    for img,label in data:
        X.append(img)
        Y.append(label)

    # Convert back to numpy array
    # Note the shape is now (total_imgs, img_height, img_width,1)
    X = np.array(X).reshape(-1,img_height,img_width,1)
    Y = np.array(Y)

    train_data = X[:179,:,:]
    test_data = X[180:,:,:]
    train_labels = Y[:179]
    test_labels = Y[180:]

    image_gen_train = ImageDataGenerator(
                        rotation_range=10,
                        width_shift_range=.1,
                        height_shift_range=.1,
                        horizontal_flip=True,
                        zoom_range=0.1,
                        validation_split = 0.111
                        )
    train_data_gen = image_gen_train.flow(
        train_data,
        y=train_labels,
        batch_size=32,
        subset='training'
    )
    val_data_gen = image_gen_train.flow(
        train_data,
        y=train_labels,
        batch_size=32,
        shuffle=True,
        subset='validation'
    )
    return train_data_gen, val_data_gen, test_data, test_labels

def build_model():
    # Build the model
    model = Sequential([
        Conv2D(32, 3, activation='relu', input_shape=(img_height,img_width,1)),
        MaxPooling2D(2),
        Conv2D(32, 3, activation='relu'),
        MaxPooling2D(2),
        Conv2D(64, 3, activation='relu'),
        MaxPooling2D(2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    model.summary()
    # set up the path for the checkpoints (for loading and saving model parameters)
    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=False,
                                                    verbose=1)
    return model, cp_callback

def load_cp(model):
    # set up the path for the checkpoints (for loading and saving model parameters)
    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    print(model)
    model.load_weights(latest)
    return model
    
def train_model(model,train_data_gen,val_data_gen,cp_callback):
    # Fit the model to the data generators
    model.fit_generator(
        train_data_gen,
        steps_per_epoch=200,
        epochs=10,
        verbose=1,
        validation_data=val_data_gen,
        validation_steps=32,
        validation_freq=1,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
        shuffle=True,
        initial_epoch=0,
        callbacks=[cp_callback]
    )
    return model

def evaluate_model(model,test_data,test_labels):
    test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)


def main(argv):
    try:
        opts, args = getopt.getopt(argv, 'h',['le', 'te','lte'])
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(0)
    for option, arg in opts:
        if option == '-h':
            usage()
            sys.exit(0)
        else:
            if option == '--le':
                train_data_gen, val_data_gen, test_data, test_labels = data_preprocess()
                model, cp_callback = build_model()
                model = load_cp(model)
                evaluate_model(model,test_data,test_labels)
            if option == '--te':
                train_data_gen, val_data_gen, test_data, test_labels = data_preprocess()
                model, cp_callback = build_model()
                model = train_model(model,train_data_gen,val_data_gen,cp_callback)
                evaluate_model(model,test_data,test_labels)
            if option == '--lte':
                train_data_gen, val_data_gen, test_data, test_labels = data_preprocess()
                model, cp_callback = build_model()
                model = load_cp(model)
                model = train_model(model,train_data_gen,val_data_gen,cp_callback)
                evaluate_model(model,test_data,test_labels)


if __name__ == '__main__':
    # sys.argv = [<script name>, argv1, argv2, ...]
    main(sys.argv[1:])
