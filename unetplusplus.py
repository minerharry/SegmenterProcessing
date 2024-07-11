from keras_unet_collection._model_unet_plus_2d import unet_plus_2d
from keras import Model
import keras

model:Model = unet_plus_2d((None,None,1),filter_num = [32, 64, 128, 256, 512],n_labels=3,stack_num_down=4,stack_num_up=4,deep_supervision=False);
import numpy as np
from skimage.io import imread
inp = np.array([imread(r"C:\Users\Harrison Truscott\OneDrive - University of North Carolina at Chapel Hill\Bear Lab\matt_data\2023.06.07\images\p_s1_1.tif")[:512,:512],imread(r"C:\Users\Harrison Truscott\OneDrive - University of North Carolina at Chapel Hill\Bear Lab\matt_data\2023.06.07\images\p_s1_1.tif")[:512,:512]])
inp = np.reshape(inp,(*inp.shape,1))
print(inp.shape)
outp = model.predict(inp)

print(outp[0][0].shape)


# # import IPython
# # IPython.embed()
# levels = [f"sup{i}" for i in range(len(outp)-1)] + ["final"]
# import matplotlib.pyplot as plt
# f,axs = plt.subplots(outp[0].shape[-1],len(outp))
# for col,level in enumerate(outp):
#     for row in range(level.shape[-1]):
#         axs[row][col].imshow(level[0,:,:,row])
#         axs[row][col].set_title(f"level output_{levels[col]}, class {row}")
# f.show()


model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

# callbacks = [
#     keras.callbacks.ModelCheckpoint("oxford_segmentation.h5", save_best_only=True)
# ]

# # Train the model, doing validation at the end of each epoch.
# epochs = 15
# model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)

import IPython
IPython.embed()
# from skimage.transform import resize
# import tensorflow as tf
# import keras.backend as K
# from keras.losses import binary_crossentropy

# from keras import Model
# from keras.callbacks import  ModelCheckpoint
# from keras.layers import LeakyReLU
# from keras.layers import ZeroPadding2D
# from keras.layers import Add
# from keras.losses import binary_crossentropy
# from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout,BatchNormalization
# from keras.layers import Conv2D, Concatenate, MaxPooling2D
# from keras.layers import UpSampling2D, Dropout, BatchNormalization

# from keras.applications.efficientnet import EfficientNetB4


# def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
#     x = Conv2D(filters, size, strides=strides, padding=padding)(x)
#     x = BatchNormalization()(x)
#     if activation == True:
#         x = LeakyReLU(alpha=0.1)(x)
#     return x

# def residual_block(blockInput, num_filters=16):
#     x = LeakyReLU(alpha=0.1)(blockInput)
#     x = BatchNormalization()(x)
#     blockInput = BatchNormalization()(blockInput)
#     x = convolution_block(x, num_filters, (3,3) )
#     x = convolution_block(x, num_filters, (3,3), activation=False)
#     x = Add()([x, blockInput])
#     return x

# def UEfficientNet(input_shape:tuple[int|None,int|None,int]=(None, None, 3),input_tensor=None,dropout_rate=0.1):

#     if input_tensor is None:
#         input_tensor = Input(input_shape)

#     if input_shape[2] != 3:
#         print("different # of channels detected; convolving to 3 channels for efficientnet reasons")
#         input_tensor = Conv2D(3,input_shape[2],input_shape=input_shape)(input_tensor)

#     backbone = EfficientNetB4(weights='imagenet',
#                             include_top=False,
#                             input_shape=(input_shape[0],input_shape[1],3))
#     start_neurons = 8

#     conv4 = backbone.layers[342].output
#     conv4 = LeakyReLU(alpha=0.1)(conv4)
#     pool4 = MaxPooling2D((2, 2))(conv4)
#     pool4 = Dropout(dropout_rate)(pool4)
    
#      # Middle
#     convm = Conv2D(start_neurons * 32, (3, 3), activation=None, padding="same",name='conv_middle')(pool4)
#     convm = residual_block(convm,start_neurons * 32)
#     convm = residual_block(convm,start_neurons * 32)
#     convm = LeakyReLU(alpha=0.1)(convm)
    
#     deconv4 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(convm)
#     deconv4_up1 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(deconv4)
#     deconv4_up2 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(deconv4_up1)
#     deconv4_up3 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(deconv4_up2)
#     uconv4 = concatenate([deconv4, conv4])
#     uconv4 = Dropout(dropout_rate)(uconv4) 
    
#     uconv4 = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(uconv4)
#     uconv4 = residual_block(uconv4,start_neurons * 16)
# #     uconv4 = residual_block(uconv4,start_neurons * 16)
#     uconv4 = LeakyReLU(alpha=0.1)(uconv4)  #conv1_2
    
#     deconv3 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(uconv4)
#     deconv3_up1 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(deconv3)
#     deconv3_up2 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(deconv3_up1)
#     conv3 = backbone.layers[154].output
#     uconv3 = concatenate([deconv3,deconv4_up1, conv3])    
#     uconv3 = Dropout(dropout_rate)(uconv3)
    
#     uconv3 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv3)
#     uconv3 = residual_block(uconv3,start_neurons * 8)
# #     uconv3 = residual_block(uconv3,start_neurons * 8)
#     uconv3 = LeakyReLU(alpha=0.1)(uconv3)

#     deconv2 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv3)
#     deconv2_up1 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(deconv2)
#     conv2 = backbone.layers[92].output
#     uconv2 = concatenate([deconv2,deconv3_up1,deconv4_up2, conv2])
        
#     uconv2 = Dropout(0.1)(uconv2)
#     uconv2 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv2)
#     uconv2 = residual_block(uconv2,start_neurons * 4)
# #     uconv2 = residual_block(uconv2,start_neurons * 4)
#     uconv2 = LeakyReLU(alpha=0.1)(uconv2)
    
#     deconv1 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv2)
#     conv1 = backbone.layers[30].output
#     uconv1 = concatenate([deconv1,deconv2_up1,deconv3_up2,deconv4_up3, conv1])
    
#     uconv1 = Dropout(0.1)(uconv1)
#     uconv1 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv1)
#     uconv1 = residual_block(uconv1,start_neurons * 2)
# #     uconv1 = residual_block(uconv1,start_neurons * 2)
#     uconv1 = LeakyReLU(alpha=0.1)(uconv1)
    
#     uconv0 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv1)   
#     uconv0 = Dropout(0.1)(uconv0)
#     uconv0 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv0)
#     uconv0 = residual_block(uconv0,start_neurons * 1)
# #     uconv0 = residual_block(uconv0,start_neurons * 1)
#     uconv0 = LeakyReLU(alpha=0.1)(uconv0)
    
#     uconv0 = Dropout(dropout_rate/2)(uconv0)
#     output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv0)    
    
#     model = Model(input_tensor, output_layer)
#     model.name = 'u-xception'

#     return model

# model = UEfficientNet((None,None,1))

# import IPython
# IPython.embed()