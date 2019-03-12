from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import ImageFile
import numpy as np
import pandas as pd
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Setting source directories
home_folder = './picnic dataset'
train_folder = home_folder + '/train'

# Getting base model (ResNet)
res_net_base_model = ResNet50(weights='imagenet', include_top=False)

# Read/import dataframe
train_df = pd.read_csv(home_folder + '/train.tsv', sep='\t')
label_names = list(set(train_df['label'].tolist()))
num_classes = len(label_names)
print(label_names)

# Add output layer to model
model_output = res_net_base_model.output
model_output = GlobalAveragePooling2D()(model_output)
model_output = Dense(num_classes,activation='softmax')(model_output)
model= Model(inputs=res_net_base_model.input, outputs=model_output)

# Set res_net_model layers to non-trainable
for layer in model.layers[:-1]:
    layer.trainable = False


train_data_generator = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess_input)
train_generator = train_data_generator.flow_from_dataframe(dataframe=train_df, directory=train_folder,x_col='file',
                                                           y_col='label', class_mode='categorical', target_size=(224,224),
                                                           batch_size=32)


model.summary()
# Compile the model using Adams optimizer
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

step_size_train=train_generator.n // train_generator.batch_size
# fit/ run the model using ONLY training data
model.fit_generator(generator=train_generator,
                    steps_per_epoch= step_size_train,
                   epochs=10)
