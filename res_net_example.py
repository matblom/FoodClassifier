from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import pandas as pd
from PIL import Image
import os

#model = ResNet50(weights='imagenet', include_top=False)
home_folder = 'picnic dataset'
train_folder = home_folder + '/train'

train_df = pd.read_csv(home_folder + '/train.tsv', sep='\t')

label_names = list(set(train_df['label'].tolist()))

print(label_names)

train_data = []
for file in train_df['file']:
    img = Image.open(train_folder + '/' + file)
    imgarray = np.array(img)
    train_data.append(imgarray)

print('done')
train_data


# for file in os.listdir(home_folder):
#     # img_path = 'The Picnic Hackathon 2019/train/5.jpeg'
#     img = image.load_img(home_folder + "/" + file, target_size=(224, 224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     preds = model.predict(x)
#     if np.amax(preds) > 0.9:
#         print('Printing: ',file)
#         print('Predicted:', decode_predictions(preds, top=3)[0])
