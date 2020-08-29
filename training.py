import os
import random
import shutil
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from imutils import paths
from keras.applications.mobilenet import preprocess_input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from keras.layers import Dense, Dropout
from keras.models import Model
from keras.optimizers import adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from CNN_model import model_ECG

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = [10, 5]
warnings.filterwarnings("ignore", category=FutureWarning)

dataset = 'dataset'

batch_size = 8
nb_epoch = 500

imagePaths = sorted(list(paths.list_images(dataset)))
random.seed(42)
random.shuffle(imagePaths)

embedded = []
labels = []

codeLabel = {0: 'normal', 1: 'AFib'}
labelCode = {}
for k, v in codeLabel.items():
    labelCode[v] = k

if not os.path.isdir("./models"):
    print("Creating directory to store models")
    os.mkdir("./models")

np.save('./models/labelName.npy', codeLabel)
for i, imagePath in enumerate(imagePaths):
    try:
        image = cv2.imread(imagePath, 1)
        image = image[..., ::-1]
        image = cv2.resize(image, (224, 224))
        image = (image / 255.).astype(np.float32)
        embedded.append(image)
    except (FileNotFoundError, Exception) as e:
        print(imagePath)
        pass

    label = imagePath.split(os.path.sep)[-2]
    labels.append(labelCode[label])

labels = np.array(labels)
print("Output data shape : ", labels.shape)

# splitting dataset
trainX, testX, trainY, testY = train_test_split(embedded, labels, test_size=0.2)

print(np.array(trainX).shape)
print(np.array(trainY).shape)

# Create the directory structure
base_dir = 'dataDir'
if not os.path.isdir(base_dir):
    print("Creating base_dir")
    os.mkdir(base_dir)

# train_dir
train_dir = os.path.join(base_dir, 'train_dir')
if not os.path.isdir(train_dir):
    print("Creating train_dir")
    os.mkdir(train_dir)

# val_dir
val_dir = os.path.join(base_dir, 'val_dir')
if not os.path.isdir(val_dir):
    print("Creating val_dir")
    os.mkdir(val_dir)

idx = 1
for image, label in zip(trainX, trainY):
    label_dir = os.path.join(train_dir, str(label))
    if not os.path.isdir(label_dir):
        os.mkdir(label_dir)
    dst = os.path.join(label_dir, "label_{}.png".format(idx))
    image = cv2.convertScaleAbs(image, alpha=(255.0))
    cv2.imwrite(dst, image)
    idx += 1

idx = 1
for image, label in zip(testX, testY):
    label_dir = os.path.join(val_dir, str(label))
    if not os.path.isdir(label_dir):
        os.mkdir(label_dir)
    dst = os.path.join(label_dir, "label_{}.png".format(idx))
    image = cv2.convertScaleAbs(image, alpha=(255.0))
    cv2.imwrite(dst, image)
    idx += 1

# Set Up the Generators
train_path = 'dataDir/train_dir'
val_path = 'dataDir/val_dir'

num_train_samples = len(trainX)
num_val_samples = len(testX)
train_batch_size = 5
val_batch_size = 5

train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)

datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = datagen.flow_from_directory(train_path, target_size=(224, 224), batch_size=train_batch_size)

val_gen = datagen.flow_from_directory(val_path, target_size=(224, 224), batch_size=val_batch_size)

test_gen = datagen.flow_from_directory(val_path, target_size=(224, 224), batch_size=1, shuffle=False)

# defining model
model = model_ECG()
model.summary()

print()
print("Total number of layers in Model :", len(model.layers))

x = model.layers[-6].output
x = Dropout(0.25)(x)
predictions = Dense(len(set(labels)), activation='softmax')(x)
models = Model(inputs=model.input, outputs=predictions)

models.summary()

for layer in models.layers[:-23]:
    layer.trainable = False

# Train the Model
models.compile(optimizer=adam(), loss='categorical_crossentropy', metrics=['accuracy'])
checkpoint = ModelCheckpoint("./models/model.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=7, verbose=1, mode='max', min_lr=0.00001)

early_stopper = EarlyStopping(monitor="val_acc", mode="max", patience=7)

csv_logger = CSVLogger(filename='training_log.csv', separator=',', append=False)

callbacks_list = [checkpoint, reduce_lr, early_stopper, csv_logger]

trainedModel = models.fit_generator(train_gen, steps_per_epoch=train_steps, validation_data=val_gen,
                                    validation_steps=val_steps, epochs=100, verbose=1, callbacks=callbacks_list)

# Get the best epoch from the training log
df = pd.read_csv('training_log.csv')

best_acc = df['val_acc'].max()

# display the row with the best accuracy
print()
display(df[df['val_acc'] == best_acc])

# plotting the results

plt.figure(figsize=(16, 5))
plt.subplot(121)
plt.plot(trainedModel.history['acc'])
plt.plot(trainedModel.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower right')
plt.subplot(122)
plt.plot(trainedModel.history['loss'])
plt.plot(trainedModel.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')

plt.show()

# Evaluate the model
models.load_weights('./models/model.h5')
_, accuracy = models.evaluate_generator(test_gen, steps=len(testX))
print('Accuracy :', accuracy)

# saving model to JSON
model_json = models.to_json()
with open("./models/CNN_ECG.json", "w") as json_file:
    json_file.write(model_json)

models.save_weights("./models/CNN_ECG.h5")

# Saving the model
print("Saving network...")
models.save("./models/CNN_ECG.model")

print("Saved model to disk")
print("Deleting unwanted folder")
shutil.rmtree("dataDir")
