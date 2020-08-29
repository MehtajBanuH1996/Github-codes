import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import load_model, model_from_json

CNNmodel = load_model('./models/CNN_ECG.model', compile=False)

graph = tf.get_default_graph()
model = model_from_json(open('./models/CNN_ECG.json').read())
model.load_weights('./models/model.h5')
labels_dict = {0: 'normal', 1: 'AFib'}


def display(img):
    plt.grid()
    plt.title("Original Query Image")
    plt.axis('off')
    plt.imshow(img)
    plt.show()


def prediction(img):
    print(img)
    img = cv2.imread(img,1)
    img = cv2.resize(img, (224, 224))
    orig = img.copy()
    #display(img)
    data = np.expand_dims(img, axis=0)
    global graph
    with graph.as_default():
        pred = model.predict(data, verbose=1)
    classInt = np.argmax(pred)
    name = labels_dict[classInt]
    output = imutils.resize(orig, width=400)
    out = "Label: {}".format(name)
    color = (255, 0, 0)
    cv2.putText(output, out, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    cv2.imwrite("output.jpeg", output)
    #plt.imshow(output)
    #plt.show()
    return name, max(pred[0])


path = "test.png"
# print(prediction(path))
