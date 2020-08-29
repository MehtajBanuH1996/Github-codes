import os
from datetime import datetime
from tkinter import *

import cv2
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
    img = cv2.imread(img)
    img = cv2.resize(img, (224, 224))
    display(img)
    data = np.expand_dims(img, axis=0)
    global graph
    with graph.as_default():
        pred = model.predict(data, verbose=1)
    classInt = np.argmax(pred)
    score = max(pred[0])
    name = labels_dict[classInt]
    s = "The patient has {} with {:.2f}% confidence".format(name, score * 100)
    return s, name


st = ""
root = Tk()
root.geometry("500x170")
root.winfo_toplevel().title("atrial fibrillation detection".title())
label1 = Label(root, text="Enter the path of the ECG file: ")
label1.pack(fill=X, pady=20)
label1.config(font=("Times New Roman", 20))
E1 = Entry(root, bd=5)


def getVal(variable, args):
    global st
    st = variable.get()
    args.destroy()


submit = Button(root, text="Submit", command=lambda: getVal(E1, root))
label1.pack()
E1.pack()
submit.pack(side=BOTTOM, pady=20)
mainloop()
imagePath = st
s, name = prediction(st)

color = "red"
if name == "normal":
    color = "green"
root2 = Tk()
root2.winfo_toplevel().title("atrial fibrillation detection".title())
root2.geometry("650x230")
label2 = Label(root2, text=s, wraplength=500, justify="center")
label2.config(font=("Courier", 25), fg=color)
label2.pack(fill=X, pady=20)

OPTIONS = [
    "yes",
    "no",
]

variable = StringVar(root2)
variable.set(OPTIONS[0])
E2 = OptionMenu(root2, variable, *OPTIONS)

done = Button(root2, text="Done", command=lambda: getVal(variable, root2))
label2.pack()
E2.pack()
done.pack(side=BOTTOM, pady=20)
mainloop()

print("Based on the response it will be added to the training dataset.")
print("Please retrain the model to have the effect of added images.")
if st == "yes":
    fileName = os.path.join("dataset", name, datetime.today().strftime("%d%B_%H_%M") + '.png')
else:
    name = [i for i in list(labels_dict.values()) if i != name][0]
    fileName = os.path.join("dataset", name, datetime.today().strftime("%d%B_%H_%M") + '.png')
cv2.imwrite(fileName, cv2.imread(imagePath))
