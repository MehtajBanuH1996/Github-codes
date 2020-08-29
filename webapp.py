from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
from datetime import datetime
from predict import prediction
import cv2
from twilio.rest import Client

app = Flask(__name__)
account_sid = ""
auth_token = ""
client = Client(account_sid, auth_token)


@app.route("/", methods=["POST", "GET"])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == "POST":
        global name
        global img
        if "button" in request.form:
            if request.form['button'] == "feedback":
                st = request.form['fb']
                if st == "yes":
                    fileName = os.path.join("dataset", name, datetime.today().strftime("%d%B_%H_%M") + '.png')
                else:
                    name = [i for i in ['normal', 'AFib'] if i != name][0]
                    fileName = os.path.join("dataset", name, datetime.today().strftime("%d%B_%H_%M") + '.png')
                cv2.imwrite(fileName, cv2.imread(img))
                return render_template('index.html')
        else:
            image = request.files.get('imgup')
            image.save('./' + secure_filename(image.filename))
            img = image.filename
            name, score = prediction(img)
            client.messages.create(body=f'Hello, This message is from ECG diagnostic module, your ECG signal '
                                        f'is/has {name}!', from_='whatsapp:+14155238886',
                                   to='whatsapp:+919789890416')
            kwargs = {'name': name, 'score': score}
            return render_template('index2.html', **kwargs)


if __name__ == '__main__':
    app.run()
