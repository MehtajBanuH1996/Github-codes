import schedule
import time


def trainingFunc():
    print("Auto training process is started")
    exec(open('training.py').read())
    print("Training completed!!")


# schedule.every(10).minutes.do(trainingFunc)
schedule.every(2).days.do(trainingFunc)

while True:
    schedule.run_pending()
    time.sleep(1)
