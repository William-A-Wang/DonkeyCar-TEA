import cv2
import tensorflow as tf
from Adafruit_GPIO import I2C
import time
import numpy as np
import Adafruit_PCA9685
import donkeycar as dk
import curses


def get_bus():
    return PCA9685_I2C_BUSNUM


def gstreamer_pipeline(capture_width=3280, capture_height=2464, output_width=224, output_height=224, framerate=21, flip_method=0):   
        return 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=%d, height=%d, format=(string)NV12, framerate=(fraction)%d/1 ! nvvidconv flip-method=%d ! nvvidconv ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! videoconvert ! appsink' % (
                capture_width, capture_height, framerate, flip_method, output_width, output_height)


stdscr = curses.initscr()
curses.cbreak()
stdscr.keypad(True)
stdscr.nodelay(True)

PCA9685_I2C_BUSNUM = 1
PCA9685_I2C_ADDR = 0x40
I2C.get_default_bus = get_bus
PCA9685_I2C_ADDR = 0x40

pwm = Adafruit_PCA9685.PCA9685(address=PCA9685_I2C_ADDR)
pwm.set_pwm_freq(60)  # frequence of PWM

"""
You need to change these PWM values.
"""
STEERING_CHANNEL = 1
STEERING_LEFT_PWM = 290  # pwm value for full left steering
STEERING_RIGHT_PWM = 490  # pwm value for full right steering

THROTTLE_CHANNEL = 0
THROTTLE_FORWARD_PWM = 430  # pwm value for max forward throttle
THROTTLE_STOPPED_PWM = 370  # pwm value for no movement
THROTTLE_REVERSE_PWM = 300  # pwm value for max reverse throttle

# init steering and ESC
pwm.set_pwm(THROTTLE_CHANNEL, 0, int(THROTTLE_FORWARD_PWM))
time.sleep(0.1)
pwm.set_pwm(THROTTLE_CHANNEL, 0, int(THROTTLE_REVERSE_PWM))
time.sleep(0.1)
pwm.set_pwm(THROTTLE_CHANNEL, 0, int(THROTTLE_STOPPED_PWM))
time.sleep(0.1)


if __name__ == '__main__':
    w = 224
    h = 224
    running = True
    frame = None
    flip_method = 6
    capture_width = 3280
    capture_height = 2464
    framerate = 21

    camera = cv2.VideoCapture(
            gstreamer_pipeline(
                capture_width=capture_width,
                capture_height=capture_height,
                output_width=w,
                output_height=h,
                framerate=framerate,
                flip_method=flip_method),
            cv2.CAP_GSTREAMER)

    model_path = "Your Model"
    ret = True
    model = tf.keras.models.load_model(model_path, compile=False)
    threshold = 204
    time1 = time.time()
    count = 0
    throttle_all = []
    steering_all = []
    throttle_pwm = []
    steering_pwm = []
    new_throttle = 0
    steering = 0
    print("Running...")

    while True:
        time0 = time.time()
        ret, img_raw = camera.read()
        img = img_raw.astype(float)
        height = 80

        # Turn into grayscale
        img = np.dot(img[..., :3], [0.299, 0.587, 0.114])[height:224, :]
        out = model(img[None, :])

        throttle = new_throttle

        steering = out[0][0]

        throttle_all.append(throttle)
        steering_all.append(steering)

        if throttle > 0:
            throttle_pulse = dk.utils.map_range(
                throttle, 0, 1, THROTTLE_STOPPED_PWM, THROTTLE_FORWARD_PWM)
        else:
            throttle_pulse = dk.utils.map_range(
                throttle, -1, 0, THROTTLE_REVERSE_PWM, THROTTLE_STOPPED_PWM)

        steering_pulse = dk.utils.map_range(
            steering, -1, 1, STEERING_LEFT_PWM, STEERING_RIGHT_PWM)

        pwm.set_pwm(THROTTLE_CHANNEL, 0, int(throttle_pulse))
        pwm.set_pwm(STEERING_CHANNEL, 0, int(steering_pulse))

        count += 1
        c = stdscr.getch()
        if c == ord('q'):
            steering = 0.0
            steering_pulse = dk.utils.map_range(
                steering, -1, 1, STEERING_LEFT_PWM, STEERING_RIGHT_PWM)
            pwm.set_pwm(STEERING_CHANNEL, 0, int(steering_pulse))
            break
        if c == ord('w'):
            new_throttle = new_throttle + 0.02
        if c == ord('s'):
            new_throttle = new_throttle - 0.02

    curses.nocbreak()
    stdscr.keypad(False)
    curses.echo()
    curses.endwin()
    pwm.set_pwm(THROTTLE_CHANNEL, 0, int(THROTTLE_STOPPED_PWM))
    time2 = time.time()
    print("Running times: ", count)
    print('Time overall: ', time2 - time1)
    print('Average time: ', (time2 - time1) / count)
    print("SHUTDOWN...")
