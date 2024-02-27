import cv2
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import Convolution2D

from tensorflow.keras.layers import Dropout, Flatten

from tensorflow.keras.models import Model

from Adafruit_GPIO import I2C
import time
import numpy as np
import time

# import cv2 have to put before the tensorflow

import Adafruit_PCA9685
import donkeycar as dk


import curses

# stdscr = curses.initscr()
# curses.cbreak()
# stdscr.keypad(True)
# stdscr.nodelay(True)

PCA9685_I2C_BUSNUM = 1
PCA9685_I2C_ADDR = 0x40


def get_bus():
    return PCA9685_I2C_BUSNUM


def gstreamer_pipeline(
    capture_width=3280,
    capture_height=2464,
    output_width=224,
    output_height=224,
    framerate=21,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=%d, height=%d, format=(string)NV12, framerate=(fraction)%d/1 ! nvvidconv flip-method=%d ! nvvidconv ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! videoconvert ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            output_width,
            output_height,
        )
    )


I2C.get_default_bus = get_bus

pwm = Adafruit_PCA9685.PCA9685(address=PCA9685_I2C_ADDR)
pwm.set_pwm_freq(60)  # frequence of PWM


STEERING_CHANNEL = 1
STEERING_LEFT_PWM = 500  # pwm value for full left steering
STEERING_RIGHT_PWM = 320  # pwm value for full right steering
CENTER_PWM = 390

# Gain values
kp = 0.36
ki = 40
kd = 0.0009

pos = 128  # Middle point: 256/2
error = 0
integral_error = 0
error_last = 0
derivative_error = 0
output = 0
TIME_STEP = 0.001
windup_guard = 50

THROTTLE_CHANNEL = 0

THROTTLE_FORWARD_PWM = 440  # pwm value for max forward throttle
THROTTLE_STOPPED_PWM = 370  # pwm value for no movement
THROTTLE_REVERSE_PWM = 300  # pwm value for max reverse throttle

"""init steering and ESC"""
pwm.set_pwm(THROTTLE_CHANNEL, 0, int(THROTTLE_FORWARD_PWM))
time.sleep(0.1)
pwm.set_pwm(THROTTLE_CHANNEL, 0, int(THROTTLE_REVERSE_PWM))
time.sleep(0.1)
pwm.set_pwm(THROTTLE_CHANNEL, 0, int(THROTTLE_STOPPED_PWM))
time.sleep(0.1)
# pwm.set_pulse(STEERING_CHANNEL,0,int(THROTTLE_STOPPED_PWM))

if __name__ == "__main__":
    w = 224
    h = 224
    running = True
    frame = None
    flip_method = 6
    # capture_width = 224 3280 820
    # capture_height = 224 2464 616
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
            flip_method=flip_method,
        ),
        cv2.CAP_GSTREAMER,
    )

    ret = True

    threshold = 202
    # print(model.summary())
    time1 = time.time()
    outputs = []
    count = 0
    throttle_all = []
    steering_all = []
    throttle_pwm = []
    steering_pwm = []
    print("runing...")
    steering = 0
    while True:
        # stdscr.addstr("runing times:" + str(count) + "\n")
        time0 = time.time()
        ret, image = camera.read()

        # calculate the difference
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # HSV
        lower_orange = np.array([0, 100, 100])  # low
        upper_orange = np.array([30, 255, 255])  # high

        orange_mask = cv2.inRange(hsv_image, lower_orange, upper_orange)

        kernel = np.ones((5, 5), np.uint8)
        dilated_mask = cv2.dilate(orange_mask, kernel, iterations=1)
        clean_mask = cv2.erode(dilated_mask, kernel, iterations=1)

        result_image = cv2.bitwise_and(image, image, mask=clean_mask)

        # turn to white
        result_image[np.where(result_image > 0)] = 255

        center_row = int(result_image.shape[0] / 2)

        # window
        half_window_height = 5
        start_row = center_row - half_window_height
        end_row = center_row + half_window_height

        white_pixel_coords = []

        for row in range(start_row, end_row):
            for col in range(result_image.shape[1]):
                if np.array_equal(result_image[row, col], [255, 255, 255]):
                    white_pixel_coords.append((col, row))

        # drop
        for coord in white_pixel_coords:
            result_image[coord[1], coord[0]] = [0, 0, 255]  # RGB for red

        cv2.line(
            result_image,
            (0, center_row),
            (result_image.shape[1], center_row),
            (0, 255, 0),
            2,
        )

        if len(white_pixel_coords) > 0:
            avg_x = sum([coord[0] for coord in white_pixel_coords]) / len(
                white_pixel_coords
            )
            avg_y = sum([coord[1] for coord in white_pixel_coords]) / len(
                white_pixel_coords
            )
        else:
            avg_x = 0
        print(f"Measured Value (Average coordinates of white pixels): X: {avg_x}")

        # # set point
        # set_point = int(image.shape[1] / 2)
        #
        # cv2.circle(result_image, (set_point, center_row), 10, (0, 255, 0), -1)
        # cv2.putText(result_image, "Set Point", (set_point + 15, center_row), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # PID Control
        target = avg_x
        # Need to tune ki kp kd gains
        error = target - pos
        # self.integral_error += self.error * TIME_STEP
        derivative_error = (error - error_last) / TIME_STEP
        error_last = error
        output = kp * error + ki * integral_error + kd * derivative_error
        integral_error += error * TIME_STEP

        if integral_error > windup_guard:
            integral_error = windup_guard
        elif integral_error < windup_guard:
            integral_error = -windup_guard

        steering_value = output

        if steering_value <= -1980 and steering_value >= -2020:
            mapped_steering_angle = 0
        elif steering_value < -2020:
            mapped_steering_angle = 1
        elif steering_value > -1980:
            mapped_steering_angle = -1

        # error = avg_x - 224 / 2
        # Only trying the P controller
        # steering = error / 1124
        steering = mapped_steering_angle

        # cv2.imwrite("/home/donkeycar1/img_BGR/"+ str(count) + '.jpg',img)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cv2.imwrite("/home/donkeycar1/img_RGB/"+ str(count) + '.jpg',img)
        # img = img.astype(float)
        # height = 80
        # height = 80
        # img = np.mean(img,axis = 2)[height:224,:]
        # out = model(img[None,:])
        throttle = 0.3  # (out[0][0][0].numpy())

        # steering = np.argmax(out,axis = 1) - 1

        # if steering == 0:
        #    steering = steering + 1
        # else:
        #    steering = steering - 1

        throttle_all.append(throttle)
        steering_all.append(steering)
        # cv2.imwrite("/home/donkeycar1/img_gray/"+ str(count) + str(steering) + '.jpg',img)

        if throttle > 0:
            throttle_pulse = dk.utils.map_range(
                throttle, 0, 1, THROTTLE_STOPPED_PWM, THROTTLE_FORWARD_PWM
            )
        else:
            throttle_pulse = dk.utils.map_range(
                throttle, -1, 0, THROTTLE_REVERSE_PWM, THROTTLE_STOPPED_PWM
            )

        steering_pulse = dk.utils.map_range(
            steering, -1, 1, STEERING_LEFT_PWM, STEERING_RIGHT_PWM
        )

        # throttle_pwm.append(throttle_pulse)
        # steering_pwm.append(steering_pulse)
        pwm.set_pwm(THROTTLE_CHANNEL, 0, int(throttle_pulse))

        pwm.set_pwm(STEERING_CHANNEL, 0, int(steering_pulse))

        count += 1

    #     c = stdscr.getch()
    #     if c == ord('q'):
    #        break

    # curses.nocbreak()
    # stdscr.keypad(False)
    # curses.echo()
    # curses.endwin()
    pwm.set_pwm(THROTTLE_CHANNEL, 0, int(THROTTLE_STOPPED_PWM))
    time2 = time.time()
    # np.save("outputs.npy",outputs)
    # np.save("throttle_all.npy", throttle_all)
    # np.save("steering_all.npy", steering_all)
    # np.save("throttle_pwm.npy", throttle_pwm)
    # np.save("steering_pwm.npy", steering_pwm)

    print("runing times: ", count)
    print("time overall: ", time2 - time1)
    print("average time: ", (time2 - time1) / (count))

    print("SHUTDOWN...")
