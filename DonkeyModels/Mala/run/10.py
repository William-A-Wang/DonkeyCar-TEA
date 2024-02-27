import cv2
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import Convolution2D

from tensorflow.keras.layers import  Dropout, Flatten

from tensorflow.keras.models import Model

from Adafruit_GPIO import I2C
import time
import numpy as np
import time
#import cv2 have to put before the tensorflow

import Adafruit_PCA9685
import donkeycar as dk


import curses

stdscr = curses.initscr()
curses.cbreak()
stdscr.keypad(True)
stdscr.nodelay(True)

PCA9685_I2C_BUSNUM = 1
PCA9685_I2C_ADDR = 0x40

def get_bus():
    return PCA9685_I2C_BUSNUM


def gstreamer_pipeline(capture_width=3280, capture_height=2464, output_width=224, output_height=224, framerate=21, flip_method=0) :   
        return 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=%d, height=%d, format=(string)NV12, framerate=(fraction)%d/1 ! nvvidconv flip-method=%d ! nvvidconv ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! videoconvert ! appsink' % (
                capture_width, capture_height, framerate, flip_method, output_width, output_height)


I2C.get_default_bus = get_bus

pwm = Adafruit_PCA9685.PCA9685(address=PCA9685_I2C_ADDR)
pwm.set_pwm_freq(60) #frequence of PWM






STEERING_CHANNEL = 1
STEERING_LEFT_PWM = 290      #pwm value for full left steering
STEERING_RIGHT_PWM = 490        #pwm value for full right steering

THROTTLE_CHANNEL = 0 

THROTTLE_FORWARD_PWM = 430      #pwm value for max forward throttle
THROTTLE_STOPPED_PWM = 370      #pwm value for no movement
THROTTLE_REVERSE_PWM = 300      #pwm value for max reverse throttle

"""init steering and ESC"""
pwm.set_pwm(THROTTLE_CHANNEL,0,int(THROTTLE_FORWARD_PWM))
time.sleep(0.1)
pwm.set_pwm(THROTTLE_CHANNEL,0,int(THROTTLE_REVERSE_PWM))
time.sleep(0.1)
pwm.set_pwm(THROTTLE_CHANNEL,0,int(THROTTLE_STOPPED_PWM))
time.sleep(0.1)
#pwm.set_pulse(STEERING_CHANNEL,0,int(THROTTLE_STOPPED_PWM))

if __name__ == '__main__':

           
     w = 224
     h = 224
     running = True
     frame = None
     flip_method = 6
     #capture_width = 224 3280 820
     #capture_height = 224 2464 616
     capture_width = 3280
     capture_height = 2464
     framerate = 21
      
    
     camera = cv2.VideoCapture(
            gstreamer_pipeline(
                capture_width = capture_width,
                capture_height = capture_height,
                output_width=w,
                output_height=h,
                framerate=framerate,
                flip_method=flip_method),
            cv2.CAP_GSTREAMER)

     model_path = "/home/donkeycar1/mycar/models/10_new_gray_regression_add_flip_data_e3000_b20_mae_1out_Tue_Nov_14_09_52_11_2023.h5"
     ret = True   
     model = tf.keras.models.load_model(model_path
                                        , compile=False)
     
     threshold = 204
     #print(model.summary())
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
         #stdscr.addstr("runing times:" + str(count) + "\n")
         time0=time.time()
         ret , img = camera.read()

         #cv2.imwrite("/home/donkeycar1/img_BGR/"+ str(count) + '.jpg',img)
         #img = cv2.imread(img) #cause error
         #cv2.imwrite("/home/donkeycar1/img_RGB/"+ str(count) + '.jpg',img)
         img = img.astype(float)
         #height = 80 
         height = 80   
         # img = np.mean(img,axis = 2)[height:224,:]
         img = np.dot(img[..., :3],[0.299, 0.587, 0.114])[height:224,:]  # Turn into grayscale (more realistic)
         out = model(img[None,:])
         throttle = 0.48 #(out[0][0][0].numpy())
         #steering = np.argmax(out,axis = 1) #- 1
         steering = out[0][0]
   
         throttle_all.append(throttle)
         steering_all.append(steering)
         #cv2.imwrite("/home/donkeycar1/img_gray/"+ str(count) + str(steering) + '.jpg',img)

         if throttle>0:

             throttle_pulse = dk.utils.map_range(throttle, 0, 1,THROTTLE_STOPPED_PWM, THROTTLE_FORWARD_PWM)
         else:

             throttle_pulse = dk.utils.map_range(throttle, -1, 0,
                                            THROTTLE_REVERSE_PWM, THROTTLE_STOPPED_PWM)
                                            

         steering_pulse = dk.utils.map_range(steering,
                                        -1, 1,
                                        STEERING_LEFT_PWM,
                                        STEERING_RIGHT_PWM)
                                        
         #throttle_pwm.append(throttle_pulse)
         #steering_pwm.append(steering_pulse)
         pwm.set_pwm(THROTTLE_CHANNEL, 0, int(throttle_pulse))

         pwm.set_pwm(STEERING_CHANNEL, 0, int(steering_pulse))

         count += 1

         
         c = stdscr.getch()
         if c == ord('q'):
            break
         
     curses.nocbreak()
     stdscr.keypad(False)
     curses.echo()
     curses.endwin()   
     pwm.set_pwm(THROTTLE_CHANNEL, 0, int(THROTTLE_STOPPED_PWM))
     time2 = time.time()
     #np.save("outputs.npy",outputs)
     #np.save("throttle_all.npy", throttle_all)
     #np.save("steering_all.npy", steering_all)
     #np.save("throttle_pwm.npy", throttle_pwm)
     #np.save("steering_pwm.npy", steering_pwm)
     print("runing times: ", count)
     print('time overall: ', time2 - time1)
     print('average time: ', (time2 - time1)/(count))

     print("SHUTDOWN...")
    
    
