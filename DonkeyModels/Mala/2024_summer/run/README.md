### README

---

# Self-Driving Car Model Runner

This script is designed to run a self-driving car model using a camera feed, and it allows control of the car's throttle and steering through a PWM controller. The script supports loading a pre-trained model and optionally converting images to grayscale for inference.

## Requirements

- Python 3.x
- TensorFlow
- OpenCV
- Adafruit_PCA9685
- DonkeyCar
- Curses
- Adafruit_GPIO

## Installation

Ensure that you have all the required Python packages installed. You can install them using pip:

```bash
pip install tensorflow opencv-python Adafruit_PCA9685 donkeycar
```

## Usage

To run the script, use the following command:

```bash
python your_script.py --model_path path/to/your/model.h5 [--gray]
```

### Command Line Arguments

- `--model_path`: **(required)** The path to the pre-trained model.
- `--gray`: **(optional)** If provided, the images will be converted to grayscale before being fed into the model.

### Example

To run the script with a model located at `models/my_model.h5` and without grayscale conversion:

```bash
python your_script.py --model_path models/my_model.h5
```

To run the script with a model located at `models/my_model.h5` and with grayscale conversion:

```bash
python your_script.py --model_path models/my_model.h5 --gray
```

## Controls

- **`q`**: Quit the script.
- **`w`**: Increase throttle.
- **`s`**: Decrease throttle.

## Script Explanation

### Functions

- `get_bus()`: Returns the I2C bus number.
- `gstreamer_pipeline()`: Returns the GStreamer pipeline string for capturing video.
- `main(args)`: Main function to set up and run the self-driving car model.

### Main Execution

1. **Initialization**: Sets up the curses screen and the PWM controller for the car.
2. **Camera Setup**: Initializes the camera using the GStreamer pipeline.
3. **Model Loading**: Loads the pre-trained model from the provided path.
4. **Image Processing**: Captures images from the camera, optionally converts them to grayscale, and performs inference using the model.
5. **Control Loop**: Reads key inputs for controlling throttle, processes the camera feed, and sends commands to the car's PWM controller.

### Cleanup

On quitting the script, it resets the car's throttle, shuts down the PWM controller, and exits the curses mode.

---

This README provides an overview of how to use and understand the script. For further details, refer to the comments within the script or the respective libraries' documentation.