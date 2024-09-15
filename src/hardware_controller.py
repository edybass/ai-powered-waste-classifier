"""
Hardware Interface for Raspberry Pi and Arduino
Real-time waste classification with servo control
"""

import cv2
import numpy as np
import time
from pathlib import Path
import RPi.GPIO as GPIO
import serial
from picamera2 import Picamera2
from src.waste_classifier import WasteClassifier
import threading
import queue

class HardwareController:
    """Control hardware components for waste sorting."""

    def __init__(self, arduino_port="/dev/ttyUSB0", baudrate=9600):
        # Initialize GPIO
        GPIO.setmode(GPIO.BCM)

        # LED pins for status indication
        self.led_pins = {
            'ready': 17,      # Green LED
            'processing': 27, # Yellow LED
            'error': 22,      # Red LED
            'recyclable': 23, # Blue LED
            'organic': 24,    # Brown LED
            'hazardous': 25   # Red LED
        }

        # Setup LEDs
        for pin in self.led_pins.values():
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)

        # Ultrasonic sensor pins
        self.trig_pin = 5
        self.echo_pin = 6
        GPIO.setup(self.trig_pin, GPIO.OUT)
        GPIO.setup(self.echo_pin, GPIO.IN)

        # Initialize Arduino connection for servo control
        try:
            self.arduino = serial.Serial(arduino_port, baudrate, timeout=1)
            time.sleep(2)  # Wait for Arduino to initialize
            print("✅ Arduino connected")
        except:
            print("⚠️  Arduino not connected, running in simulation mode")
            self.arduino = None

        # Initialize camera
        self.picam2 = Picamera2()
        camera_config = self.picam2.create_preview_configuration(
            main={"size": (1920, 1080)},
            lores={"size": (640, 480)},
            display="lores"
        )
        self.picam2.configure(camera_config)
        self.picam2.start()

        # Initialize classifier
        self.classifier = WasteClassifier(model_path="data/models/waste_classifier_mobilenet.h5")

        # Threading for continuous operation
        self.detection_queue = queue.Queue()
        self.running = False

    def measure_distance(self):
        """Measure distance using ultrasonic sensor."""
        # Send trigger pulse
        GPIO.output(self.trig_pin, GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(self.trig_pin, GPIO.LOW)

        # Measure echo response
        pulse_start = time.time()
        pulse_end = time.time()

        while GPIO.input(self.echo_pin) == 0:
            pulse_start = time.time()

        while GPIO.input(self.echo_pin) == 1:
            pulse_end = time.time()

        # Calculate distance
        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17150  # Speed of sound / 2
        distance = round(distance, 2)

        return distance

    def control_servo(self, bin_type):
        """Control servo to direct waste to appropriate bin."""
        servo_positions = {
            'recyclable': 0,
            'organic': 60,
            'hazardous': 120,
            'general': 180
        }

        if self.arduino:
            position = servo_positions.get(bin_type, 90)
            command = f"SERVO:{position}\n"
            self.arduino.write(command.encode())
            time.sleep(1)  # Wait for servo to move

            # Return to neutral position
            self.arduino.write(b"SERVO:90\n")
        else:
            print(f"[Simulation] Moving servo to {bin_type} position")

    def set_led(self, led_name, state):
        """Control LED states."""
        if led_name in self.led_pins:
            GPIO.output(self.led_pins[led_name], GPIO.HIGH if state else GPIO.LOW)

    def capture_and_classify(self):
        """Capture image and classify waste."""
        # Set processing LED
        self.set_led('processing', True)
        self.set_led('ready', False)

        # Capture image
        image = self.picam2.capture_array()

        # Convert to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Classify
        result = self.classifier.classify_image(image_bgr)

        # Set appropriate LED based on waste type
        self.set_led('processing', False)

        waste_type = result['type']
        if waste_type == 'recyclable':
            self.set_led('recyclable', True)
        elif waste_type == 'organic':
            self.set_led('organic', True)
        elif waste_type == 'hazardous':
            self.set_led('hazardous', True)

        # Control servo
        self.control_servo(waste_type)

        # Reset LEDs after delay
        time.sleep(2)
        for led in ['recyclable', 'organic', 'hazardous']:
            self.set_led(led, False)
        self.set_led('ready', True)

        return result

    def detection_loop(self):
        """Continuous detection loop."""
        self.set_led('ready', True)

        while self.running:
            # Check distance
            distance = self.measure_distance()

            # If object detected within 30cm
            if distance < 30:
                print(f"Object detected at {distance}cm")
                result = self.capture_and_classify()

                # Log result
                self.log_classification(result)

                # Wait before next detection
                time.sleep(3)

            time.sleep(0.1)

    def log_classification(self, result):
        """Log classification results."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} - {result['category']} ({result['confidence']:.2%}) - Bin: {result['bin_color']}\n"

        with open("logs/classifications.log", "a") as f:
            f.write(log_entry)

    def start(self):
        """Start the detection system."""
        self.running = True
        self.detection_thread = threading.Thread(target=self.detection_loop)
        self.detection_thread.start()
        print("🚀 Waste detection system started")

    def stop(self):
        """Stop the detection system."""
        self.running = False
        if hasattr(self, 'detection_thread'):
            self.detection_thread.join()

        # Cleanup
        self.picam2.stop()
        GPIO.cleanup()
        if self.arduino:
            self.arduino.close()

        print("🛑 System stopped")

    def calibrate_servo(self):
        """Calibrate servo positions."""
        if not self.arduino:
            print("Arduino not connected")
            return

        print("Starting servo calibration...")
        positions = [0, 45, 90, 135, 180]

        for pos in positions:
            input(f"Press Enter to move servo to {pos} degrees...")
            self.arduino.write(f"SERVO:{pos}\n".encode())
            time.sleep(1)

        self.arduino.write(b"SERVO:90\n")  # Return to center
        print("Calibration complete")

# Arduino sketch (save as waste_sorter.ino)
ARDUINO_CODE = """
#include <Servo.h>

Servo wasteServo;
const int servoPin = 9;

void setup() {
  Serial.begin(9600);
  wasteServo.attach(servoPin);
  wasteServo.write(90); // Center position
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');

    if (command.startsWith("SERVO:")) {
      int position = command.substring(6).toInt();
      position = constrain(position, 0, 180);
      wasteServo.write(position);
      Serial.println("OK:" + String(position));
    }
  }
}
"""

if __name__ == "__main__":
    # Create Arduino sketch file
    with open("hardware/waste_sorter.ino", "w") as f:
        f.write(ARDUINO_CODE)

    # Run hardware controller
    controller = HardwareController()

    try:
        # Calibrate first
        controller.calibrate_servo()

        # Start detection
        controller.start()

        # Run until interrupted
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down...")
        controller.stop()
