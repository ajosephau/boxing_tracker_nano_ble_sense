/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

//Arduino IDE version 1.8.13
//Arduino mbed-enabled board files version 2.0.0
//TensorFlow Lite for Microcontrollers Arudino library version 2.4.0-ALPHA
//Arduino_LSM9DS1 Arudino library version 1.1.0
//ArduinoBLE version Arudino library 1.2.0
//Havrard_TinyMLx Arudino library version 0.1.0

// https://github.com/arduino/ArduinoAI/blob/master/BLESense-test-dashboard/BLESense/BLESense.ino
#include <stdlib.h>

#include <TensorFlowLite.h>

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "magic_wand_model_data.h"
#include "rasterize_stroke.h"
#include "imu_provider.h"

#define BLE_SENSE_UUID(val) ("4798e0f2-" val "-4d68-af64-838352588e8e")

#define SIDE_L "L"    // Left
#define SIDE_C "C"    // Centre
#define SIDE_R "R"    // Right

#define SIDE SIDE_R
#define BLE_DEVICE_NAME(side) ("BoxingTracker-" side "-")

namespace {

  const int VERSION = 0x00000000;

  // Constants for image rasterization
  constexpr int raster_width = 32;
  constexpr int raster_height = 32;
  constexpr int raster_channels = 3;
  constexpr int raster_byte_count = raster_height * raster_width * raster_channels;
  int8_t raster_buffer[raster_byte_count];

  // BLE settings
  BLEService        service                       (BLE_SENSE_UUID("0000"));
  BLECharacteristic strokeCharacteristic          (BLE_SENSE_UUID("300a"), BLERead, stroke_struct_byte_count);
  BLECharacteristic accelerationCharacteristic    (BLE_SENSE_UUID("3001"), BLENotify, 3 * sizeof(float)); // Array of 3 floats, G
  BLECharacteristic gyroscopeCharacteristic       (BLE_SENSE_UUID("3002"), BLENotify, 3 * sizeof(float)); // Array of 3 floats, dps
  BLECharacteristic magneticFieldCharacteristic   (BLE_SENSE_UUID("3003"), BLENotify, 3 * sizeof(float)); // Array of 3 floats, uT

  // String to calculate the local and device name
  String name;
  
  // Create an area of memory to use for input, output, and intermediate arrays.
  // The size of this will depend on the model you're using, and may need to be
  // determined by experimentation.
  constexpr int kTensorArenaSize = 30 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
  
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  
  // -------------------------------------------------------------------------------- //
  // UPDATE THESE VARIABLES TO MATCH THE NUMBER AND LIST OF GESTURES IN YOUR DATASET  //
  // -------------------------------------------------------------------------------- //
  constexpr int label_count = 10;
  const char* labels[label_count] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};

  bool found_logger = false;
  bool scan_started = false;
  BLECharacteristic dataPeripheralCharacteristic;
  BLECharacteristic strokePeripheralCharacteristic;
  BLEDevice peripheral;
  BLEDevice central;

  int wheel_colour = 0;
}  // namespace

void setup() {
  // Start serial
  Serial.begin(115200);
  // while (!Serial) {}
  Serial.println("Started");

  // Start IMU
  if (!IMU.begin()) {
    Serial.println("Failed to initialized IMU!");
    while (1);
  }
  SetupIMU();

  // Start BLE
  if (!BLE.begin()) {
    Serial.println("Failed to initialized BLE!");
    while (1);
  }
  String address = BLE.address();

  // Output BLE settings over Serial
  Serial.print("address = ");
  Serial.println(address);

  address.toUpperCase();

  name = BLE_DEVICE_NAME(SIDE);
  name += address[address.length() - 2];
  name += address[address.length() - 1];
  name += address[address.length() - 5];
  name += address[address.length() - 4];

  Serial.print("name = ");
  Serial.println(name);

  BLE.setLocalName(name.c_str());
  BLE.setDeviceName(name.c_str());
  BLE.setAdvertisedService(service);

  service.addCharacteristic(strokeCharacteristic);
  service.addCharacteristic(accelerationCharacteristic);
  service.addCharacteristic(gyroscopeCharacteristic);
  service.addCharacteristic(magneticFieldCharacteristic);
  
  BLE.addService(service);
  BLE.advertise();

  // start scanning for peripherals
  BLE.scan();
  scan_started = true;

  // // Set up logging. Google style is to avoid globals or statics because of
  // // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // static tflite::MicroErrorReporter micro_error_reporter;  // NOLINT
  // error_reporter = &micro_error_reporter;

  // // Map the model into a usable data structure. This doesn't involve any
  // // copying or parsing, it's a very lightweight operation.
  // model = tflite::GetModel(g_magic_wand_model_data);
  // if (model->version() != TFLITE_SCHEMA_VERSION) {
  //   TF_LITE_REPORT_ERROR(error_reporter,
  //                        "Model provided is schema version %d not equal "
  //                        "to supported version %d.",
  //                        model->version(), TFLITE_SCHEMA_VERSION);
  //   return;
  // }

  // // Pull in only the operation implementations we need.
  // // This relies on a complete list of all the ops needed by this graph.
  // // An easier approach is to just use the AllOpsResolver, but this will
  // // incur some penalty in code space for op implementations that are not
  // // needed by this graph.
  // static tflite::MicroMutableOpResolver<4> micro_op_resolver;  // NOLINT
  // micro_op_resolver.AddConv2D();
  // micro_op_resolver.AddMean();
  // micro_op_resolver.AddFullyConnected();
  // micro_op_resolver.AddSoftmax();

  // // Build an interpreter to run the model with.
  // static tflite::MicroInterpreter static_interpreter(
  //     model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  // interpreter = &static_interpreter;

  // // Allocate memory from the tensor_arena for the model's tensors.
  // interpreter->AllocateTensors();

  // // Set model input settings
  // TfLiteTensor* model_input = interpreter->input(0);
  // if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
  //     (model_input->dims->data[1] != raster_height) ||
  //     (model_input->dims->data[2] != raster_width) ||
  //     (model_input->dims->data[3] != raster_channels) ||
  //     (model_input->type != kTfLiteInt8)) {
  //   TF_LITE_REPORT_ERROR(error_reporter,
  //                        "Bad input tensor parameters in model");
  //   return;
  // }

  // // Set model output settings
  // TfLiteTensor* model_output = interpreter->output(0);
  // if ((model_output->dims->size != 2) || (model_output->dims->data[0] != 1) ||
  //     (model_output->dims->data[1] != label_count) ||
  //     (model_output->type != kTfLiteInt8)) {
  //   TF_LITE_REPORT_ERROR(error_reporter,
  //                        "Bad output tensor parameters in model");
  //   return;
  // }
}

void setLedPinValue(int pin, int value) {
  // RGB LED's are pulled up, so the PWM needs to be inverted

  if (value == 0) {
    // special hack to clear LED
    analogWrite(pin, 256);
  } else {
    analogWrite(pin, 255 - value);
  }
}

void set_led_ble_disconnected(void) {
  setLedPinValue(LEDR, 255);
  setLedPinValue(LEDG, 0);
  setLedPinValue(LEDB, 0);
}

void set_led_ble_connected(void) {
  wheel_colour = wheel_colour + 2;
  if(wheel_colour > 255) { 
    wheel_colour = 0;
  }
  set_led_colour(wheel_colour);
}

void set_led_colour(int wheel_colour) {
  if(wheel_colour < 85) {
    setLedPinValue(LEDR, wheel_colour * 3);
    setLedPinValue(LEDG, 255 - wheel_colour * 3);
    setLedPinValue(LEDB, 0);
  } 
  else if(wheel_colour < 170) {
    wheel_colour -= 85;
    setLedPinValue(LEDR, 255 - wheel_colour * 3);
    setLedPinValue(LEDG, 0);
    setLedPinValue(LEDB, wheel_colour * 3);
  } 
  else {
    wheel_colour -= 170;
    setLedPinValue(LEDR, 0);
    setLedPinValue(LEDG, wheel_colour * 3);
    setLedPinValue(LEDB, 255 - wheel_colour * 3);
  }
}

void loop() {
  central = BLE.central();

  if (!found_logger && !scan_started) {
      BLE.scan();
      scan_started = true;
  }
  // check if a peripheral has been discovered
  peripheral = BLE.available();

  if (peripheral || !found_logger) {
    // discovered a peripheral, print out address, local name, and advertised service
    Serial.print("Found ");
    Serial.print(peripheral.address());
    Serial.print(" '");
    Serial.print(peripheral.localName());
    Serial.print("' ");
    Serial.print(peripheral.advertisedServiceUuid());
    Serial.println();

    if (peripheral.localName().startsWith("BoxingTrackerServer")) {
      found_logger = true;
      BLE.stopScan();
      scan_started = false;
      peripheral.connect();
      peripheral.discoverAttributes();
      BLEService service = peripheral.service("e7cdff01-8262-4851-847d-cd1961238f47");
      dataPeripheralCharacteristic = service.characteristic("e7cdff02-8262-4851-847d-cd1961238f47");
      strokePeripheralCharacteristic = service.characteristic("e7cdff03-8262-4851-847d-cd1961238f47");
    }
  }
  
  // if a central is connected to the peripheral:
  static bool was_connected_last = false;  
  if (central && !was_connected_last) {
    Serial.print("Connected to central: ");
    // print the central's BT address:
    Serial.println(central.address());
  }
  else {
  }
  was_connected_last = central;

  if (found_logger) {
    set_led_ble_connected();
  }
  else {
    set_led_ble_disconnected();    
  }


  // make sure IMU data is available then read in data
  const bool data_available = IMU.accelerationAvailable() || IMU.gyroscopeAvailable();
  if (!data_available) {
    return;
  }
  int accelerometer_samples_read;
  int gyroscope_samples_read;

  if (IMU.accelerationAvailable()) {
    float x, y, z;

    IMU.readAcceleration(x, y, z);

    float acceleration[3] = { x, y, z };

    accelerationCharacteristic.writeValue(acceleration, sizeof(acceleration));
    if (found_logger) {
      String val = "accel,";
      val.concat(x);
      val.concat(",");
      val.concat(y);
      val.concat(",");
      val.concat(z);
      val.concat("\n");
      if(!dataPeripheralCharacteristic.writeValue(val.c_str())) {
        found_logger = false;
      }
    }
  }

  if (IMU.gyroscopeAvailable()) {
    float x, y, z;

    IMU.readGyroscope(x, y, z);

    float dps[3] = { x, y, z };

    gyroscopeCharacteristic.writeValue(dps, sizeof(dps));
    if (found_logger) {
      String val = "gyro,";
      val.concat(x);
      val.concat(",");
      val.concat(y);
      val.concat(",");
      val.concat(z);
      val.concat("\n");
      if(!dataPeripheralCharacteristic.writeValue(val.c_str())) {
        found_logger = false;
      }
    }
  }

  if (IMU.magneticFieldAvailable()) {
    float x, y, z;

    IMU.readMagneticField(x, y, z);

    float magneticField[3] = { x, y, z };

    magneticFieldCharacteristic.writeValue(magneticField, sizeof(magneticField));
    if (found_logger) {
      String val = "mag,";
      val.concat(x);
      val.concat(",");
      val.concat(y);
      val.concat(",");
      val.concat(z);
      val.concat("\n");
      if(!dataPeripheralCharacteristic.writeValue(val.c_str())) {
        found_logger = false;
      }
    }
  }  

  ReadAccelerometerAndGyroscope(&accelerometer_samples_read, &gyroscope_samples_read);

  // Parse and process IMU data
  bool done_just_triggered = false;
  if (gyroscope_samples_read > 0) {
    EstimateGyroscopeDrift(current_gyroscope_drift);
    UpdateOrientation(gyroscope_samples_read, current_gravity, current_gyroscope_drift);
    UpdateStroke(gyroscope_samples_read, &done_just_triggered);
    if (central && central.connected()) {
      strokeCharacteristic.writeValue(stroke_struct_buffer, stroke_struct_byte_count);
    }
    if(found_logger) {
      if(!strokePeripheralCharacteristic.writeValue(stroke_struct_buffer, stroke_struct_byte_count)) {
        found_logger = false;
      }
    }
  }
  // if (accelerometer_samples_read > 0) {
  //   EstimateGravityDirection(current_gravity);
  //   UpdateVelocity(accelerometer_samples_read, current_gravity);
  // }

  // // Wait for a gesture to be done
  // if (done_just_triggered) {
  //   // Rasterize the gesture
  //   RasterizeStroke(stroke_points, *stroke_transmit_length, 0.6f, 0.6f, raster_width, raster_height, raster_buffer);
  //   for (int y = 0; y < raster_height; ++y) {
  //     char line[raster_width + 1];
  //     for (int x = 0; x < raster_width; ++x) {
  //       const int8_t* pixel = &raster_buffer[(y * raster_width * raster_channels) + (x * raster_channels)];
  //       const int8_t red = pixel[0];
  //       const int8_t green = pixel[1];
  //       const int8_t blue = pixel[2];
  //       char output;
  //       if ((red > -128) || (green > -128) || (blue > -128)) {
  //         output = '#';
  //       } else {
  //         output = '.';
  //       }
  //       line[x] = output;
  //     }
  //     line[raster_width] = 0;
  //     Serial.println(line);
  //   }

  //   // Pass to the model and run the interpreter
  //   TfLiteTensor* model_input = interpreter->input(0);
  //   for (int i = 0; i < raster_byte_count; ++i) {
  //     model_input->data.int8[i] = raster_buffer[i];
  //   }
  //   TfLiteStatus invoke_status = interpreter->Invoke();
  //   if (invoke_status != kTfLiteOk) {
  //     TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
  //     return;
  //   }
  //   TfLiteTensor* output = interpreter->output(0);

  //   // Parse the model output
  //   int8_t max_score;
  //   int max_index;
  //   for (int i = 0; i < label_count; ++i) {
  //     const int8_t score = output->data.int8[i];
  //     if ((i == 0) || (score > max_score)) {
  //       max_score = score;
  //       max_index = i;
  //     }
  //   }
  //   TF_LITE_REPORT_ERROR(error_reporter, "Found %s (%d)", labels[max_index], max_score);
  // }
}
