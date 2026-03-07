#include <Wire.h>
#include <SD.h>
#include <SPI.h>
#include <MPU6050_tockn.h>

// Define pins and addresses
#define SD_CS 53
#define BUZZER_PIN 8
#define RED_LED_PIN 9
#define GREEN_LED_PIN 10
#define BLUE_LED_PIN 11

MPU6050 mpu6050[4] = { 
  MPU6050(Wire, 0x68), 
  MPU6050(Wire, 0x68), 
  MPU6050(Wire, 0x68), 
  MPU6050(Wire, 0x68) 
};

File dataFile;

void setup() {
  Wire.begin();
  Serial.begin(9600);

  // Initialize SD card
  if (!SD.begin(SD_CS)) {
    Serial.println("Card initialization failed.");
    return;
  }
  Serial.println("Card initialized.");

  // Open file for data writing
  dataFile = SD.open("data.csv", FILE_WRITE);
  if (!dataFile) {
    Serial.println("Failed to open data.csv for writing.");
    return;
  }
  dataFile.println("IMU1AccelX;IMU1AccelY;IMU1AccelZ;IMU1GyroX;IMU1GyroY;IMU1GyroZ;..."); // Extend this header as needed
  
  // Initialize and calibrate each MPU6050
  for (int i = 0; i < 4; i++) {
    selectMultiplexerChannel(i);
    mpu6050[i].begin();
    mpu6050[i].calcGyroOffsets(true);
  }

  // Set up pins
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(RED_LED_PIN, OUTPUT);
  pinMode(GREEN_LED_PIN, OUTPUT);
  pinMode(BLUE_LED_PIN, OUTPUT);
  
  // Initial LED and Buzzer states
  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(RED_LED_PIN, LOW);
  digitalWrite(GREEN_LED_PIN, HIGH);
  digitalWrite(BLUE_LED_PIN, LOW);
}

void loop() {
  String csvLine = "";

  for (int i = 0; i < 4; i++) {
    selectMultiplexerChannel(i);
    mpu6050[i].update();

    // Check each data reading
    checkDataAndUpdateComponents(mpu6050[i].getAccX());
    checkDataAndUpdateComponents(mpu6050[i].getAccY());
    checkDataAndUpdateComponents(mpu6050[i].getAccZ());
    checkDataAndUpdateComponents(mpu6050[i].getGyroX());
    checkDataAndUpdateComponents(mpu6050[i].getGyroY());
    checkDataAndUpdateComponents(mpu6050[i].getGyroZ());

    // Append to CSV line
    csvLine += String(mpu6050[i].getAccX()) + ";";
    csvLine += String(mpu6050[i].getAccY()) + ";";
    csvLine += String(mpu6050[i].getAccZ()) + ";";
    csvLine += String(mpu6050[i].getGyroX()) + ";";
    csvLine += String(mpu6050[i].getGyroY()) + ";";
    csvLine += String(mpu6050[i].getGyroZ()) + ";";
  }

  // Save to the open file
  if (dataFile) {
    dataFile.println(csvLine);
    dataFile.flush(); // Ensure data is immediately written to SD card
  }

  delay(100);
}

void selectMultiplexerChannel(byte channel) {
  if (channel > 7) return;

  Wire.beginTransmission(0x70);  // TCA9548A address
  Wire.write(1 << channel);      // select the specified channel
  Wire.endTransmission();
}

void checkDataAndUpdateComponents(float data) {
  if (data == -1) {
    digitalWrite(RED_LED_PIN, HIGH);
    digitalWrite(GREEN_LED_PIN, LOW);
    digitalWrite(BLUE_LED_PIN, LOW);
    digitalWrite(BUZZER_PIN, HIGH);
  } else {
    digitalWrite(RED_LED_PIN, LOW);
    digitalWrite(GREEN_LED_PIN, HIGH);
    digitalWrite(BLUE_LED_PIN, LOW);
    digitalWrite(BUZZER_PIN, LOW);
  }
}

void endDataLogging() {
  // Close the file properly when finishing logging
  if (dataFile) {
    dataFile.close();
  }
}
