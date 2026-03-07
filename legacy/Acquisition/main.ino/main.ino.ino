#include <Wire.h>
#include <MPU6050_tockn.h>
#include <SD.h>

MPU6050 mpu6050(Wire);
const int multiplexerAddress = 0x70;
const int chipSelect = 53;  // CS pin for SD card on Arduino Mega
const int buttonPin = 10;
const int buzzerPin = 22;

long  measureCount = 0; 

#define RED_LED_PIN 2
#define GREEN_LED_PIN 3
#define BLUE_LED_PIN 4

File dataFile;  // Global file handle
bool recording = false;

void setup() {
  Serial.begin(9600);
  pinMode(buttonPin, INPUT);
  pinMode(buzzerPin, OUTPUT);

  initializeIMUsAndSDCard();

  buzzNotification();

  waitForButtonClick();

  recording = true;
  dataFile = SD.open("data.txt", O_WRITE | O_CREAT | O_TRUNC);
  dataFile.println("IMU1AccelX;IMU1AccelY;IMU1AccelZ;IMU1GyroX;IMU1GyroY;IMU1GyroZ;IMU2AccelX;IMU2AccelY;IMU2AccelZ;IMU2GyroX;IMU2GyroY;IMU2GyroZ;IMU3AccelX;IMU3AccelY;IMU3AccelZ;IMU3GyroX;IMU3GyroY;IMU3GyroZ;IMU4AccelX;IMU4AccelY;IMU4AccelZ;IMU4GyroX;IMU4GyroY;IMU4GyroZ");

}

void loop() {
  if (recording) {
    recordData();
  }

  checkAndToggleRecording();
}

void initializeIMUsAndSDCard() {
  pinMode(BLUE_LED_PIN, OUTPUT);
  pinMode(RED_LED_PIN, OUTPUT);
  pinMode(GREEN_LED_PIN, OUTPUT);
  digitalWrite(BLUE_LED_PIN, HIGH);

  Wire.begin();
  for (int i = 0; i < 4; i++) {
    selectMultiplexerChannel(i);
    mpu6050.begin();
    mpu6050.calcGyroOffsets(true);
  }

  if (!SD.begin(chipSelect)) {
    Serial.println("Card failed, or not present");
    while (1);
  }
  Serial.println("SD card initialized.");
}

void buzzNotification() {
  digitalWrite(buzzerPin, HIGH);
  delay(100);  // A short buzz
  digitalWrite(buzzerPin, LOW);
}

void waitForButtonClick() {
  while (digitalRead(buttonPin) == HIGH) {
    delay(100);  // Small delay to avoid CPU spinning too fast
  }
  Serial.print("Starting recording");
}

void recordData() {
  digitalWrite(BLUE_LED_PIN, LOW);  // If recording, turn off blue LED

  for (int i = 0; i < 4; i++) {
    selectMultiplexerChannel(i);
    mpu6050.update();
    checkSensorValues();

    dataFile.print(mpu6050.getAccX(), 10);
    dataFile.print(";");
    dataFile.print(mpu6050.getAccY(), 10);
    dataFile.print(";");
    dataFile.print(mpu6050.getAccZ(), 10);
    dataFile.print(";");
    dataFile.print(mpu6050.getGyroX(), 10);
    dataFile.print(";");
    dataFile.print(mpu6050.getGyroY(), 10);
    dataFile.print(";");
    dataFile.print(mpu6050.getGyroZ(), 10);
    dataFile.print(";");
  }
  dataFile.println();
}

void checkAndToggleRecording() {
  static bool prevButtonState = LOW;
  bool currentButtonState = digitalRead(buttonPin);
  
  if (currentButtonState == HIGH && prevButtonState == LOW) { // Detecting button press (rising edge)
    recording = !recording;  // Toggle recording state
    
    if (recording) {
      digitalWrite(BLUE_LED_PIN, LOW);  // If paused, set LED to blue indicating pause
      dataFile = SD.open("data.txt", FILE_WRITE);  // Re-open the file if recording is resumed
      Serial.print("Resuming recording"); 
    } else {
      digitalWrite(BLUE_LED_PIN, HIGH);  // If paused, set LED to blue indicating pause
      Serial.print("Pausing recording"); 
      Serial.println("Count : "); 
      Serial.print(measureCount);
      if (dataFile) {
        dataFile.close();
      }
    }
  }
  prevButtonState = currentButtonState;
}

void selectMultiplexerChannel(byte channel) {
  if (channel > 7) return;
  Wire.beginTransmission(multiplexerAddress);
  Wire.write(1 << channel);
  Wire.endTransmission();
}

void checkSensorValues() {
  if(mpu6050.getAccX() != -0 && mpu6050.getAccY() != -0 && mpu6050.getAccZ() != -0) {
    // ok 
    digitalWrite(GREEN_LED_PIN, LOW);
    digitalWrite(RED_LED_PIN, HIGH);
    
  } else {
    // error 
    digitalWrite(GREEN_LED_PIN, HIGH);
    digitalWrite(RED_LED_PIN, LOW);
  }
}