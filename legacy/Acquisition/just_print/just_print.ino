#include <Wire.h>
#include <MPU6050_tockn.h>

MPU6050 mpu6050(Wire);
const int multiplexerAddress = 0x70;

void setup() {
  Serial.begin(9600);
  Wire.begin();

  for (int i = 0; i < 4; i++) {
    selectMultiplexerChannel(i);
    mpu6050.begin();
    mpu6050.calcGyroOffsets(true);
    delay(10); 
  }
}

void loop() {
  for (int i = 0; i < 4; i++) {
    selectMultiplexerChannel(i);
    mpu6050.update();
    
    // Print the X-axis acceleration value of the current sensor
    Serial.print(mpu6050.getGyroX());
    
    // If it's not the last sensor, print a comma to separate the values
    if (i != 3) {
      Serial.print(",");
    }
  }

  // Move to the next line after printing all sensor values
  Serial.println();

  delay(100);  // Optional: Add delay for readability and to reduce data rate
}

void selectMultiplexerChannel(byte channel) {
  if (channel > 7) return;

  Wire.beginTransmission(multiplexerAddress);
  Wire.write(1 << channel);
  Wire.endTransmission();
}
