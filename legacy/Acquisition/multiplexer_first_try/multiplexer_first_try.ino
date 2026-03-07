#include <Wire.h>
#include <MPU6050.h>

MPU6050 mpu;
const int multiplexerAddress = 0x70; // Standard I2C address for TCA9548A

void setup() {
  Serial.begin(9600);
  Wire.begin();

  for (int i = 0; i < 4; i++) {
    selectMultiplexerChannel(i);
    mpu.begin(); // Use default initialization
    Serial.print("Calibrating MPU6050 on Channel "); Serial.println(i);
    mpu.CalibrateGyro(); // Use suggested calibration method
    delay(10); 
  }
}

void loop() {
  for (int i = 0; i < 4; i++) {
    selectMultiplexerChannel(i);
    float ax = mpu.getAccX();
    float ay = mpu.getAccY();
    float az = mpu.getAccZ();
    float gx = mpu.getGyroX();
    float gy = mpu.getGyroY();
    float gz = mpu.getGyroZ();

    Serial.print("MPU6050 (Channel "); Serial.print(i); Serial.println("): ");
    Serial.print("AccX = "); Serial.println(ax);
    Serial.print("AccY = "); Serial.println(ay);
    Serial.print("AccZ = "); Serial.println(az);
    Serial.print("GyroX = "); Serial.println(gx);
    Serial.print("GyroY = "); Serial.println(gy);
    Serial.print("GyroZ = "); Serial.println(gz);
    Serial.println("-----------------------");
    delay(500);
  }
}

void selectMultiplexerChannel(byte channel) {
  if (channel > 7) return;
  
  Wire.beginTransmission(multiplexerAddress);
  Wire.write(1 << channel);
  Wire.endTransmission();
}
