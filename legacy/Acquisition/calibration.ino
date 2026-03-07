#include <Wire.h>

const int MPU_addr = 0x68;

int16_t AcX, AcY, AcZ, GyX, GyY, GyZ;

void setup()
{
    Wire.begin();
    Wire.beginTransmission(MPU_addr);
    Wire.write(0x6B);
    Wire.write(0);
    Wire.endTransmission(true);
    Serial.begin(9600);
}

void loop()
{
    Wire.beginTransmission(MPU_addr);
    Wire.write(0x3B);
    Wire.endTransmission(false);
    Wire.requestFrom(MPU_addr, 14, true);

    AcX = Wire.read() << 8 | Wire.read();
    AcY = Wire.read() << 8 | Wire.read();
    AcZ = Wire.read() << 8 | Wire.read();
    GyX = Wire.read() << 8 | Wire.read();
    GyY = Wire.read() << 8 | Wire.read();
    GyZ = Wire.read() << 8 | Wire.read();

    Serial.print("AcX = ");
    Serial.println(AcX);
    Serial.print("AcY = ");
    Serial.println(AcY);
    Serial.print("AcZ = ");
    Serial.println(AcZ);
    Serial.print("GyX = ");
    Serial.println(GyX);
    Serial.print("GyY = ");
    Serial.println(GyY);
    Serial.print("GyZ = ");
    Serial.println(GyZ);
    Serial.println();

    delay(500);
}
