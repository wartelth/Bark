#include <Wire.h>
#include <MPU6050_tockn.h>

MPU6050 mpu6050(Wire);

void setup() {
  Serial.begin(9600);
  Wire.begin();
  mpu6050.begin();
  mpu6050.calcGyroOffsets(true);
}

void loop() {
  mpu6050.update();

/*
  Serial.print("AccelX: "); Serial.print(mpu6050.getAccX(), 4);  // Display 4 decimal places
  Serial.print("\tAccelY: "); Serial.print(mpu6050.getAccY(), 4);  // Display 4 decimal places
  Serial.print("\tAccelZ: "); Serial.print(mpu6050.getAccZ(), 4);  // Display 4 decimal places

  Serial.print("\tGyroX: "); Serial.print(mpu6050.getGyroX(), 4);  // Display 4 decimal places
  Serial.print("\tGyroY: "); Serial.print(mpu6050.getGyroY(), 4);  // Display 4 decimal places
  Serial.print("\tGyroZ: "); Serial.println(mpu6050.getGyroZ(), 4);  // Display 4 decimal places
*/

  Serial.println(mpu6050.getAccX(), 4);  // Display 4 decimal places

  delay(100);
}
