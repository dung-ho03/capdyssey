// USB acquisition reference: XIAO nRF52840 Sense + MAX30101 SEN-16474.
// Hardware upload/validation is required. No AF inference runs on this board.
#include <Wire.h>
#include <MAX30105.h>
#include <LSM6DS3.h>

MAX30105 ppg;
LSM6DS3 imu(I2C_MODE, 0x6A);
bool sensorsOK = false, streaming = false;
uint32_t sequenceNumber = 0;
String command;

uint64_t clockUs() {
  static uint32_t previous = 0;
  static uint64_t high = 0;
  uint32_t now = micros();
  if (now < previous) high += (1ULL << 32);
  previous = now;
  return high + now;
}

void startStream() {
  if (!sensorsOK) { Serial.println("#ERROR,sensor_init"); return; }
  ppg.clearFIFO();
  while (ppg.available()) ppg.nextSample();
  sequenceNumber = 0;
  streaming = true;
  Serial.println("#CAPDYSSEY,1,100,red,g");
  Serial.println("#TIMING,board_poll_time_approximate");
}

void setup() {
  Serial.begin(115200);
  Wire.begin();
  imu.settings.accelRange = 4;
  imu.settings.accelSampleRate = 104;
  if (imu.begin() != 0) return;
  if (!ppg.begin(Wire, I2C_SPEED_FAST)) return;
  // One raw sample per 100 Hz period; averaging must remain 1.
  // Red + IR acquisition; only raw RED is sent to the current model.
  ppg.setup(0x1F, 1, 2, 100, 411, 4096);
  ppg.disableFIFORollover();
  sensorsOK = true;
}

void loop() {
  clockUs(); // Extend micros rollover even when not streaming.
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      command.trim();
      if (command == "START") startStream();
      if (command == "STOP") streaming = false;
      command = "";
    } else if (command.length() < 30) command += c;
  }
  if (!streaming) { delay(1); return; }
  // The MAX3010x driver has a small host ring buffer. Never silently backfill
  // batches with invented timestamps after a USB/I2C delay.
  uint8_t overflow = ppg.readRegister8(0x57, 0x05);
  uint16_t newlyRead = ppg.check();
  if (overflow || newlyRead > 1) {
    while (ppg.available()) ppg.nextSample();
    ppg.clearFIFO();
    Serial.println("#GAP,ppg_backlog");
    return;
  }
  if (!ppg.available()) return;
  uint64_t stamp = clockUs();
  uint32_t red = ppg.getFIFORed();
  ppg.nextSample();
  uint16_t previousErrors = imu.nonSuccessCounter;
  float ax = imu.readFloatAccelX();
  float ay = imu.readFloatAccelY();
  float az = imu.readFloatAccelZ();
  if (imu.nonSuccessCounter != previousErrors) {
    Serial.println("#GAP,imu_read_error"); return;
  }
  if (abs(ax) >= 3.95 || abs(ay) >= 3.95 || abs(az) >= 3.95) {
    Serial.println("#GAP,imu_saturated"); return;
  }
  char timestamp[24];
  snprintf(timestamp, sizeof(timestamp), "%llu", (unsigned long long)stamp);
  Serial.print("D,"); Serial.print(sequenceNumber++); Serial.print(',');
  Serial.print(timestamp); Serial.print(','); Serial.print(red); Serial.print(',');
  Serial.print(ax, 6); Serial.print(','); Serial.print(ay, 6); Serial.print(',');
  Serial.println(az, 6);
}
