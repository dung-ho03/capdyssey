// USB acquisition reference: XIAO nRF52840 Sense + MAX30101 SEN-16474.
// Hardware upload/validation is required. No AF inference runs on this board.
#include <Wire.h>
#include <MAX30105.h>
#include <Adafruit_TinyUSB.h>

MAX30105 ppg;
uint8_t imuAddress = 0x6A;
char imuDiagnostic[100];
const char *initError = "sensor_init";
// Internal IMU bus only. Release lines with pullups; bound clock stretching.
const int imuSda = PIN_WIRE1_SDA, imuScl = PIN_WIRE1_SCL;
void lowLine(int p){ digitalWrite(p,LOW); pinMode(p,OUTPUT); }
void releaseLine(int p){ pinMode(p,INPUT_PULLUP); }
bool highClock(){
  releaseLine(imuScl);
  uint32_t t=micros();
  while(!digitalRead(imuScl)) if((uint32_t)(micros()-t)>500) return false;
  delayMicroseconds(3); return true;
}
bool iStart(){
  releaseLine(imuSda);
  if(!highClock() || !digitalRead(imuSda)) return false;
  lowLine(imuSda); delayMicroseconds(3); lowLine(imuScl); return true;
}
void iStop(){ lowLine(imuSda); highClock(); releaseLine(imuSda); delayMicroseconds(3); }
bool bitClock(bool &v){
  delayMicroseconds(3);
  if(!highClock()) return false;
  v=digitalRead(imuSda); lowLine(imuScl); return true;
}
bool sendByte(uint8_t b){
  bool v;
  for(int i=7;i>=0;i--){
    if(b&(1<<i)) releaseLine(imuSda); else lowLine(imuSda);
    if(!bitClock(v)) return false;
  }
  releaseLine(imuSda); return bitClock(v) && !v;
}
bool imuWrite(uint8_t addr,uint8_t value){
  bool ok=iStart() && sendByte(imuAddress << 1) && sendByte(addr) && sendByte(value);
  iStop(); return ok;
}
bool imuRead(uint8_t addr,uint8_t *out,uint8_t n){
  bool ok=iStart() && sendByte(imuAddress << 1) && sendByte(addr) && iStart() && sendByte((imuAddress << 1) | 1);
  for(int j=0;ok && j<n;j++){
    uint8_t b=0; bool v=false; releaseLine(imuSda);
    for(int k=0;k<8;k++){ if(!(ok=bitClock(v))) break; b=(b<<1)|v; }
    if(!ok) break;
    if(j+1<n) lowLine(imuSda); else releaseLine(imuSda);
    ok=bitClock(v); out[j]=b;
  }
  iStop(); return ok;
}
float axisG(const uint8_t* p) {
  return (int16_t)((uint16_t)p[0] | ((uint16_t)p[1]<<8)) * 0.000122f;
}
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

bool readPPG(uint8_t reg,uint8_t *data,uint8_t len) {
  Wire.beginTransmission(0x57);
  Wire.write(reg);
  if(Wire.endTransmission()!=0 || Wire.requestFrom((uint8_t)0x57,len)!=len) return false;
  for(int i=0;i<len;i++) data[i]=Wire.read();
  return true;
}
void startStream() {
  if (!sensorsOK) { Serial.print("#ERROR,"); Serial.println(initError); return; }
  ppg.shutDown(); ppg.clearFIFO(); ppg.wakeUp();
  while (ppg.available()) ppg.nextSample();
  sequenceNumber = 0;
  streaming = true;
  Serial.println("#CAPDYSSEY,1,100,red,g");
  Serial.println("#TIMING,board_poll_time_approximate");
}

void setup() {
  Serial.begin(115200);
  Wire.begin();
  initError = "ppg_not_found";
  if (!ppg.begin(Wire, I2C_SPEED_FAST)) return;
  ppg.setup(0x1F, 1, 2, 100, 411, 4096);
  pinMode(PIN_LSM6DS3TR_C_POWER, OUTPUT);
  digitalWrite(PIN_LSM6DS3TR_C_POWER, HIGH);
  delay(100); releaseLine(imuSda); releaseLine(imuScl);
  uint8_t id;
  initError = "imu_id";
  bool found = false;
  int ids[2] = {-1,-1};
  for (int i=0; i<2; i++) {
    imuAddress = 0x6A+i;
    if (imuRead(0x0F,&id,1)) {
      ids[i]=id;
      if (id==0x69 || id==0x6A) { found=true; break; }
    }
  }
  if (!found) {
    snprintf(imuDiagnostic,sizeof(imuDiagnostic),"imu_id addr6A=%d addr6B=%d SDA=%d SCL=%d",ids[0],ids[1],digitalRead(imuSda),digitalRead(imuScl));
    initError=imuDiagnostic;
    return;
  }
  initError = "imu_config";
  if (!imuWrite(0x12,0x44) || !imuWrite(0x10,0x48)) return; // BDU, increment, 104 Hz, +/-4g
  delay(30);
  // One raw sample per 100 Hz period; averaging must remain 1.
  // Red + IR acquisition; only raw RED is sent to the current model.

  ppg.disableFIFORollover();
  sensorsOK = true;
}

void loop() {
  static uint32_t reportAt = 0;
  if (!sensorsOK && Serial && millis()-reportAt >= 1000) {
    reportAt = millis(); Serial.print("#ERROR,"); Serial.println(initError);
  }
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
  uint8_t fifo[3];
  if (!readPPG(0x04,fifo,3)) {
    streaming=false; Serial.println("#ERROR,ppg_status_read"); return;
  }
  const uint8_t count=(fifo[0]-fifo[2]) & 31;
  if ((fifo[1]&31) || count>1) {
    Serial.print("#GAP,ppg_fifo,"); Serial.print(count); Serial.print(','); Serial.println(fifo[1]&31);
    ppg.shutDown(); ppg.clearFIFO(); ppg.wakeUp(); delay(1);
    return;
  }
  if(count==0){ delayMicroseconds(250); return; }
  uint64_t stamp=clockUs();
  uint8_t raw[6];
  if(!readPPG(0x07,raw,6)){
    streaming=false; Serial.println("#ERROR,ppg_sample_read"); return;
  }
  uint32_t red=((uint32_t)(raw[0]&3)<<16)|((uint32_t)raw[1]<<8)|raw[2];
  uint8_t a[6];
  if (!imuRead(0x28,a,6)) { Serial.println("#GAP,imu_read_error"); return; }
  float ax=axisG(a), ay=axisG(a+2), az=axisG(a+4);
  if (abs(ax) >= 3.95 || abs(ay) >= 3.95 || abs(az) >= 3.95) {
    Serial.println("#GAP,imu_saturated"); return;
  }
  char timestamp[24];
  char *end = timestamp + sizeof(timestamp) - 1;
  *end = 0;
  do { *--end = '0' + stamp % 10; stamp /= 10; } while (stamp);
  Serial.print("D,"); Serial.print(sequenceNumber++); Serial.print(',');
  Serial.print(end); Serial.print(','); Serial.print(red); Serial.print(',');
  Serial.print(ax, 6); Serial.print(','); Serial.print(ay, 6); Serial.print(',');
  Serial.println(az, 6);
}
