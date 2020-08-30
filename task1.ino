int ECG_value;
float My_Time = 0;

// the setup routine runs once when you press reset:
void setup() {
  // initialize serial communication at 9600 bits per second:
  Serial.begin(9600);
   while (!Serial) {
    ; // wait for serial port to connect. Needed for native USB
 }
  Serial.println("START");
}

// the loop routine runs over and over again forever:
void loop() {
 int ECG_value = analogRead(A0);
 Serial.print(ECG_value);
 Serial.print(",");
 Serial.println(My_Time);ss
 My_Time = My_Time + 3.90;
 delayMicroseconds(3900);
}
