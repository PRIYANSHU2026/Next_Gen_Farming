/*
 * ESP32 Soil Health Monitoring System
 * 
 * This code reads data from soil sensors and sends it to the Python interface
 * via serial communication in JSON format.
 * 
 * Sensors:
 * - DHT22 (Temperature and Humidity)
 * - Capacitive Soil Moisture Sensor
 * - pH Sensor (Analog)
 * - NPK Sensor (via UART)
 */

#include <ArduinoJson.h>
#include <DHT.h>
#include <SoftwareSerial.h>

// Pin Definitions
#define DHT_PIN 4        // DHT22 data pin
#define MOISTURE_PIN 36   // Soil moisture sensor analog pin
#define PH_PIN 39         // pH sensor analog pin
#define NPK_RX_PIN 16     // NPK sensor UART RX
#define NPK_TX_PIN 17     // NPK sensor UART TX

// Sensor Configuration
#define DHT_TYPE DHT22    // DHT sensor type
#define SAMPLE_INTERVAL 5000  // Sensor reading interval in milliseconds

// Initialize sensors
DHT dht(DHT_PIN, DHT_TYPE);
SoftwareSerial npkSerial(NPK_RX_PIN, NPK_TX_PIN); // RX, TX for NPK sensor

// Variables for sensor readings
float temperature = 0.0;
float humidity = 0.0;
float moisture = 0.0;
float ph = 0.0;
float nitrogen = 0.0;
float phosphorus = 0.0;
float potassium = 0.0;

// Timing variables
unsigned long lastSampleTime = 0;

// JSON document for data transmission
StaticJsonDocument<256> jsonDoc;

void setup() {
  // Initialize serial communication with computer
  // IMPORTANT: This baud rate (115200) must match the one in the Python interface
  Serial.begin(115200);
  Serial.println("ESP32 Soil Health Monitoring System Initializing...");
  
  // Initialize NPK sensor serial
  npkSerial.begin(9600);
  
  // Initialize DHT sensor
  dht.begin();
  
  // Initialize analog pins
  pinMode(MOISTURE_PIN, INPUT);
  pinMode(PH_PIN, INPUT);
  
  // Wait for sensors to stabilize
  delay(2000);
  
  Serial.println("System initialized and ready!");
}

void loop() {
  // Check if it's time to read sensors
  if (millis() - lastSampleTime >= SAMPLE_INTERVAL) {
    lastSampleTime = millis();
    
    // Read all sensors
    readDHT();
    readMoisture();
    readPH();
    readNPK();
    
    // Send data via serial
    sendDataJSON();
  }
  
  // Check for any incoming commands
  checkCommands();
}

// Read temperature and humidity from DHT sensor
void readDHT() {
  // Read temperature and humidity
  humidity = dht.readHumidity();
  temperature = dht.readTemperature();
  
  // Check if reading failed
  if (isnan(humidity) || isnan(temperature)) {
    Serial.println("Failed to read from DHT sensor!");
    return;
  }
}

// Read soil moisture
void readMoisture() {
  // Read analog value from moisture sensor
  int rawValue = analogRead(MOISTURE_PIN);
  
  // Convert to percentage (calibrate these values for your specific sensor)
  // Assuming 4095 is dry and 1000 is in water
  moisture = map(rawValue, 4095, 1000, 0, 100);
  
  // Constrain to valid range
  moisture = constrain(moisture, 0, 100);
}

// Read pH value
void readPH() {
  // Read analog value from pH sensor
  int rawValue = analogRead(PH_PIN);
  
  // Convert to pH (calibrate these values for your specific sensor)
  // Example conversion - adjust based on your sensor's specifications
  float voltage = rawValue * (3.3 / 4095.0);
  ph = 3.5 * voltage + 0.0; // Example formula: pH = m*voltage + b
  
  // Constrain to valid pH range
  ph = constrain(ph, 0, 14);
}

// Read NPK values from NPK sensor
void readNPK() {
  // This is a simplified example - actual implementation depends on your specific NPK sensor
  // Many NPK sensors use UART communication with specific protocols
  
  // For demonstration, we'll simulate NPK readings
  // In a real implementation, you would parse the data from the npkSerial
  
  // Example: Request data from NPK sensor
  // npkSerial.write(requestCommand);
  
  // Wait for response
  // delay(500);
  
  // Read response
  // if (npkSerial.available()) {
  //   Parse the response to get N, P, K values
  // }
  
  // For now, we'll use simulated values based on moisture and temperature
  nitrogen = 100 + (moisture * 0.5) + (temperature * 2);
  phosphorus = 10 + (moisture * 0.1) + (temperature * 0.5);
  potassium = 200 + (moisture * 0.8) + (temperature * 1.5);
  
  // Constrain to reasonable ranges
  nitrogen = constrain(nitrogen, 0, 300);
  phosphorus = constrain(phosphorus, 0, 100);
  potassium = constrain(potassium, 0, 500);
}

// Send all sensor data as JSON via serial
void sendDataJSON() {
  // Clear previous data
  jsonDoc.clear();
  
  // Add sensor readings to JSON document
  jsonDoc["temperature"] = temperature;
  jsonDoc["humidity"] = humidity;
  jsonDoc["moisture"] = moisture;
  jsonDoc["ph"] = ph;
  jsonDoc["nitrogen"] = nitrogen;
  jsonDoc["phosphorus"] = phosphorus;
  jsonDoc["potassium"] = potassium;
  
  // Serialize JSON to serial
  serializeJson(jsonDoc, Serial);
  Serial.println(); // Add newline for easier parsing
}

// Check for incoming commands
void checkCommands() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    // Process commands
    if (command == "READ") {
      // Force immediate sensor reading
      readDHT();
      readMoisture();
      readPH();
      readNPK();
      sendDataJSON();
    }
    else if (command == "STATUS") {
      // Send system status
      jsonDoc.clear();
      jsonDoc["status"] = "online";
      jsonDoc["uptime_ms"] = millis();
      serializeJson(jsonDoc, Serial);
      Serial.println();
    }
    // Add more commands as needed
  }
}