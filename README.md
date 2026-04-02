# NDT Coconut Ripeness Edge Classifier 🥥⚡

An end-to-end TinyML deployment for Non-Destructive Testing (NDT) of agricultural produce. This system uses an ESP32 microcontroller and a piezoelectric acoustic sensor to classify coconuts as 'Ripe' or 'Unripe' based on acoustic resonance decay.

## 🛠 Hardware Architecture
The system is fully automated and triggers without physical buttons.
* **Brain:** ESP32-C3 Microcontroller
* **Triggers:** 2x Force Sensitive Resistors (FSRs)
* **Actuator:** 12V Solenoid (controlled via 5V Relay module)
* **Sensor:** Piezoelectric Transducer (Uniaxial)
* **Display:** SSD1306 0.96" OLED (I2C)

### Circuit Diagram
*(Generated via Wokwi)*
![Wokwi Circuit Diagram](circuit_diagram.png)

---

## 🧠 Machine Learning Pipeline
This project benchmarked 5 distinct AI architectures to find the optimal balance between Accuracy, ESP32 SRAM usage, and Inference Latency. 

| Model | Accuracy | Flash Footprint | Peak RAM | PC Latency | TFLite Req? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1D-CNN** | 100.00% | 9.62 KB | 7.71 KB | 0.0049 ms | Yes |
| **2D-CNN** | 99.33% | 5.97 KB | 3.16 KB | 0.0014 ms | Yes |
| **LSTM** | 99.67% | 168.79 KB | 1.92 KB | 0.0146 ms | Yes |
| **Standard DNN**| 100.00% | 3.80 KB | 0.28 KB | 0.0012 ms | Yes |
| **Random Forest** | **100.00%** | **~27 KB** | **< 0.5 KB** | **Sub-ms**| **No (Pure C++)** |

### Why Random Forest?
While CNNs and LSTMs are traditionally favored for time-series decay, the physical amplitude gap of the impact strike was sufficient for traditional ML. Using `micromlgen`, the Random Forest was exported as pure native C++ `if/else` statements. This bypassed the TensorFlow Lite interpreter entirely, reducing RAM usage to near-zero and eliminating floating-point scaling preprocessing.

---

## 🚀 How to Run the Project

### 1. Train the Model (Python)
1. Ensure you have Anaconda installed and your environment activated.
2. Install dependencies: `pip install scikit-learn pandas micromlgen`
3. Run the dataset generator: `python dataset_generator.py`
4. Train the RF model: `python RF.py` (This generates `coconut_rf.h`).

### 2. Flash the Firmware (C++ / Arduino IDE)
1. Open `coconut_classifier.ino` in the Arduino IDE.
2. Ensure `coconut_rf.h` is in the same sketch folder.
3. Install the `Adafruit_GFX` and `Adafruit_SSD1306` libraries via the Library Manager.
4. Select your ESP32 board and COM port.
5. Hit **Upload**.

### 3. Usage
1. Place a coconut on the dual FSR platform.
2. The system will detect the weight, trigger the solenoid, and record the acoustic wave over 3.2ms.
3. The embedded C++ Random Forest instantly classifies the decay.
4. The result ('RIPE' or 'UNRIPE') is displayed on the OLED screen.
5. Remove the coconut to reset the system for the next batch.
