# 🛡️ SmartGuard AI: Advanced Network Threat Detection System

SmartGuard AI is a comprehensive network security solution that leverages machine learning and rule-based techniques to identify and mitigate potential security threats in real-time. The system features an interactive dashboard for monitoring network traffic, detecting anomalies, and visualizing potential security incidents.

## 🚀 Key Features

- **Interactive Web Dashboard** - Beautiful, real-time visualization of network traffic and threats
- **Advanced Anomaly Detection** - Multiple ML models including Isolation Forest and Random Forest
- **Real-time Network Analysis** - Live packet capture and analysis
- **Comprehensive Visualization** - Interactive charts and graphs for traffic patterns
- **Threat Intelligence** - Rule-based detection of known attack patterns
- **Modular Architecture** - Easy to extend with new detection algorithms
- **Automated Reporting** - Generate detailed security reports
- **Cross-platform** - Works on Windows, Linux, and macOS

## 🏗️ Project Structure

```
AI-Driven-Threat-Detection-System/
├── data/                   # Data storage
│   ├── raw/               # Raw captured packets
│   └── processed/         # Processed features and datasets
├── docs/                  # Documentation
├── src/                   # Source code
│   ├── config/            # Configuration files
│   ├── core/              # Core functionality
│   ├── dashboard/         # Web dashboard components
│   ├── detection/         # Anomaly detection algorithms
│   ├── features/          # Feature engineering
│   ├── models/            # Model definitions and training
│   ├── utils/             # Utility functions
│   └── main.py            # Main application entry point
├── tests/                 # Unit and integration tests
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## 🛠️ Prerequisites

- Python 3.8+
- Required Python packages (see `requirements.txt`)
- Network interface with packet capture permissions

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Network interface with packet capture permissions (for live capture)
- Windows users: Install [Npcap](https://npcap.com/) for packet capture support

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/AI-Driven-Threat-Detection-System.git
   cd AI-Driven-Threat-Detection-System
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On Linux/Mac
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🖥️ Dashboard Usage

### Starting the Dashboard

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
python run_dashboard.py
```

Then open your web browser to [http://localhost:8501](http://localhost:8501)

### Dashboard Features

- **Real-time Monitoring**: View live network traffic statistics
- **Anomaly Detection**: Visualize detected anomalies in network traffic
- **Traffic Analysis**: Explore protocol distributions and traffic patterns
- **Threat Intelligence**: View detected threats and their details
- **Export Data**: Save analysis results for further investigation

## ⚙️ Command Line Interface

```bash
# Start capturing network traffic
python src/main.py capture -i <interface> -t <timeout>

# Process captured packets
python src/main.py process -i <input_file> -o <output_file>

# Train the detection model
python src/main.py train -d <dataset> -m <model_output>

# Detect anomalies in network traffic
python src/main.py detect -i <input_file> -m <model_file>

# List available network interfaces
python src/main.py list-interfaces
```

## 🏃 Usage

### 1. Capture Network Traffic

Capture live network traffic from a specific interface:

```bash
python src/main.py capture -i <interface> -t <timeout> -c <packet_count>
```

Example:
```bash
python src/main.py capture -i Ethernet -t 60 -c 1000
```

### 2. Process Captured Packets

Process captured packets and extract features:

```bash
python src/main.py process -i data/raw/capture_20230808_123456.csv
```

### 3. Train the Model

Train the anomaly detection model:

```bash
python src/main.py train -i data/processed/features.csv
```

### 4. Detect Anomalies

Detect anomalies in network traffic:

```bash
python src/main.py detect -i data/processed/live_features.csv
```

## 📊 Features

### Core Components

- **Packet Capture**: Real-time network traffic capture and analysis
- **Feature Engineering**: Advanced feature extraction from network packets
- **Anomaly Detection**: Multiple detection algorithms with configurable thresholds
- **Visualization**: Interactive dashboards for monitoring and analysis

### Advanced Functionality

- Multiple detection algorithms (Isolation Forest, One-Class SVM, etc.)
- Real-time monitoring and alerting
- Historical data analysis
- Custom rule-based detection

## 📚 Documentation

For detailed documentation, please see the [docs](docs/) directory.

## 🤝 Contributing

Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) before submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👩‍💻 Author

Melisa Sever

## 📝 Citation

If you use this project in your research, please cite it as:

```
@misc{smartguardai2023,
  author = {Sever, Melisa},
  title = {SmartGuard AI: Network Threat Detection System},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/AI-Driven-Threat-Detection-System}}
}
```
