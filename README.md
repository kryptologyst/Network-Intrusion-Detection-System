# Network Intrusion Detection System

A comprehensive Network Intrusion Detection System (NIDS) for security research and education. This project demonstrates modern machine learning approaches to detect network intrusions using flow-based features.

## ⚠️ DISCLAIMER

**This is a research and educational demonstration project. It is NOT intended for production security operations or real-world deployment.**

- This system may produce inaccurate results
- It should not be used as a replacement for professional SOC tools
- All data is synthetic or anonymized for demonstration purposes
- No real network traffic or sensitive information is processed

## Features

- **Modern ML Pipeline**: Comprehensive feature engineering, multiple model types, and robust evaluation
- **Flow-based Detection**: Analyzes network flows using statistical and behavioral features
- **Multiple Algorithms**: Random Forest, XGBoost, 1D-CNN, and TCN models
- **Imbalanced Learning**: Handles rare intrusion events with focal loss and class weighting
- **Explainability**: SHAP values and feature importance for model interpretability
- **Interactive Demo**: Streamlit-based web interface for real-time analysis
- **Production Ready**: Proper configuration management, logging, and testing

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Network-Intrusion-Detection-System.git
cd Network-Intrusion-Detection-System

# Install dependencies
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"
```

### Basic Usage

```bash
# Generate synthetic network data
python scripts/generate_data.py --output data/synthetic_flows.parquet

# Train models
python scripts/train.py --config configs/default.yaml

# Evaluate models
python scripts/evaluate.py --model-path models/best_model.pkl

# Launch interactive demo
streamlit run demo/app.py
```

## Project Structure

```
Network-Intrusion-Detection-System/
├── src/                    # Source code
│   ├── data/              # Data processing modules
│   ├── features/          # Feature engineering
│   ├── models/            # Model implementations
│   ├── defenses/          # Adversarial defenses
│   ├── eval/              # Evaluation metrics
│   ├── viz/               # Visualization utilities
│   └── utils/              # Common utilities
├── data/                  # Data storage
├── configs/               # Configuration files
├── scripts/               # Training and evaluation scripts
├── notebooks/             # Jupyter notebooks for analysis
├── tests/                 # Unit tests
├── assets/                # Generated plots and results
├── demo/                  # Streamlit demo application
└── models/                # Trained model artifacts
```

## Dataset Schema

The system works with network flow data containing the following features:

### Core Flow Features
- `duration`: Connection duration in seconds
- `protocol_type`: Network protocol (TCP=0, UDP=1, ICMP=2)
- `service`: Network service (http, ftp, smtp, etc.)
- `flag`: Connection state flag
- `src_bytes`: Bytes sent from source to destination
- `dst_bytes`: Bytes sent from destination to source
- `land`: Whether source and destination IPs are the same
- `wrong_fragment`: Number of wrong fragments
- `urgent`: Number of urgent packets

### Statistical Features
- `count`: Number of connections to the same host
- `srv_count`: Number of connections to the same service
- `serror_rate`: Rate of SYN errors
- `srv_serror_rate`: Rate of SYN errors for service
- `rerror_rate`: Rate of REJ errors
- `srv_rerror_rate`: Rate of REJ errors for service
- `same_srv_rate`: Rate of connections to same service
- `diff_srv_rate`: Rate of connections to different services
- `srv_diff_host_rate`: Rate of connections to different hosts
- `dst_host_count`: Count of connections to destination host
- `dst_host_srv_count`: Count of connections to destination host service
- `dst_host_same_srv_rate`: Rate of connections to same service on destination host
- `dst_host_diff_srv_rate`: Rate of connections to different services on destination host
- `dst_host_same_src_port_rate`: Rate of connections from same source port
- `dst_host_srv_diff_host_rate`: Rate of connections to different hosts for same service
- `dst_host_serror_rate`: Rate of SYN errors for destination host
- `dst_host_srv_serror_rate`: Rate of SYN errors for destination host service
- `dst_host_rerror_rate`: Rate of REJ errors for destination host
- `dst_host_srv_rerror_rate`: Rate of REJ errors for destination host service

### Label
- `label`: Binary classification (0=normal, 1=intrusion)

## Models

### Baseline Models
- **Random Forest**: Ensemble of decision trees with bootstrap sampling
- **XGBoost**: Gradient boosting with advanced regularization
- **Logistic Regression**: Linear baseline with L2 regularization

### Deep Learning Models
- **1D-CNN**: Convolutional neural network for sequence analysis
- **TCN**: Temporal Convolutional Network for time-series patterns
- **LSTM**: Long Short-Term Memory for temporal dependencies

### Imbalanced Learning
- **Focal Loss**: Addresses class imbalance in deep learning models
- **SMOTE**: Synthetic Minority Oversampling Technique
- **Class Weighting**: Adjusts loss function based on class frequencies

## Evaluation Metrics

### Detection Performance
- **AUCPR**: Area Under Precision-Recall Curve (primary metric for rare events)
- **Precision@K**: Precision at top K predictions
- **Recall@Fixed-Precision**: Recall at specific precision thresholds
- **FPR@Target-TPR**: False Positive Rate at target True Positive Rate

### Operational Metrics
- **Alert Volume**: Number of alerts per 1000 events
- **Detection Latency**: Time to detect intrusions
- **False Alarm Rate**: Rate of false positives per hour

### Model Interpretability
- **SHAP Values**: Feature importance and contribution analysis
- **Permutation Importance**: Model-agnostic feature importance
- **LIME**: Local interpretable model-agnostic explanations

## Configuration

The system uses YAML configuration files for easy customization:

```yaml
# configs/default.yaml
data:
  train_path: "data/train.parquet"
  test_path: "data/test.parquet"
  val_split: 0.2
  random_seed: 42

models:
  random_forest:
    n_estimators: 100
    max_depth: 10
    min_samples_split: 5
  
  xgboost:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
  
  cnn:
    filters: [32, 64, 128]
    kernel_sizes: [3, 5, 7]
    dropout: 0.5

training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  early_stopping_patience: 10

evaluation:
  metrics: ["aucpr", "precision_at_k", "fpr_at_tpr"]
  k_values: [10, 50, 100]
  target_tpr: 0.95
```

## Demo Application

The Streamlit demo provides an interactive interface for:

- **Real-time Analysis**: Upload network flow data for instant scoring
- **Model Comparison**: Compare different models side-by-side
- **Feature Analysis**: Explore feature importance and SHAP values
- **Alert Management**: Review and prioritize security alerts
- **Performance Monitoring**: Track model performance over time

Launch the demo:
```bash
streamlit run demo/app.py
```

## Privacy and Security

### Data Protection
- All IP addresses are hashed using SHA-256
- Usernames and emails are anonymized
- No PII is stored or transmitted
- Synthetic data generation for demonstrations

### Security Measures
- Input validation and sanitization
- Schema validation for all data inputs
- Audit logging for all operations
- Rate limiting for API endpoints

## Limitations

- **Synthetic Data**: Models trained on synthetic data may not generalize to real networks
- **Feature Limitations**: Only flow-based features are used (no packet-level analysis)
- **Protocol Coverage**: Limited to common protocols (TCP, UDP, ICMP)
- **Temporal Patterns**: Basic temporal analysis without advanced time-series modeling
- **Adversarial Robustness**: Models may be vulnerable to adversarial attacks

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{network_intrusion_detection,
  title={Network Intrusion Detection System for Security Research},
  author={Security Research Team},
  year={2026},
  url={https://github.com/kryptologyst/Network-Intrusion-Detection-System}
}
```

## Acknowledgments

- NSL-KDD dataset for inspiration
- Scikit-learn and PyTorch communities
- Security research community for best practices
# Network-Intrusion-Detection-System
