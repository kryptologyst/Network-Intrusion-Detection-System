#!/usr/bin/env python3
"""Setup script for Network Intrusion Detection System."""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("🛡️ Setting up Network Intrusion Detection System")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install package in development mode
    if not run_command("pip install -e .", "Installing package in development mode"):
        sys.exit(1)
    
    # Install development dependencies
    if not run_command("pip install -e .[dev]", "Installing development dependencies"):
        print("⚠️ Development dependencies installation failed, continuing...")
    
    # Create necessary directories
    directories = ["data", "models", "results", "logs", "assets"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Generate sample data
    if not run_command("python scripts/generate_data.py --n-samples 1000 --n-intrusions 100", "Generating sample data"):
        print("⚠️ Sample data generation failed, continuing...")
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Train a model: python scripts/train.py --model-type random_forest")
    print("2. Launch demo: streamlit run demo/app.py")
    print("3. Run tests: pytest tests/")
    print("\nFor more information, see README.md")


if __name__ == "__main__":
    main()
