#!/usr/bin/env python3
"""Script to generate synthetic network flow data."""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.processor import SyntheticDataGenerator
from src.utils.utils import setup_logging, set_random_seeds, ensure_dir


def main():
    """Main function for data generation."""
    parser = argparse.ArgumentParser(description="Generate synthetic network flow data")
    parser.add_argument("--output", "-o", type=str, default="data/synthetic_flows.parquet",
                       help="Output file path")
    parser.add_argument("--n-samples", type=int, default=10000,
                       help="Total number of samples to generate")
    parser.add_argument("--n-intrusions", type=int, default=1000,
                       help="Number of intrusion samples")
    parser.add_argument("--random-seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--format", choices=["parquet", "csv", "json"], default="parquet",
                       help="Output file format")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(level=log_level)
    
    # Set random seeds
    set_random_seeds(args.random_seed)
    
    # Ensure output directory exists
    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    
    logger.info("Starting synthetic data generation")
    logger.info(f"Parameters: n_samples={args.n_samples}, n_intrusions={args.n_intrusions}")
    
    # Generate synthetic data
    generator = SyntheticDataGenerator(
        n_samples=args.n_samples,
        n_intrusions=args.n_intrusions,
        random_state=args.random_seed,
        logger=logger
    )
    
    data = generator.generate_data()
    
    # Save data
    logger.info(f"Saving data to {output_path}")
    if args.format == "parquet":
        data.to_parquet(output_path, index=False)
    elif args.format == "csv":
        data.to_csv(output_path, index=False)
    elif args.format == "json":
        data.to_json(output_path, orient="records", indent=2)
    
    # Print summary
    logger.info("Data generation completed")
    logger.info(f"Generated {len(data)} samples")
    logger.info(f"Normal samples: {len(data[data['label'] == 0])}")
    logger.info(f"Intrusion samples: {len(data[data['label'] == 1])}")
    logger.info(f"Intrusion rate: {data['label'].mean():.2%}")
    
    print(f"\nSynthetic data saved to: {output_path}")
    print(f"Dataset shape: {data.shape}")
    print(f"Features: {list(data.columns)}")


if __name__ == "__main__":
    main()
