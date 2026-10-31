#!/usr/bin/env python3
"""CLI utility to generate synthetic mammography dataset."""

import argparse
import sys
from pathlib import Path

# Ensure root in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.synthetic import generate_synthetic_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic multi-site mammography dataset.")
    parser.add_argument("--output-dir", default="synthetic_data", help="Directory where dataset is generated")
    parser.add_argument("--patients-per-site", type=int, default=12, help="Patients per site")
    parser.add_argument("--images-per-patient", type=int, default=2, help="Images per patient")
    parser.add_argument("--img-size", type=int, default=224, help="Image resolution")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    out = generate_synthetic_dataset(
        data_root=args.output_dir,
        patients_per_site=args.patients_per_site,
        images_per_patient=args.images_per_patient,
        img_size=args.img_size,
        seed=args.seed,
    )
    print(f"Synthetic dataset generated successfully at: {out}")


if __name__ == "__main__":
    main()
