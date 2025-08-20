#!/usr/bin/env python3
"""
Swift YOLO QAT Training Example Script

This script demonstrates how to use the Swift YOLO QAT implementation.
Adapt the configuration paths and model paths based on your setup.
"""

import os
import sys
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Swift YOLO QAT Training Example")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/swift_yolo/swift_yolo_qat_template.py",
        help="QAT configuration file path"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="Path to pre-trained Swift YOLO checkpoint"
    )
    parser.add_argument(
        "--work-dir", 
        type=str, 
        default="work_dirs/swift_yolo_qat",
        help="Working directory for QAT training"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=5,
        help="Number of QAT training epochs"
    )
    parser.add_argument(
        "--test-only", 
        action="store_true",
        help="Only test the quantized model"
    )
    parser.add_argument(
        "--gpu-id", 
        type=int, 
        default=0,
        help="GPU ID to use for training"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    
    # Paths
    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    work_dir = Path(args.work_dir)
    
    # Validation
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        print("Please create a QAT configuration file based on the template.")
        sys.exit(1)
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        print("Please provide a valid pre-trained Swift YOLO checkpoint.")
        sys.exit(1)
    
    # Create work directory
    work_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Swift YOLO QAT Training")
    print("=" * 60)
    print(f"Configuration: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Work Directory: {work_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Test Only: {args.test_only}")
    print("=" * 60)
    
    # Build command
    cmd_parts = [
        "python", "tools/quantization.py",
        str(config_path),
        str(checkpoint_path),
        "--work-dir", str(work_dir),
        "--cfg-options", f"epochs={args.epochs}"
    ]
    
    if args.test_only:
        cmd_parts.append("--test")
    
    cmd = " ".join(cmd_parts)
    
    print(f"Running command: {cmd}")
    print("=" * 60)
    
    # Execute
    exit_code = os.system(cmd)
    
    if exit_code == 0:
        print("=" * 60)
        print("QAT training completed successfully!")
        
        if not args.test_only:
            print(f"Quantized model saved in: {work_dir}")
            print(f"TFLite model: {work_dir}/qat/qat_model_int8.tflite")
            
            # Test command suggestion
            test_cmd = (
                f"python tools/quantization.py {config_path} "
                f"{work_dir}/epoch_{args.epochs}.pth --test --work-dir {work_dir}_test"
            )
            print(f"\nTo test the quantized model, run:")
            print(f"  {test_cmd}")
        
        print("=" * 60)
    else:
        print("=" * 60)
        print("QAT training failed!")
        print("Please check the error messages above.")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()