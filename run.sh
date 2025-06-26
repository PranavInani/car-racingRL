#!/bin/bash

# Simple Car RL Environment Setup and Run Script

echo "=== Car Racing RL Environment ==="
echo "Setup and training script for reinforcement learning car environment"
echo

# Activate virtual environment
echo "Activating virtual environment..."
source rl_env/bin/activate

# Check if packages are installed
echo "Checking installed packages..."
python -c "import pygame, numpy, gymnasium, stable_baselines3; print('✓ All packages installed successfully!')" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "⚠ Some packages are missing. Installing now..."
    pip install -r requirements.txt
    echo "✓ Installation complete!"
fi

echo
echo "Choose an option:"
echo "1. Test environment (manual control)"
echo "2. Test environment (random agent)"
echo "3. Train new model"
echo "4. Test trained model"
echo "5. Demo environment"

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo "Starting manual control test..."
        python test_environment.py --mode manual
        ;;
    2)
        echo "Starting random agent test..."
        python test_environment.py --mode random
        ;;
    3)
        echo "Starting training (this may take a while)..."
        echo "Use --render flag to watch training: python train_car.py --render"
        python train_car.py --timesteps 50000
        ;;
    4)
        echo "Testing trained model..."
        python test_car.py
        ;;
    5)
        echo "Starting demo..."
        python demo.py --mode manual
        ;;
    *)
        echo "Invalid choice. Please run the script again."
        ;;
esac
