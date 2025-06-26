#!/usr/bin/env python3
"""
Quick start script for the Car Racing RL project
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"{description}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"stderr: {e.stderr}")
        return False

def main():
    print("Car Racing RL Project - Quick Start")
    print("===================================")
    
    # Check if we're in the right directory
    if not os.path.exists("requirements.txt"):
        print("Error: Please run this script from the RL project directory")
        return
    
    print("\nWhat would you like to do?")
    print("1. Test the environment (see how it works)")
    print("2. Quick training (5000 timesteps)")
    print("3. Full training (50000 timesteps)")
    print("4. Test a trained model")
    print("5. Install dependencies")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == "1":
        print("Testing the CarRacing environment...")
        run_command("python test_environment.py --mode observation", "Checking environment details")
        
        answer = input("\nWould you like to see the environment in action? (y/n): ").lower()
        if answer == 'y':
            run_command("python test_environment.py --mode manual", "Running visual test")
    
    elif choice == "2":
        print("Starting quick training session...")
        run_command("python train_car.py --timesteps 5000", "Quick training (5K timesteps)")
    
    elif choice == "3":
        print("Starting full training session...")
        answer = input("This will take a while. Continue? (y/n): ").lower()
        if answer == 'y':
            run_command("python train_car.py --timesteps 50000 --render", "Full training with visualization")
    
    elif choice == "4":
        print("Testing trained model...")
        if os.path.exists("models") and os.listdir("models"):
            run_command("python test_car.py --episodes 3", "Testing latest model")
        else:
            print("No trained models found. Please train a model first.")
    
    elif choice == "5":
        print("Installing dependencies...")
        run_command("pip install -r requirements.txt", "Installing Python packages")
    
    else:
        print("Invalid choice. Please run the script again.")
    
    print("\n" + "="*50)
    print("Quick start completed!")
    print("\nNext steps:")
    print("- Check the README.md for detailed instructions")
    print("- Monitor training with: tensorboard --logdir logs/")
    print("- Experiment with different hyperparameters")

if __name__ == "__main__":
    main()
