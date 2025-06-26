# Simple Reinforcement Learning Car Racing Project

This project uses Gymnasium's CarRacing-v3 environment to train an RL agent using Stable Baselines3 PPO algorithm.

## Features

- Uses standard CarRacing-v3 environment from Gymnasium
- CNN-based PPO agent for image observations
- Real-time visualization during training
- Model saving and loading
- Training progress tracking with TensorBoard

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Testing the Environment
```bash
python test_environment.py --mode random     # Random agent test
python test_environment.py --mode manual     # Manual observation
python test_environment.py --mode observation # Check observation space
```

### Training the Agent
```bash
python train_car.py                          # Basic training
python train_car.py --render                 # Training with visualization
python train_car.py --timesteps 50000        # Custom timesteps
```

### Testing the Trained Agent
```bash
python test_car.py                           # Test latest model
python test_car.py --model models/your_model.zip # Test specific model
python test_car.py --episodes 10             # Custom number of episodes
```

## Environment Details

- **Environment**: CarRacing-v3 from Gymnasium
- **Observation Space**: 96x96x3 RGB images
- **Action Space**: Continuous [steering, gas, brake]
  - steering: -1.0 (left) to 1.0 (right)
  - gas: 0.0 to 1.0
  - brake: 0.0 to 1.0
- **Rewards**: Based on track progress and staying on track

## Files Structure

- `test_environment.py` - Test the CarRacing environment
- `train_car.py` - Training script with PPO
- `test_car.py` - Testing script with visualization
- `requirements.txt` - Python dependencies
- `models/` - Directory for saved models
- `logs/` - TensorBoard logs

## Training Tips

1. Start with shorter training sessions (10K-50K timesteps) to test
2. Use CNN policy for image-based observations
3. Monitor progress with TensorBoard: `tensorboard --logdir logs/`
4. Adjust hyperparameters in `train_car.py` if needed

## Next Steps

- Try different RL algorithms (A2C, SAC, etc.)
- Experiment with frame stacking for better temporal information
- Add reward shaping for better learning
- Try different hyperparameters
