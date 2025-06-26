#!/usr/bin/env python3
"""
Training script for the car racing RL environment using Gymnasium's CarRacing-v3.
Uses Stable Baselines3 PPO algorithm.
"""

import argparse
import os
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack
import torch


def create_env(render_mode=None):
    """Create and return the car racing environment"""
    env = gym.make('CarRacing-v3', render_mode=render_mode)
    return env


def main():
    parser = argparse.ArgumentParser(description="Train a car racing RL agent")
    parser.add_argument("--render", action="store_true", help="Render environment during training")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps to train")
    parser.add_argument("--eval-freq", type=int, default=5000, help="Evaluation frequency")
    parser.add_argument("--save-freq", type=int, default=10000, help="Model save frequency")
    parser.add_argument("--model-name", type=str, default="car_ppo", help="Model name for saving")
    parser.add_argument("--load-model", type=str, help="Path to pre-trained model to continue training")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    print("=== Car Racing RL Training ===")
    print(f"Total timesteps: {args.timesteps}")
    print(f"Render mode: {'ON' if args.render else 'OFF'}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print()
    
    # Create environment
    render_mode = "human" if args.render else None
    
    # Create vectorized environment for training
    train_env = make_vec_env(
        lambda: Monitor(create_env(render_mode=render_mode)),
        n_envs=1,
        seed=42
    )
    
    # Create evaluation environment
    eval_env = Monitor(create_env())
    
    # Create or load model
    if args.load_model and os.path.exists(args.load_model):
        print(f"Loading pre-trained model: {args.load_model}")
        model = PPO.load(args.load_model, env=train_env)
        print("Model loaded successfully!")
    else:
        print("Creating new PPO model...")
        model = PPO(
            "CnnPolicy",  # Use CNN for image-based observations
            train_env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=32,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log="./logs/"
        )
    
    # Set up callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{args.model_name}_best_{timestamp}",
        log_path="./logs/",
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False
    )
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=f"./models/checkpoints_{timestamp}/",
        name_prefix=args.model_name
    )
    
    callbacks = [eval_callback, checkpoint_callback]
    
    # Training loop with progress tracking
    print("Starting training...")
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            tb_log_name=f"{args.model_name}_{timestamp}",
            progress_bar=True
        )
        
        # Save final model
        final_model_path = f"./models/{args.model_name}_final_{timestamp}"
        model.save(final_model_path)
        
        training_time = time.time() - start_time
        print(f"\nTraining completed!")
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Final model saved to: {final_model_path}")
        
        # Quick evaluation
        print("\nRunning final evaluation...")
        obs, _ = eval_env.reset()
        total_reward = 0
        steps = 0
        
        for _ in range(1000):  # Max steps for evaluation
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        print(f"Final evaluation - Total reward: {total_reward:.2f}, Steps: {steps}")
        print(f"Checkpoints reached: {info.get('checkpoint', 0)}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        # Save current model
        interrupt_model_path = f"./models/{args.model_name}_interrupted_{timestamp}"
        model.save(interrupt_model_path)
        print(f"Model saved to: {interrupt_model_path}")
    
    finally:
        # Clean up
        train_env.close()
        eval_env.close()
        print("Training session ended.")


if __name__ == "__main__":
    main()
