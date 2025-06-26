#!/usr/bin/env python3
"""
Demo script to test the car environment with manual controls.
Use this to verify the environment works correctly before training.
"""

import pygame
import numpy as np
from car_environment import CarEnvironment


def manual_control_demo():
    """Demo with manual keyboard controls"""
    print("=== Manual Control Demo ===")
    print("Controls:")
    print("  Arrow Keys - Steer and accelerate")
    print("  UP/DOWN - Accelerate/Brake")
    print("  LEFT/RIGHT - Steer")
    print("  R - Reset")
    print("  ESC - Exit")
    print()
    
    env = CarEnvironment(render_mode="human")
    obs, _ = env.reset()
    
    clock = pygame.time.Clock()
    running = True
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    obs, _ = env.reset()
                    print("Environment reset")
        
        # Get keyboard input
        keys = pygame.key.get_pressed()
        
        # Convert keyboard input to actions
        steering = 0.0
        acceleration = 0.0
        
        if keys[pygame.K_LEFT]:
            steering = -1.0
        elif keys[pygame.K_RIGHT]:
            steering = 1.0
        
        if keys[pygame.K_UP]:
            acceleration = 1.0
        elif keys[pygame.K_DOWN]:
            acceleration = -1.0
        
        # Step environment
        action = np.array([steering, acceleration])
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Print info occasionally
        if env.steps % 60 == 0:  # Every second at 60 FPS
            print(f"Steps: {env.steps}, Reward: {reward:.2f}, Checkpoint: {info.get('checkpoint', 0)}, Speed: {info.get('speed', 0):.2f}")
        
        # Reset if episode ends
        if terminated or truncated:
            print(f"Episode ended! Checkpoints reached: {info.get('checkpoint', 0)}")
            if info.get('collision', False):
                print("Collision detected!")
            obs, _ = env.reset()
        
        # Render
        env.render()
        clock.tick(60)
    
    env.close()


def random_agent_demo():
    """Demo with random actions"""
    print("=== Random Agent Demo ===")
    print("Watching a random agent for 3 episodes...")
    print("Press ESC to exit early")
    print()
    
    env = CarEnvironment(render_mode="human")
    
    for episode in range(3):
        print(f"Episode {episode + 1}/3")
        obs, _ = env.reset()
        total_reward = 0
        
        while True:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    env.close()
                    return
            
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            env.render()
            
            if terminated or truncated:
                print(f"Episode {episode + 1} ended - Total reward: {total_reward:.2f}, Checkpoints: {info.get('checkpoint', 0)}")
                break
            
            pygame.time.wait(50)  # Slow down for better visualization
    
    env.close()


def environment_info():
    """Display environment information"""
    env = CarEnvironment()
    
    print("=== Environment Information ===")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Track size: {env.track_width} x {env.track_height}")
    print(f"Car size: {env.car_size}")
    print(f"Max speed: {env.max_speed}")
    print(f"Number of checkpoints: {len(env.checkpoints)}")
    print(f"Sensor range: {env.sensor_range}")
    print(f"Number of sensors: {env.num_sensors}")
    print()
    
    # Sample observation
    obs, _ = env.reset()
    print("Sample observation:")
    print(f"  Car position: ({obs[0]:.3f}, {obs[1]:.3f}) [normalized]")
    print(f"  Car velocity: ({obs[2]:.3f}, {obs[3]:.3f}) [normalized]")
    print(f"  Car angle: {obs[4]:.3f} [normalized]")
    print(f"  Sensor readings: {obs[5:13]}")
    print(f"  Checkpoint distance: {obs[13]:.3f}")
    print(f"  Checkpoint angle: {obs[14]:.3f}")
    print()
    
    env.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Demo the car racing environment")
    parser.add_argument("--mode", choices=["manual", "random", "info"], default="manual",
                       help="Demo mode: manual control, random agent, or environment info")
    
    args = parser.parse_args()
    
    try:
        if args.mode == "manual":
            manual_control_demo()
        elif args.mode == "random":
            random_agent_demo()
        elif args.mode == "info":
            environment_info()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"Error during demo: {e}")


if __name__ == "__main__":
    main()
