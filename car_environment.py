#!/usr/bin/env python3
"""
Custom Car Racing Environment for Reinforcement Learning
"""

import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import math


class CarEnvironment(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super(CarEnvironment, self).__init__()

        # Environment parameters
        self.width = 800
        self.height = 600
        self.render_mode = render_mode

        # Car parameters
        self.car_width = 30
        self.car_height = 15
        self.max_speed = 8
        self.max_angular_velocity = 0.2
        self.acceleration = 0.5
        self.friction = 0.95

        # Track parameters
        self.track_width = 100
        self.checkpoints = self._create_checkpoints()
        
        # Define action and observation space
        # Actions: [acceleration, steering]
        # acceleration: -1 (brake) to 1 (accelerate)
        # steering: -1 (left) to 1 (right)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # Observations: [x, y, angle, velocity_x, velocity_y, angular_velocity, 
        #                distance_to_next_checkpoint, angle_to_next_checkpoint,
        #                8 lidar distances]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(16,),
            dtype=np.float32
        )

        # Initialize pygame
        pygame.init()
        self.screen = None
        self.clock = None

        # Reset environment
        self.reset()

    def _create_checkpoints(self):
        """Create a simple oval track with checkpoints"""
        checkpoints = []
        center_x, center_y = self.width // 2, self.height // 2
        radius_x, radius_y = 250, 180
        
        for i in range(8):  # 8 checkpoints around the track
            angle = (i / 8) * 2 * math.pi
            x = center_x + radius_x * math.cos(angle)
            y = center_y + radius_y * math.sin(angle)
            checkpoints.append((x, y))
        
        return checkpoints

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize car state
        self.car_x = self.checkpoints[0][0]
        self.car_y = self.checkpoints[0][1]
        self.car_angle = 0.0
        self.car_velocity_x = 0.0
        self.car_velocity_y = 0.0
        self.car_angular_velocity = 0.0
        
        # Track progress
        self.current_checkpoint = 0
        self.checkpoints_passed = 0
        self.steps = 0
        self.max_steps = 2000
        
        # Reward tracking
        self.last_distance_to_checkpoint = self._distance_to_next_checkpoint()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info

    def step(self, action):
        self.steps += 1
        
        # Parse actions
        acceleration = np.clip(action[0], -1.0, 1.0)
        steering = np.clip(action[1], -1.0, 1.0)
        
        # Update car physics
        self._update_car_physics(acceleration, steering)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.steps >= self.max_steps
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info

    def _update_car_physics(self, acceleration, steering):
        """Update car position and velocity based on actions"""
        # Update angular velocity based on steering
        self.car_angular_velocity += steering * 0.05
        self.car_angular_velocity = np.clip(self.car_angular_velocity, 
                                          -self.max_angular_velocity, 
                                          self.max_angular_velocity)
        
        # Update angle
        self.car_angle += self.car_angular_velocity
        
        # Calculate acceleration in car's direction
        acc_x = acceleration * self.acceleration * math.cos(self.car_angle)
        acc_y = acceleration * self.acceleration * math.sin(self.car_angle)
        
        # Update velocity
        self.car_velocity_x += acc_x
        self.car_velocity_y += acc_y
        
        # Apply friction
        self.car_velocity_x *= self.friction
        self.car_velocity_y *= self.friction
        
        # Limit speed
        speed = math.sqrt(self.car_velocity_x**2 + self.car_velocity_y**2)
        if speed > self.max_speed:
            self.car_velocity_x = (self.car_velocity_x / speed) * self.max_speed
            self.car_velocity_y = (self.car_velocity_y / speed) * self.max_speed
        
        # Update position
        self.car_x += self.car_velocity_x
        self.car_y += self.car_velocity_y
        
        # Apply angular friction
        self.car_angular_velocity *= 0.9

    def _get_observation(self):
        """Get current observation"""
        # Car state
        obs = [
            self.car_x / self.width,  # Normalized position
            self.car_y / self.height,
            self.car_angle / (2 * math.pi),  # Normalized angle
            self.car_velocity_x / self.max_speed,  # Normalized velocity
            self.car_velocity_y / self.max_speed,
            self.car_angular_velocity / self.max_angular_velocity,
        ]
        
        # Distance and angle to next checkpoint
        distance = self._distance_to_next_checkpoint() / math.sqrt(self.width**2 + self.height**2)
        angle = self._angle_to_next_checkpoint() / (2 * math.pi)
        obs.extend([distance, angle])
        
        # Lidar-like distance measurements (8 directions)
        lidar_distances = self._get_lidar_distances()
        obs.extend(lidar_distances)
        
        return np.array(obs, dtype=np.float32)

    def _get_lidar_distances(self):
        """Get distance measurements in 8 directions"""
        distances = []
        max_distance = 200.0
        
        for i in range(8):
            angle = self.car_angle + (i * math.pi / 4)
            distance = self._cast_ray(self.car_x, self.car_y, angle, max_distance)
            distances.append(distance / max_distance)  # Normalize
        
        return distances

    def _cast_ray(self, start_x, start_y, angle, max_distance):
        """Cast a ray and return distance to nearest obstacle (track boundary)"""
        step_size = 5.0
        distance = 0.0
        
        x, y = start_x, start_y
        dx = step_size * math.cos(angle)
        dy = step_size * math.sin(angle)
        
        while distance < max_distance:
            x += dx
            y += dy
            distance += step_size
            
            # Check if ray hit boundary
            if (x < 0 or x >= self.width or y < 0 or y >= self.height or
                not self._is_on_track(x, y)):
                break
        
        return min(distance, max_distance)

    def _is_on_track(self, x, y):
        """Check if position is on the track"""
        center_x, center_y = self.width // 2, self.height // 2
        
        # Distance from center
        dx = x - center_x
        dy = y - center_y
        
        # Elliptical track
        outer_radius_x, outer_radius_y = 300, 220
        inner_radius_x, inner_radius_y = 150, 120
        
        # Check if inside outer ellipse but outside inner ellipse
        outer_dist = (dx/outer_radius_x)**2 + (dy/outer_radius_y)**2
        inner_dist = (dx/inner_radius_x)**2 + (dy/inner_radius_y)**2
        
        return inner_dist >= 1.0 and outer_dist <= 1.0

    def _distance_to_next_checkpoint(self):
        """Calculate distance to the next checkpoint"""
        next_checkpoint = self.checkpoints[self.current_checkpoint]
        dx = self.car_x - next_checkpoint[0]
        dy = self.car_y - next_checkpoint[1]
        return math.sqrt(dx*dx + dy*dy)

    def _angle_to_next_checkpoint(self):
        """Calculate angle to the next checkpoint"""
        next_checkpoint = self.checkpoints[self.current_checkpoint]
        dx = next_checkpoint[0] - self.car_x
        dy = next_checkpoint[1] - self.car_y
        target_angle = math.atan2(dy, dx)
        angle_diff = target_angle - self.car_angle
        
        # Normalize angle difference to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        return angle_diff

    def _calculate_reward(self):
        """Calculate reward for current step"""
        reward = 0.0
        
        # Check if passed a checkpoint
        distance_to_checkpoint = self._distance_to_next_checkpoint()
        if distance_to_checkpoint < 30:  # Checkpoint radius
            reward += 100.0  # Large reward for checkpoint
            self.current_checkpoint = (self.current_checkpoint + 1) % len(self.checkpoints)
            self.checkpoints_passed += 1
        
        # Reward for getting closer to checkpoint
        improvement = self.last_distance_to_checkpoint - distance_to_checkpoint
        reward += improvement * 0.1
        self.last_distance_to_checkpoint = distance_to_checkpoint
        
        # Penalty for going off track
        if not self._is_on_track(self.car_x, self.car_y):
            reward -= 10.0
        
        # Small penalty for each step to encourage efficiency
        reward -= 0.1
        
        # Penalty for being stationary
        speed = math.sqrt(self.car_velocity_x**2 + self.car_velocity_y**2)
        if speed < 0.5:
            reward -= 1.0
        
        return reward

    def _is_terminated(self):
        """Check if episode should terminate"""
        # Terminate if car goes too far off track
        if not self._is_on_track(self.car_x, self.car_y):
            return True
        
        # Terminate if car goes out of bounds
        if (self.car_x < 0 or self.car_x >= self.width or 
            self.car_y < 0 or self.car_y >= self.height):
            return True
        
        return False

    def _get_info(self):
        """Get additional info"""
        return {
            "checkpoint": self.checkpoints_passed,
            "distance_to_next": self._distance_to_next_checkpoint(),
            "car_speed": math.sqrt(self.car_velocity_x**2 + self.car_velocity_y**2),
            "on_track": self._is_on_track(self.car_x, self.car_y)
        }

    def render(self):
        if self.render_mode == "human":
            return self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()

    def _render_human(self):
        if self.screen is None:
            pygame.display.init()
            pygame.display.set_caption("Car Racing RL")
            self.screen = pygame.display.set_mode((self.width, self.height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Clear screen
        self.screen.fill((50, 150, 50))  # Green background

        # Draw track
        self._draw_track()
        
        # Draw checkpoints
        self._draw_checkpoints()
        
        # Draw car
        self._draw_car()
        
        # Draw UI
        self._draw_ui()

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def _draw_track(self):
        """Draw the racing track"""
        center_x, center_y = self.width // 2, self.height // 2
        
        # Outer track boundary (black)
        pygame.draw.ellipse(self.screen, (0, 0, 0), 
                          (center_x - 300, center_y - 220, 600, 440), 3)
        
        # Inner track boundary (black)
        pygame.draw.ellipse(self.screen, (0, 0, 0), 
                          (center_x - 150, center_y - 120, 300, 240), 3)
        
        # Track surface (gray)
        pygame.draw.ellipse(self.screen, (128, 128, 128), 
                          (center_x - 297, center_y - 217, 594, 434))
        pygame.draw.ellipse(self.screen, (50, 150, 50), 
                          (center_x - 153, center_y - 123, 306, 246))

    def _draw_checkpoints(self):
        """Draw checkpoints"""
        for i, (x, y) in enumerate(self.checkpoints):
            color = (255, 255, 0) if i == self.current_checkpoint else (255, 255, 255)
            pygame.draw.circle(self.screen, color, (int(x), int(y)), 15, 2)
            
            # Draw checkpoint number
            font = pygame.font.Font(None, 24)
            text = font.render(str(i), True, color)
            self.screen.blit(text, (x - 6, y - 8))

    def _draw_car(self):
        """Draw the car"""
        # Car body (rectangle rotated)
        car_corners = [
            (-self.car_width//2, -self.car_height//2),
            (self.car_width//2, -self.car_height//2),
            (self.car_width//2, self.car_height//2),
            (-self.car_width//2, self.car_height//2)
        ]
        
        # Rotate corners
        rotated_corners = []
        cos_angle = math.cos(self.car_angle)
        sin_angle = math.sin(self.car_angle)
        
        for x, y in car_corners:
            rotated_x = x * cos_angle - y * sin_angle
            rotated_y = x * sin_angle + y * cos_angle
            rotated_corners.append((
                self.car_x + rotated_x,
                self.car_y + rotated_y
            ))
        
        pygame.draw.polygon(self.screen, (255, 0, 0), rotated_corners)
        
        # Car direction indicator
        front_x = self.car_x + 20 * cos_angle
        front_y = self.car_y + 20 * sin_angle
        pygame.draw.line(self.screen, (255, 255, 255), 
                        (self.car_x, self.car_y), (front_x, front_y), 3)

    def _draw_ui(self):
        """Draw UI information"""
        font = pygame.font.Font(None, 36)
        
        # Checkpoints passed
        text = font.render(f"Checkpoints: {self.checkpoints_passed}", True, (255, 255, 255))
        self.screen.blit(text, (10, 10))
        
        # Current speed
        speed = math.sqrt(self.car_velocity_x**2 + self.car_velocity_y**2)
        text = font.render(f"Speed: {speed:.1f}", True, (255, 255, 255))
        self.screen.blit(text, (10, 50))
        
        # Steps
        text = font.render(f"Steps: {self.steps}", True, (255, 255, 255))
        self.screen.blit(text, (10, 90))

    def _render_rgb_array(self):
        if self.screen is None:
            self.screen = pygame.Surface((self.width, self.height))
        
        self._render_human()
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()


# Register the environment
gym.register(
    id='CarRacing-v0',
    entry_point='car_environment:CarEnvironment',
    max_episode_steps=1000,
)
