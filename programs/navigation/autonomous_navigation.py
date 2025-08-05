#!/usr/bin/env python3
"""
Embodied AI Use Case 1: Autonomous Navigation
Demonstrates path planning with dynamic obstacle avoidance using reinforcement learning.
This would typically be deployed to C++ for real-time performance in production.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import random

class NavigationEnvironment:
    """Simulated 2D environment with obstacles and goal"""
    
    def __init__(self, width=20, height=20):
        self.width = width
        self.height = height
        self.reset()
    
    def reset(self):
        """Reset environment with random obstacles and goal"""
        self.grid = np.zeros((self.height, self.width))
        
        # Add random obstacles (1 = obstacle)
        num_obstacles = random.randint(5, 15)
        for _ in range(num_obstacles):
            x, y = random.randint(0, self.width-1), random.randint(0, self.height-1)
            self.grid[y, x] = 1
        
        # Set start position (avoid obstacles)
        while True:
            self.agent_pos = [random.randint(0, self.width-1), random.randint(0, self.height-1)]
            if self.grid[self.agent_pos[1], self.agent_pos[0]] == 0:
                break
        
        # Set goal position (avoid obstacles and start)
        while True:
            self.goal_pos = [random.randint(0, self.width-1), random.randint(0, self.height-1)]
            if (self.grid[self.goal_pos[1], self.goal_pos[0]] == 0 and 
                self.goal_pos != self.agent_pos):
                break
        
        return self.get_state()
    
    def get_state(self):
        """Get current state representation for neural network"""
        # Create state with agent position, goal position, and local obstacle map
        state = np.zeros((4, self.height, self.width))
        
        # Channel 0: Agent position
        state[0, self.agent_pos[1], self.agent_pos[0]] = 1
        
        # Channel 1: Goal position
        state[1, self.goal_pos[1], self.goal_pos[0]] = 1
        
        # Channel 2: Obstacles
        state[2] = self.grid
        
        # Channel 3: Distance to goal (normalized)
        for y in range(self.height):
            for x in range(self.width):
                dist = np.sqrt((x - self.goal_pos[0])**2 + (y - self.goal_pos[1])**2)
                state[3, y, x] = dist / (self.width + self.height)
        
        return state.flatten()
    
    def step(self, action):
        """Execute action and return new state, reward, done"""
        # Actions: 0=up, 1=right, 2=down, 3=left
        moves = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        dx, dy = moves[action]
        
        new_x = max(0, min(self.width-1, self.agent_pos[0] + dx))
        new_y = max(0, min(self.height-1, self.agent_pos[1] + dy))
        
        # Check collision with obstacles
        if self.grid[new_y, new_x] == 1:
            reward = -10  # Penalty for hitting obstacle
            done = False
        else:
            self.agent_pos = [new_x, new_y]
            
            # Calculate reward based on distance to goal
            dist_to_goal = np.sqrt((self.agent_pos[0] - self.goal_pos[0])**2 + 
                                 (self.agent_pos[1] - self.goal_pos[1])**2)
            
            if self.agent_pos == self.goal_pos:
                reward = 100  # Large reward for reaching goal
                done = True
            else:
                reward = -dist_to_goal * 0.1  # Small penalty for distance
                done = False
        
        return self.get_state(), reward, done

class DQNAgent(nn.Module):
    """Deep Q-Network for navigation policy"""
    
    def __init__(self, state_size, action_size, hidden_size=256):
        super(DQNAgent, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class NavigationTrainer:
    """Trainer for the navigation agent using DQN"""
    
    def __init__(self, env_size=20):
        self.env = NavigationEnvironment(env_size, env_size)
        state_size = env_size * env_size * 4  # 4 channels
        self.agent = DQNAgent(state_size, 4)  # 4 actions
        self.optimizer = optim.Adam(self.agent.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randrange(4)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.agent(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self, batch_size=32):
        """Train the agent on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.agent(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.agent(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train(self, episodes=1000):
        """Train the navigation agent"""
        scores = []
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            max_steps = 200
            
            while steps < max_steps:
                action = self.act(state)
                next_state, reward, done = self.env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
                
                self.replay()
            
            scores.append(total_reward)
            
            if episode % 100 == 0:
                avg_score = np.mean(scores[-100:])
                print(f"Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {self.epsilon:.3f}")
        
        return scores
    
    def demonstrate(self):
        """Demonstrate trained agent navigation"""
        state = self.env.reset()
        path = [self.env.agent_pos.copy()]
        
        for _ in range(100):  # Max steps
            action = self.act(state)
            state, reward, done = self.env.step(action)
            path.append(self.env.agent_pos.copy())
            
            if done:
                break
        
        # Visualize the path
        plt.figure(figsize=(10, 8))
        
        # Plot obstacles
        obstacles_y, obstacles_x = np.where(self.env.grid == 1)
        plt.scatter(obstacles_x, obstacles_y, c='red', s=100, marker='s', label='Obstacles')
        
        # Plot start and goal
        plt.scatter(path[0][0], path[0][1], c='green', s=200, marker='o', label='Start')
        plt.scatter(self.env.goal_pos[0], self.env.goal_pos[1], c='blue', s=200, marker='*', label='Goal')
        
        # Plot path
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        plt.plot(path_x, path_y, 'k-', linewidth=2, alpha=0.7, label='Path')
        plt.scatter(path_x, path_y, c='orange', s=50, alpha=0.6)
        
        plt.xlim(-0.5, self.env.width-0.5)
        plt.ylim(-0.5, self.env.height-0.5)
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.title('Autonomous Navigation - Learned Path')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.show()
        
        return len(path) - 1, path[-1] == self.env.goal_pos

def main():
    """Main training and demonstration"""
    print("Training Autonomous Navigation Agent...")
    
    trainer = NavigationTrainer(env_size=15)
    scores = trainer.train(episodes=500)
    
    print("\nTraining completed!")
    print("Demonstrating learned navigation...")
    
    steps, reached_goal = trainer.demonstrate()
    print(f"Navigation completed in {steps} steps. Goal reached: {reached_goal}")
    
    # Export model for C++/Rust deployment using torch.export()
    trainer.agent.eval()
    
    # Create example input for export
    env_size = 15
    example_state = torch.randn(1, env_size * env_size * 4)  # Batch size 1
    
    try:
        # Export using the modern torch.export API
        exported_program = torch.export.export(trainer.agent, (example_state,))
        
        # Save the exported program
        torch.export.save(exported_program, 'navigation_model.pt2')
        print("Model exported as 'navigation_model.pt2' for C++/Rust deployment using torch.export()")
        
        # Also save state dict as fallback
        torch.save(trainer.agent.state_dict(), 'navigation_model_weights.pth')
        print("Model weights saved as 'navigation_model_weights.pth'")
        
        # Print deployment information
        print("\nDeployment Information:")
        print("- Use 'navigation_model.pt2' with ExecuTorch for mobile/embedded deployment")
        print("- Use 'navigation_model_weights.pth' with LibTorch C++ API")
        print("- Input shape: [batch_size, {}]".format(env_size * env_size * 4))
        print("- Output shape: [batch_size, 4] (Q-values for 4 actions)")
        
    except Exception as e:
        print(f"torch.export() failed: {e}")
        print("Falling back to state dict save for manual deployment")
        torch.save(trainer.agent.state_dict(), 'navigation_model_weights.pth')
        print("Model weights saved as 'navigation_model_weights.pth'")

if __name__ == "__main__":
    main()