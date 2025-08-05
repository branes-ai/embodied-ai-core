#!/usr/bin/env python3
"""
Embodied AI Use Case 2: Robotic Manipulation
Demonstrates vision-guided pick and place using CNN + policy network.
This would typically be deployed to C++ with real robot hardware.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List
import random

@dataclass
class RobotState:
    """Robot arm state representation"""
    joint_angles: np.ndarray  # 6 DOF arm
    gripper_pos: np.ndarray   # 3D gripper position
    gripper_open: bool        # Gripper state
    
@dataclass
class Object:
    """Object in the workspace"""
    pos: np.ndarray          # 3D position
    size: float              # Object size
    color: Tuple[int, int, int]  # RGB color
    picked: bool = False

class RobotWorkspace:
    """Simulated robot workspace with camera and objects"""
    
    def __init__(self, workspace_size=(400, 400), num_objects=5):
        self.workspace_size = workspace_size
        self.reset(num_objects)
    
    def reset(self, num_objects=5):
        """Reset workspace with random objects"""
        self.objects = []
        self.target_object_idx = 0
        
        # Generate random objects
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        for i in range(num_objects):
            pos = np.array([
                random.uniform(50, self.workspace_size[0] - 50),
                random.uniform(50, self.workspace_size[1] - 50),
                random.uniform(10, 30)  # Height above table
            ])
            self.objects.append(Object(
                pos=pos,
                size=random.uniform(15, 25),
                color=colors[i % len(colors)]
            ))
        
        # Initialize robot state
        self.robot_state = RobotState(
            joint_angles=np.zeros(6),
            gripper_pos=np.array([200, 200, 100]),  # Start above workspace
            gripper_open=True
        )
        
        # Goal position for placing objects
        self.goal_pos = np.array([350, 350, 20])
    
    def render_workspace(self) -> np.ndarray:
        """Render current workspace state as RGB image"""
        img = np.ones((self.workspace_size[1], self.workspace_size[0], 3), dtype=np.uint8) * 240
        
        # Draw workspace boundary
        cv2.rectangle(img, (10, 10), (self.workspace_size[0]-10, self.workspace_size[1]-10), 
                     (100, 100, 100), 2)
        
        # Draw goal area
        goal_center = (int(self.goal_pos[0]), int(self.goal_pos[1]))
        cv2.circle(img, goal_center, 40, (200, 200, 200), -1)
        cv2.circle(img, goal_center, 40, (150, 150, 150), 2)
        cv2.putText(img, "GOAL", (goal_center[0]-20, goal_center[1]+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        # Draw objects
        for i, obj in enumerate(self.objects):
            if not obj.picked:
                center = (int(obj.pos[0]), int(obj.pos[1]))
                radius = int(obj.size)
                cv2.circle(img, center, radius, obj.color, -1)
                cv2.circle(img, center, radius, (0, 0, 0), 2)
                
                # Highlight target object
                if i == self.target_object_idx:
                    cv2.circle(img, center, radius + 5, (0, 0, 0), 3)
        
        # Draw gripper position
        gripper_center = (int(self.robot_state.gripper_pos[0]), 
                         int(self.robot_state.gripper_pos[1]))
        gripper_color = (0, 255, 0) if self.robot_state.gripper_open else (255, 0, 0)
        cv2.circle(img, gripper_center, 8, gripper_color, -1)
        cv2.circle(img, gripper_center, 12, (0, 0, 0), 2)
        
        return img
    
    def execute_action(self, action: np.ndarray) -> Tuple[float, bool]:
        """Execute robot action and return reward and done status"""
        # Action: [dx, dy, dz, gripper_action] (normalized -1 to 1)
        move_scale = 20.0
        dx, dy, dz, gripper_action = action
        
        # Update gripper position
        new_pos = self.robot_state.gripper_pos + np.array([dx, dy, dz]) * move_scale
        new_pos = np.clip(new_pos, [20, 20, 5], [380, 380, 150])
        self.robot_state.gripper_pos = new_pos
        
        # Handle gripper action
        reward = 0
        done = False
        
        if gripper_action > 0.5 and self.robot_state.gripper_open:
            # Try to pick up object
            target_obj = self.objects[self.target_object_idx]
            if not target_obj.picked:
                dist = np.linalg.norm(self.robot_state.gripper_pos[:2] - target_obj.pos[:2])
                if dist < target_obj.size + 10:  # Close enough to pick
                    target_obj.picked = True
                    self.robot_state.gripper_open = False
                    reward = 50  # Reward for successful pick
                else:
                    reward = -5  # Penalty for failed pick attempt
        
        elif gripper_action < -0.5 and not self.robot_state.gripper_open:
            # Try to place object
            target_obj = self.objects[self.target_object_idx]
            if target_obj.picked:
                goal_dist = np.linalg.norm(self.robot_state.gripper_pos[:2] - self.goal_pos[:2])
                if goal_dist < 50:  # Close enough to goal
                    target_obj.pos = self.robot_state.gripper_pos.copy()
                    target_obj.picked = False
                    self.robot_state.gripper_open = True
                    reward = 100  # Large reward for successful place
                    
                    # Move to next object
                    self.target_object_idx += 1
                    if self.target_object_idx >= len(self.objects):
                        done = True  # All objects placed
                else:
                    reward = -5  # Penalty for placing away from goal
        
        # Small reward for moving towards target
        if not done:
            target_obj = self.objects[self.target_object_idx]
            if not target_obj.picked:
                # Reward moving towards object
                dist_to_obj = np.linalg.norm(self.robot_state.gripper_pos[:2] - target_obj.pos[:2])
                reward += max(0, (100 - dist_to_obj) * 0.01)
            else:
                # Reward moving towards goal
                dist_to_goal = np.linalg.norm(self.robot_state.gripper_pos[:2] - self.goal_pos[:2])
                reward += max(0, (100 - dist_to_goal) * 0.01)
        
        return reward, done

class VisionManipulationNet(nn.Module):
    """CNN + Policy network for vision-guided manipulation"""
    
    def __init__(self, img_height=400, img_width=400, action_dim=4):
        super(VisionManipulationNet, self).__init__()
        
        # Vision encoder (CNN)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        
        # Calculate conv output size
        conv_out_size = self._get_conv_out_size(img_height, img_width)
        
        # State encoder (robot state)
        self.state_fc = nn.Linear(10, 64)  # joint_angles(6) + gripper_pos(3) + gripper_open(1)
        
        # Policy network
        self.fc1 = nn.Linear(conv_out_size + 64, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_dim)
        
        self.dropout = nn.Dropout(0.3)
        
    def _get_conv_out_size(self, h, w):
        """Calculate output size of convolutional layers"""
        x = torch.zeros(1, 3, h, w)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return x.numel()
    
    def forward(self, img, state):
        # Process vision input
        x_img = F.relu(self.conv1(img))
        x_img = F.relu(self.conv2(x_img))
        x_img = F.relu(self.conv3(x_img))
        x_img = F.relu(self.conv4(x_img))
        x_img = x_img.view(x_img.size(0), -1)
        
        # Process state input
        x_state = F.relu(self.state_fc(state))
        
        # Combine vision and state
        x = torch.cat([x_img, x_state], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Actions normalized to [-1, 1]
        
        return x

class ManipulationTrainer:
    """Trainer for vision-guided manipulation"""
    
    def __init__(self):
        self.workspace = RobotWorkspace()
        self.policy_net = VisionManipulationNet()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0003)
        self.epsilon = 1.0
        self.epsilon_decay = 0.998
        self.epsilon_min = 0.1
        
    def get_state_vector(self) -> np.ndarray:
        """Convert robot state to vector"""
        state = np.concatenate([
            self.workspace.robot_state.joint_angles,
            self.workspace.robot_state.gripper_pos / 400.0,  # Normalize
            [1.0 if self.workspace.robot_state.gripper_open else 0.0]
        ])
        return state
    
    def collect_demonstrations(self, num_episodes=100):
        """Collect demonstration data using simple heuristic policy"""
        demonstrations = []
        
        for episode in range(num_episodes):
            self.workspace.reset()
            episode_data = []
            
            for step in range(200):  # Max steps per episode
                # Get current observation
                img = self.workspace.render_workspace()
                state = self.get_state_vector()
                
                # Simple heuristic policy for demonstration
                target_obj = self.workspace.objects[self.workspace.target_object_idx]
                
                if not target_obj.picked:
                    # Move towards object
                    target_pos = target_obj.pos[:2]
                    current_pos = self.workspace.robot_state.gripper_pos[:2]
                    direction = target_pos - current_pos
                    direction = direction / (np.linalg.norm(direction) + 1e-6)
                    
                    action = np.array([direction[0] * 0.5, direction[1] * 0.5, -0.1, 0.0])
                    
                    # Try to pick if close enough
                    if np.linalg.norm(direction) < 0.1:
                        action[3] = 1.0  # Pick action
                else:
                    # Move towards goal
                    goal_pos = self.workspace.goal_pos[:2]
                    current_pos = self.workspace.robot_state.gripper_pos[:2]
                    direction = goal_pos - current_pos
                    direction = direction / (np.linalg.norm(direction) + 1e-6)
                    
                    action = np.array([direction[0] * 0.5, direction[1] * 0.5, 0.0, 0.0])
                    
                    # Try to place if close enough
                    if np.linalg.norm(direction) < 0.15:
                        action[3] = -1.0  # Place action
                
                # Add noise for exploration
                action += np.random.normal(0, 0.1, 4)
                action = np.clip(action, -1, 1)
                
                reward, done = self.workspace.execute_action(action)
                
                episode_data.append({
                    'image': img.copy(),
                    'state': state.copy(),
                    'action': action.copy(),
                    'reward': reward
                })
                
                if done:
                    break
            
            # Only keep successful episodes (completed task)
            if self.workspace.target_object_idx >= len(self.workspace.objects):
                demonstrations.extend(episode_data)
                
            if episode % 20 == 0:
                print(f"Collected {episode} episodes, {len(demonstrations)} demonstrations")
        
        return demonstrations
    
    def train_imitation_learning(self, demonstrations, epochs=100):
        """Train policy network using imitation learning"""
        print(f"Training on {len(demonstrations)} demonstrations")
        
        for epoch in range(epochs):
            random.shuffle(demonstrations)
            total_loss = 0
            
            for i in range(0, len(demonstrations), 32):  # Batch size 32
                batch = demonstrations[i:i+32]
                
                images = torch.FloatTensor([d['image'] for d in batch]).permute(0, 3, 1, 2) / 255.0
                states = torch.FloatTensor([d['state'] for d in batch])
                actions = torch.FloatTensor([d['action'] for d in batch])
                
                predicted_actions = self.policy_net(images, states)
                loss = F.mse_loss(predicted_actions, actions)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                avg_loss = total_loss / (len(demonstrations) // 32)
                print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
    
    def demonstrate(self):
        """Demonstrate trained manipulation policy"""
        self.workspace.reset()
        
        images = []
        successful_picks = 0
        
        for step in range(300):
            # Get observation
            img = self.workspace.render_workspace()
            state = self.get_state_vector()
            
            # Get action from policy
            img_tensor = torch.FloatTensor(img).permute(2, 0, 1).unsqueeze(0) / 255.0
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                action = self.policy_net(img_tensor, state_tensor)[0].numpy()
            
            # Execute action
            old_target_idx = self.workspace.target_object_idx
            reward, done = self.workspace.execute_action(action)
            
            if self.workspace.target_object_idx > old_target_idx:
                successful_picks += 1
            
            # Store image for visualization
            if step % 20 == 0:
                images.append(img.copy())
            
            if done:
                print(f"Task completed in {step} steps!")
                break
        
        # Show final result
        plt.figure(figsize=(15, 10))
        for i, img in enumerate(images[:6]):
            plt.subplot(2, 3, i+1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(f'Step {i*20}')
            plt.axis('off')
        
        plt.suptitle(f'Vision-Guided Manipulation Demo - {successful_picks}/{len(self.workspace.objects)} Objects Completed')
        plt.tight_layout()
        plt.show()
        
        return successful_picks, len(self.workspace.objects)

def main():
    """Main training and demonstration"""
    print("Training Vision-Guided Manipulation Agent...")
    
    trainer = ManipulationTrainer()
    
    # Collect demonstration data
    print("Collecting demonstrations...")
    demonstrations = trainer.collect_demonstrations(num_episodes=50)
    
    if not demonstrations:
        print("No successful demonstrations collected!")
        return
    
    # Train policy network
    print("Training policy network...")
    trainer.train_imitation_learning(demonstrations, epochs=50)
    
    print("\nTraining completed!")
    print("Demonstrating learned manipulation...")
    
    successful, total = trainer.demonstrate()
    print(f"Manipulation completed: {successful}/{total} objects successfully placed")
    
    # Export model for C++/Rust deployment using torch.export()
    trainer.policy_net.eval()
    
    # Create example inputs for export
    example_image = torch.randn(1, 3, 400, 400)  # Batch size 1, RGB image
    example_state = torch.randn(1, 10)           # Robot state vector
    
    try:
        # Export using the modern torch.export API
        exported_program = torch.export.export(
            trainer.policy_net, 
            (example_image, example_state)
        )
        
        # Save the exported program
        torch.export.save(exported_program, 'manipulation_model.pt2')
        print("Model exported as 'manipulation_model.pt2' for C++/Rust deployment using torch.export()")
        
        # Also save state dict as fallback
        torch.save(trainer.policy_net.state_dict(), 'manipulation_model_weights.pth')
        print("Model weights saved as 'manipulation_model_weights.pth'")
        
        # Print deployment information
        print("\nDeployment Information:")
        print("- Use 'manipulation_model.pt2' with ExecuTorch for mobile/embedded deployment")
        print("- Use 'manipulation_model_weights.pth' with LibTorch C++ API")
        print("- Image input shape: [batch_size, 3, 400, 400] (RGB, normalized 0-1)")
        print("- State input shape: [batch_size, 10] (robot state vector)")
        print("- Output shape: [batch_size, 4] (action: dx, dy, dz, gripper)")
        print("- Output range: [-1, 1] (normalized actions)")
        
    except Exception as e:
        print(f"torch.export() failed: {e}")
        print("Falling back to state dict save for manual deployment")
        torch.save(trainer.policy_net.state_dict(), 'manipulation_model_weights.pth')
        print("Model weights saved as 'manipulation_model_weights.pth'")

if __name__ == "__main__":
    main()