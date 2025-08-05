#!/usr/bin/env python3
"""
Embodied AI Use Case 3: Human-Robot Interaction
Demonstrates natural language command understanding and execution using transformer models.
This would typically be deployed to C++ with speech recognition and robot control systems.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import json
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class ActionType(Enum):
    MOVE = "move"
    PICK = "pick"
    PLACE = "place"
    ROTATE = "rotate"
    WAIT = "wait"
    SPEAK = "speak"

@dataclass
class RobotAction:
    """Structured robot action"""
    action_type: ActionType
    parameters: Dict
    confidence: float = 1.0

@dataclass
class WorldState:
    """Current state of robot's world"""
    robot_position: np.ndarray
    robot_orientation: float
    objects: Dict[str, Dict]  # object_name -> {position, type, properties}
    humans: List[Dict]        # List of detected humans
    task_context: str         # Current task being performed

class LanguageGrounder:
    """Maps natural language to robot actions and world understanding"""
    
    def __init__(self):
        # Pre-trained language model for understanding
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.language_model = AutoModel.from_pretrained('distilbert-base-uncased')
        
        # Action classifier network
        self.action_classifier = nn.Sequential(
            nn.Linear(768, 512),  # DistilBERT hidden size
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, len(ActionType))
        )
        
        # Parameter extraction network
        self.parameter_extractor = nn.Sequential(
            nn.Linear(768 + len(ActionType), 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  # x, y, z, rotation, speed, force, duration, object_id, etc.
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.action_classifier.parameters()) + 
            list(self.parameter_extractor.parameters()), 
            lr=0.001
        )
        
        # Training data generation templates
        self.command_templates = self._create_command_templates()
        
    def _create_command_templates(self) -> Dict:
        """Create templates for generating training data"""
        return {
            ActionType.MOVE: [
                "go to {location}",
                "move to {location}",
                "navigate to the {location}",
                "walk to {location}",
                "head to the {location}",
                "travel to {location}",
                "move {direction} {distance} meters",
                "go {direction}",
                "move forward {distance}",
                "back up {distance}"
            ],
            ActionType.PICK: [
                "pick up the {object}",
                "grab the {object}",
                "take the {object}",
                "get the {object}",
                "lift the {object}",
                "pick up {object} from {location}",
                "grasp the {object}",
                "collect the {object}"
            ],
            ActionType.PLACE: [
                "put the {object} on {location}",
                "place {object} on the {location}",
                "set {object} down on {location}",
                "drop {object} at {location}",
                "put {object} in the {location}",
                "place {object} here",
                "set down the {object}"
            ],
            ActionType.ROTATE: [
                "turn {direction}",
                "rotate {direction}",
                "face {direction}",
                "look {direction}",
                "turn around",
                "spin {direction} {degrees} degrees",
                "turn to face {object}"
            ],
            ActionType.WAIT: [
                "wait {duration} seconds",
                "pause for {duration}",
                "hold on",
                "wait here",
                "stay put",
                "pause",
                "stop and wait"
            ],
            ActionType.SPEAK: [
                "say {message}",
                "tell me {message}",
                "speak {message}",
                "announce {message}",
                "report {status}",
                "explain {concept}"
            ]
        }
    
    def generate_training_data(self, num_samples=2000) -> List[Dict]:
        """Generate synthetic training data for command understanding"""
        training_data = []
        
        # Vocabulary for filling templates
        locations = ["kitchen", "living room", "bedroom", "table", "counter", "shelf", 
                    "chair", "sofa", "door", "window", "corner", "center"]
        objects = ["cup", "book", "phone", "keys", "bottle", "box", "pen", "paper",
                  "remote", "plate", "bowl", "tool", "toy", "bag"]
        directions = ["left", "right", "forward", "backward", "north", "south", "east", "west"]
        
        for _ in range(num_samples):
            # Randomly select action type
            action_type = random.choice(list(ActionType))
            template = random.choice(self.command_templates[action_type])
            
            # Fill template with random values
            command = template
            parameters = {}
            
            if "{location}" in template:
                location = random.choice(locations)
                command = command.replace("{location}", location)
                parameters["target_location"] = location
                parameters["x"] = random.uniform(-5, 5)
                parameters["y"] = random.uniform(-5, 5)
                parameters["z"] = random.uniform(0, 2)
            
            if "{object}" in template:
                obj = random.choice(objects)
                command = command.replace("{object}", obj)
                parameters["object_name"] = obj
                parameters["object_id"] = random.randint(0, 9)
            
            if "{direction}" in template:
                direction = random.choice(directions)
                command = command.replace("{direction}", direction)
                parameters["direction"] = direction
                if direction in ["left", "right"]:
                    parameters["rotation"] = random.uniform(-180, 180)
            
            if "{distance}" in template:
                distance = random.uniform(0.5, 5.0)
                command = command.replace("{distance}", f"{distance:.1f}")
                parameters["distance"] = distance
            
            if "{duration}" in template:
                duration = random.uniform(1, 10)
                command = command.replace("{duration}", f"{duration:.1f}")
                parameters["duration"] = duration
            
            if "{degrees}" in template:
                degrees = random.choice([30, 45, 60, 90, 180])
                command = command.replace("{degrees}", str(degrees))
                parameters["rotation"] = degrees
            
            if "{message}" in template:
                messages = ["hello", "task complete", "ready", "understood", "help needed"]
                message = random.choice(messages)
                command = command.replace("{message}", message)
                parameters["message"] = message
            
            if "{status}" in template:
                statuses = ["current status", "battery level", "task progress"]
                status = random.choice(statuses)
                command = command.replace("{status}", status)
                parameters["status"] = status
            
            if "{concept}" in template:
                concepts = ["the task", "the plan", "the problem", "what happened"]
                concept = random.choice(concepts)
                command = command.replace("{concept}", concept)
                parameters["concept"] = concept
            
            # Add some variation and noise
            if random.random() < 0.3:  # 30% chance to add politeness
                politeness = random.choice(["please ", "could you ", "can you "])
                command = politeness + command
            
            if random.random() < 0.2:  # 20% chance to add urgency
                urgency = random.choice([" quickly", " now", " immediately"])
                command = command + urgency
            
            training_data.append({
                "command": command,
                "action_type": action_type,
                "parameters": parameters
            })
        
        return training_data
    
    def encode_command(self, command: str) -> torch.Tensor:
        """Encode natural language command using language model"""
        inputs = self.tokenizer(command, return_tensors='pt', padding=True, truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = self.language_model(**inputs)
            # Use CLS token embedding as sentence representation
            command_embedding = outputs.last_hidden_state[:, 0, :]
        
        return command_embedding
    
    def train_command_understanding(self, training_data: List[Dict], epochs=50):
        """Train the command understanding model"""
        print(f"Training on {len(training_data)} command examples...")
        
        # Prepare training data
        commands = [data["command"] for data in training_data]
        action_labels = [list(ActionType).index(data["action_type"]) for data in training_data]
        
        # Create parameter vectors (normalized)
        parameter_vectors = []
        for data in training_data:
            params = data["parameters"]
            param_vector = np.zeros(10)
            
            # Encode common parameters
            param_vector[0] = params.get("x", 0) / 10.0  # Normalize position
            param_vector[1] = params.get("y", 0) / 10.0
            param_vector[2] = params.get("z", 0) / 10.0
            param_vector[3] = params.get("rotation", 0) / 180.0  # Normalize rotation
            param_vector[4] = params.get("distance", 0) / 10.0    # Normalize distance
            param_vector[5] = params.get("duration", 0) / 10.0    # Normalize duration
            param_vector[6] = params.get("object_id", 0) / 10.0   # Normalize object ID
            
            # Binary indicators
            param_vector[7] = 1.0 if "object_name" in params else 0.0
            param_vector[8] = 1.0 if "target_location" in params else 0.0
            param_vector[9] = 1.0 if "message" in params else 0.0
            
            parameter_vectors.append(param_vector)
        
        # Training loop
        for epoch in range(epochs):
            total_action_loss = 0
            total_param_loss = 0
            
            # Shuffle data
            indices = list(range(len(training_data)))
            random.shuffle(indices)
            
            batch_size = 32
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                
                # Prepare batch
                batch_commands = [commands[idx] for idx in batch_indices]
                batch_action_labels = torch.LongTensor([action_labels[idx] for idx in batch_indices])
                batch_param_labels = torch.FloatTensor([parameter_vectors[idx] for idx in batch_indices])
                
                # Encode commands
                command_embeddings = []
                for cmd in batch_commands:
                    embedding = self.encode_command(cmd)
                    command_embeddings.append(embedding)
                command_embeddings = torch.cat(command_embeddings, dim=0)
                
                # Forward pass
                action_logits = self.action_classifier(command_embeddings)
                action_probs = F.softmax(action_logits, dim=1)
                
                # Combine embeddings with action predictions for parameter extraction
                combined_features = torch.cat([command_embeddings, action_probs], dim=1)
                predicted_params = self.parameter_extractor(combined_features)
                
                # Compute losses
                action_loss = F.cross_entropy(action_logits, batch_action_labels)
                param_loss = F.mse_loss(predicted_params, batch_param_labels)
                
                total_loss = action_loss + 0.5 * param_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                total_action_loss += action_loss.item()
                total_param_loss += param_loss.item()
            
            if epoch % 10 == 0:
                avg_action_loss = total_action_loss / (len(training_data) // batch_size)
                avg_param_loss = total_param_loss / (len(training_data) // batch_size)
                print(f"Epoch {epoch}: Action Loss: {avg_action_loss:.4f}, Param Loss: {avg_param_loss:.4f}")
    
    def understand_command(self, command: str, world_state: WorldState) -> RobotAction:
        """Parse natural language command into structured robot action"""
        # Encode command
        command_embedding = self.encode_command(command)
        
        # Predict action type
        with torch.no_grad():
            action_logits = self.action_classifier(command_embedding)
            action_probs = F.softmax(action_logits, dim=1)
            predicted_action_idx = torch.argmax(action_probs, dim=1).item()
            confidence = torch.max(action_probs, dim=1)[0].item()
            
            # Predict parameters
            combined_features = torch.cat([command_embedding, action_probs], dim=1)
            predicted_params = self.parameter_extractor(combined_features)[0].numpy()
        
        # Convert to structured action
        action_type = list(ActionType)[predicted_action_idx]
        
        # Decode parameters based on action type and world context
        parameters = self._decode_parameters(action_type, predicted_params, command, world_state)
        
        return RobotAction(
            action_type=action_type,
            parameters=parameters,
            confidence=confidence
        )
    
    def _decode_parameters(self, action_type: ActionType, param_vector: np.ndarray, 
                          command: str, world_state: WorldState) -> Dict:
        """Decode parameter vector into meaningful parameters"""
        parameters = {}
        
        if action_type == ActionType.MOVE:
            # Check if command mentions specific object or location
            target_location = self._extract_location_from_command(command, world_state)
            if target_location:
                parameters["target_location"] = target_location
                parameters["x"] = target_location[0]
                parameters["y"] = target_location[1]
                parameters["z"] = target_location[2]
            else:
                parameters["x"] = param_vector[0] * 10.0
                parameters["y"] = param_vector[1] * 10.0
                parameters["z"] = max(0, param_vector[2] * 2.0)
            
            parameters["speed"] = 1.0  # Default speed
        
        elif action_type == ActionType.PICK:
            object_name = self._extract_object_from_command(command, world_state)
            if object_name:
                parameters["object_name"] = object_name
                if object_name in world_state.objects:
                    obj_pos = world_state.objects[object_name]["position"]
                    parameters["x"] = obj_pos[0]
                    parameters["y"] = obj_pos[1]
                    parameters["z"] = obj_pos[2]
            
            parameters["force"] = 0.5  # Default gripping force
        
        elif action_type == ActionType.PLACE:
            target_location = self._extract_location_from_command(command, world_state)
            if target_location:
                parameters["target_location"] = target_location
                parameters["x"] = target_location[0]
                parameters["y"] = target_location[1]
                parameters["z"] = target_location[2]
        
        elif action_type == ActionType.ROTATE:
            if "degrees" in command:
                # Extract specific rotation amount
                import re
                match = re.search(r'(\d+)\s*degrees?', command)
                if match:
                    parameters["rotation"] = float(match.group(1))
            else:
                parameters["rotation"] = param_vector[3] * 180.0
        
        elif action_type == ActionType.WAIT:
            if "second" in command:
                import re
                match = re.search(r'(\d+\.?\d*)\s*seconds?', command)
                if match:
                    parameters["duration"] = float(match.group(1))
            else:
                parameters["duration"] = max(1.0, param_vector[5] * 10.0)
        
        elif action_type == ActionType.SPEAK:
            # Extract message from command
            message = self._extract_message_from_command(command)
            parameters["message"] = message
        
        return parameters
    
    def _extract_location_from_command(self, command: str, world_state: WorldState) -> Optional[np.ndarray]:
        """Extract target location from command"""
        locations = {
            "kitchen": np.array([2, 0, 0]),
            "living room": np.array([0, 2, 0]),
            "bedroom": np.array([-2, 0, 0]),
            "table": np.array([1, 1, 0]),
            "counter": np.array([2, 1, 0]),
            "door": np.array([0, -3, 0])
        }
        
        for location, pos in locations.items():
            if location in command.lower():
                return pos
        
        return None
    
    def _extract_object_from_command(self, command: str, world_state: WorldState) -> Optional[str]:
        """Extract object name from command"""
        for obj_name in world_state.objects.keys():
            if obj_name.lower() in command.lower():
                return obj_name
        
        # Check for common object words
        common_objects = ["cup", "book", "phone", "keys", "bottle", "box"]
        for obj in common_objects:
            if obj in command.lower():
                return obj
        
        return None
    
    def _extract_message_from_command(self, command: str) -> str:
        """Extract message content from speak command"""
        if "say" in command:
            parts = command.split("say", 1)
            if len(parts) > 1:
                return parts[1].strip()
        elif "tell" in command:
            parts = command.split("tell", 1)
            if len(parts) > 1:
                return parts[1].strip()
        
        return "Message received"

class HumanRobotInteractionSystem:
    """Complete HRI system with natural language understanding"""
    
    def __init__(self):
        self.language_grounder = LanguageGrounder()
        self.world_state = WorldState(
            robot_position=np.array([0, 0, 0]),
            robot_orientation=0,
            objects={
                "red_cup": {"position": np.array([1, 0.5, 0.8]), "type": "cup", "color": "red"},
                "blue_book": {"position": np.array([0.5, 1, 0.9]), "type": "book", "color": "blue"},
                "phone": {"position": np.array([2, 0.5, 0.9]), "type": "phone", "color": "black"},
                "keys": {"position": np.array([1.5, 1.5, 0.85]), "type": "keys", "color": "silver"}
            },
            humans=[{"position": np.array([0, 1, 0]), "name": "user"}],
            task_context="idle"
        )
        self.conversation_history = []
    
    def train_system(self):
        """Train the natural language understanding system"""
        print("Generating training data for human-robot interaction...")
        training_data = self.language_grounder.generate_training_data(num_samples=1000)
        
        print("Training language grounding model...")
        self.language_grounder.train_command_understanding(training_data, epochs=30)
        
        print("Training completed!")
    
    def process_command(self, command: str) -> Dict:
        """Process natural language command and return response"""
        print(f"\nUser: {command}")
        
        # Understand the command
        action = self.language_grounder.understand_command(command, self.world_state)
        
        # Execute the action (simulated)
        response = self._execute_action(action)
        
        # Store in conversation history
        self.conversation_history.append({
            "user_command": command,
            "understood_action": action,
            "robot_response": response
        })
        
        return {
            "understood_action": action,
            "response": response,
            "confidence": action.confidence
        }
    
    def _execute_action(self, action: RobotAction) -> str:
        """Simulate action execution and generate response"""
        if action.action_type == ActionType.MOVE:
            target = action.parameters.get("target_location", "specified location")
            self.world_state.robot_position = np.array([
                action.parameters.get("x", 0),
                action.parameters.get("y", 0),
                action.parameters.get("z", 0)
            ])
            return f"Moving to {target}. Estimated arrival in 5 seconds."
        
        elif action.action_type == ActionType.PICK:
            obj_name = action.parameters.get("object_name", "object")
            if obj_name in self.world_state.objects:
                return f"Picking up the {obj_name}. Grasp successful."
            else:
                return f"I cannot find the {obj_name}. Could you point it out?"
        
        elif action.action_type == ActionType.PLACE:
            location = action.parameters.get("target_location", "specified location")
            return f"Placing object at {location}. Task completed."
        
        elif action.action_type == ActionType.ROTATE:
            rotation = action.parameters.get("rotation", 90)
            self.world_state.robot_orientation += rotation
            return f"Rotating {rotation} degrees. New orientation: {self.world_state.robot_orientation:.1f}°"
        
        elif action.action_type == ActionType.WAIT:
            duration = action.parameters.get("duration", 5)
            return f"Waiting for {duration} seconds as requested."
        
        elif action.action_type == ActionType.SPEAK:
            message = action.parameters.get("message", "Hello")
            return f"Speaking: {message}"
        
        else:
            return "I understood your command but I'm not sure how to execute it."
    
    def demonstrate_interaction(self):
        """Demonstrate human-robot interaction with various commands"""
        test_commands = [
            "go to the kitchen",
            "pick up the red cup",
            "place it on the table",
            "turn left 90 degrees",
            "wait 3 seconds",
            "say hello to everyone",
            "can you grab the blue book please?",
            "move forward 2 meters",
            "rotate to face the door",
            "tell me the current status"
        ]
        
        print("\n" + "="*60)
        print("HUMAN-ROBOT INTERACTION DEMONSTRATION")
        print("="*60)
        
        results = []
        for command in test_commands:
            result = self.process_command(command)
            print(f"Robot: {result['response']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Action: {result['understood_action'].action_type.value}")
            print("-" * 40)
            
            results.append({
                "command": command,
                "confidence": result['confidence'],
                "action_type": result['understood_action'].action_type.value
            })
        
        # Visualize results
        self._visualize_interaction_results(results)
        
        return results
    
    def _visualize_interaction_results(self, results: List[Dict]):
        """Visualize interaction analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Confidence scores
        commands = [r["command"][:20] + "..." if len(r["command"]) > 20 else r["command"] 
                   for r in results]
        confidences = [r["confidence"] for r in results]
        
        bars1 = ax1.barh(range(len(commands)), confidences, color='skyblue')
        ax1.set_yticks(range(len(commands)))
        ax1.set_yticklabels(commands, fontsize=8)
        ax1.set_xlabel('Confidence Score')
        ax1.set_title('Command Understanding Confidence')
        ax1.set_xlim(0, 1)
        
        # Add confidence values on bars
        for i, (bar, conf) in enumerate(zip(bars1, confidences)):
            ax1.text(conf + 0.01, i, f'{conf:.2f}', va='center', fontsize=8)
        
        # Action type distribution
        action_types = [r["action_type"] for r in results]
        action_counts = {}
        for action in action_types:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        ax2.pie(action_counts.values(), labels=action_counts.keys(), autopct='%1.1f%%')
        ax2.set_title('Action Type Distribution')
        
        plt.tight_layout()
        plt.suptitle('Human-Robot Interaction Analysis', y=1.02, fontsize=14)
        plt.show()

def main():
    """Main training and demonstration"""
    print("Initializing Human-Robot Interaction System...")
    
    hri_system = HumanRobotInteractionSystem()
    
    # Train the system
    hri_system.train_system()
    
    # Demonstrate interaction
    results = hri_system.demonstrate_interaction()
    
    # Calculate overall performance
    avg_confidence = np.mean([r["confidence"] for r in results])
    print(f"\nOverall Performance:")
    print(f"Average Confidence: {avg_confidence:.3f}")
    print(f"Commands Processed: {len(results)}")
    
    # Export models for C++/Rust deployment using torch.export()
    hri_system.language_grounder.action_classifier.eval()
    hri_system.language_grounder.parameter_extractor.eval()
    
    # Create example inputs for export
    example_command_embedding = torch.randn(1, 768)  # DistilBERT embedding size
    example_action_probs = torch.randn(1, len(ActionType))  # Action probabilities
    example_combined = torch.cat([example_command_embedding, example_action_probs], dim=1)
    
    try:
        # Export action classifier
        exported_action_classifier = torch.export.export(
            hri_system.language_grounder.action_classifier, 
            (example_command_embedding,)
        )
        torch.export.save(exported_action_classifier, 'hri_action_classifier.pt2')
        
        # Export parameter extractor
        exported_parameter_extractor = torch.export.export(
            hri_system.language_grounder.parameter_extractor,
            (example_combined,)
        )
        torch.export.save(exported_parameter_extractor, 'hri_parameter_extractor.pt2')
        
        print("Models exported successfully:")
        print("- 'hri_action_classifier.pt2' for action classification")
        print("- 'hri_parameter_extractor.pt2' for parameter extraction")
        
        # Also save state dicts as fallback
        torch.save({
            'action_classifier': hri_system.language_grounder.action_classifier.state_dict(),
            'parameter_extractor': hri_system.language_grounder.parameter_extractor.state_dict()
        }, 'hri_model_weights.pth')
        print("Model weights saved as 'hri_model_weights.pth'")
        
        # Print deployment information
        print("\nDeployment Information:")
        print("- Use .pt2 files with ExecuTorch for mobile/embedded deployment")
        print("- Use weights file with LibTorch C++ API")
        print("- Action classifier input: [batch_size, 768] (command embedding)")
        print("- Action classifier output: [batch_size, {}] (action logits)".format(len(ActionType)))
        print("- Parameter extractor input: [batch_size, {}] (embedding + action_probs)".format(768 + len(ActionType)))
        print("- Parameter extractor output: [batch_size, 10] (parameter vector)")
        print("\nNote: You'll need to integrate DistilBERT separately in C++/Rust for text encoding")
        
    except Exception as e:
        print(f"torch.export() failed: {e}")
        print("Falling back to state dict save for manual deployment")
        torch.save({
            'action_classifier': hri_system.language_grounder.action_classifier.state_dict(),
            'parameter_extractor': hri_system.language_grounder.parameter_extractor.state_dict()
        }, 'hri_model_weights.pth')
        print("Model weights saved as 'hri_model_weights.pth'")

if __name__ == "__main__":
    main()