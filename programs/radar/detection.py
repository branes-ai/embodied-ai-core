# Modified version of radar_detectionv2.py with decorator instrumentation
# Only showing the key changes - import the instrumentation and add decorators

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from collections import defaultdict
import time

# Import the instrumentation
from decorator_instrumentation import (
    trace_kalman_method, 
    start_kalman_tracing, 
    stop_kalman_tracing, 
    export_kalman_graph,
    get_graph_statistics
)

@dataclass
class RadarDetection:
    """Single radar detection/measurement - unchanged"""
    range_m: float
    azimuth_rad: float
    elevation_rad: float
    timestamp: float
    signal_strength: float
    
    def to_cartesian(self, radar_pos: np.ndarray = np.array([0, 0, 0])) -> np.ndarray:
        """Convert polar coordinates to cartesian (x, y, z) - unchanged"""
        x = self.range_m * np.cos(self.elevation_rad) * np.sin(self.azimuth_rad)
        y = self.range_m * np.cos(self.elevation_rad) * np.cos(self.azimuth_rad)
        z = self.range_m * np.sin(self.elevation_rad)
        return radar_pos + np.array([x, y, z])

@dataclass
class AircraftTrack:
    """Represents a tracked aircraft - unchanged except for decorator"""
    track_id: int
    state: np.ndarray
    covariance: np.ndarray
    last_update: float
    detection_count: int
    predicted_state: Optional[np.ndarray] = None
    confidence: float = 0.0
    
    @trace_kalman_method("kalman_predict")  # <-- ONLY CHANGE: Add decorator
    def predict(self, dt: float):
        """Predict next state using constant velocity model - ORIGINAL CODE UNCHANGED"""
        # State transition matrix for constant velocity
        F = np.eye(6)
        F[0:3, 3:6] = np.eye(3) * dt
        
        # Process noise - much more realistic for aircraft
        pos_noise = 1.0  # 1 m^2/s^2 position process noise
        vel_noise = 0.5   # 0.5 (m/s)^2/s^2 - realistic for aircraft maneuvers
        
        Q = np.eye(6)
        Q[0:3, 0:3] *= pos_noise
        Q[3:6, 3:6] *= vel_noise
        
        self.predicted_state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + Q

class RadarStation:
    """Main radar processing and tracking system - unchanged except for decorator"""
    
    def __init__(self, position: np.ndarray = np.array([0, 0, 0])):
        self.position = position
        self.tracks: List[AircraftTrack] = []
        self.next_track_id = 1
        self.detection_threshold = 0.3
        self.max_track_age = 30.0
        self.association_threshold = 500.0
        
        # Radar parameters
        self.max_range = 50000.0
        self.angular_resolution = np.deg2rad(1.0)
        self.range_resolution = 75.0
    
    # ... other methods unchanged (simulate_radar_scan, associate_detections, etc.) ...
    
    @trace_kalman_method("kalman_update")  # <-- ONLY CHANGE: Add decorator
    def update_track_kalman(self, track: AircraftTrack, detection: RadarDetection):
        """Update track using Kalman filter - ORIGINAL CODE UNCHANGED"""
        # Measurement model (observing position only)
        H = np.zeros((3, 6))
        H[:3, :3] = np.eye(3)
        
        # Realistic measurement noise for airport radar
        R = np.eye(3)
        R[0, 0] = 50.0**2    # X position measurement variance (m^2)
        R[1, 1] = 50.0**2    # Y position measurement variance (m^2)  
        R[2, 2] = 100.0**2   # Z position measurement variance (m^2)
        
        # Innovation
        z = detection.to_cartesian(self.position)  # Measurement
        predicted_measurement = H @ track.predicted_state
        y = z - predicted_measurement  # Innovation
        
        # Innovation covariance
        S = H @ track.covariance @ H.T + R
        
        # Kalman gain
        K = track.covariance @ H.T @ np.linalg.inv(S)
        
        # Update state and covariance
        track.state = track.predicted_state + K @ y
        track.covariance = (np.eye(6) - K @ H) @ track.covariance
        track.last_update = detection.timestamp
        track.detection_count += 1
        
        # Update confidence based on detection count and consistency
        innovation_norm = np.linalg.norm(y)
        track.confidence = min(1.0, track.detection_count / 10.0) * np.exp(-innovation_norm / 1000.0)
    
    # ... rest of methods unchanged (initiate_new_track, manage_tracks, etc.) ...

# Example usage with instrumentation
if __name__ == "__main__":
    print("=== Instrumented Radar System Demo ===\n")
    
    # Create radar station
    radar = RadarStation(position=np.array([0, 0, 10]))
    
    # Aircraft with realistic positions and velocities
    aircraft_states = [
        np.array([5000, 8000, 3000, 150, 100, 0], dtype=np.float64),
        np.array([-3000, 12000, 5000, -80, -120, -5], dtype=np.float64),
        np.array([15000, -5000, 8000, 50, 200, 10], dtype=np.float64)
    ]
    
    scan_interval = 1.0
    
    # START TRACING - This is where instrumentation begins
    print("Starting Kalman filter tracing...")
    start_kalman_tracing()
    
    # Run radar processing (original code unchanged)
    for scan_num in range(5):  # Reduced for demo
        print(f"\n--- Radar Scan {scan_num + 1} ---")
        
        # Update aircraft positions
        for i, state in enumerate(aircraft_states):
            aircraft_states[i][:3] += aircraft_states[i][3:6] * scan_interval
            vel_change = np.random.normal(0, 2.0, 3)
            aircraft_states[i][3:6] += vel_change
            aircraft_states[i][3] = np.clip(aircraft_states[i][3], -400, 400)
            aircraft_states[i][4] = np.clip(aircraft_states[i][4], -400, 400)  
            aircraft_states[i][5] = np.clip(aircraft_states[i][5], -50, 50)
        
        # Extract positions for radar simulation
        aircraft_positions = [state[:3] for state in aircraft_states]
        
        # Get radar detections (unchanged)
        detections = radar.simulate_radar_scan(aircraft_positions, noise_std=25.0)
        print(f"Detections: {len(detections)}")
        
        # Process the scan (unchanged) - but now being traced!
        tracks = radar.process_radar_scan(detections)
        print(f"Active tracks: {len(tracks)}")
        
        # Display track information
        for track_info in radar.get_track_info():
            speed = np.linalg.norm(track_info['velocity'][:2])
            print(f"Track {track_info['id']}: "
                  f"Speed={speed:.1f} m/s ({speed*1.944:.0f} knots) "
                  f"Conf={track_info['confidence']:.2f}")
    
    # STOP TRACING and export results
    print("\nStopping Kalman filter tracing...")
    stop_kalman_tracing()
    
    # Export computational graphs
    print("Exporting computational graphs...")
    export_kalman_graph("radar_computation_graph.json", "radar_computation_graph.dot")
    
    # Display statistics
    stats = get_graph_statistics()
    print(f"\n=== Computational Graph Statistics ===")
    print(f"Total operations captured: {stats['total_operations']}")
    print(f"Total variables tracked: {stats['total_variables']}")
    print(f"Methods instrumented: {stats['methods_traced']}")
    print("Operation types:")
    for op_type, count in stats['operation_types'].items():
        print(f"  {op_type}: {count}")
    
    print(f"\n=== Files Generated ===")
    print("- radar_computation_graph.json: Complete computational graph")
    print("- radar_computation_graph.dot: Visualization graph") 
    print("- Use 'dot -Tpng radar_computation_graph.dot -o graph.png' to visualize")
    
    print(f"\n=== Integration Summary ===")
    print("✓ Zero changes to Kalman filter algorithm logic")
    print("✓ Just added @trace_kalman_method decorators")
    print("✓ Captured all intermediate numpy operations")
    print("✓ Built computational dependency graph")
    print("✓ Minimal performance overhead when tracing disabled")

# Additional helper methods you can add to your radar station
class InstrumentedRadarStation(RadarStation):
    """Extended radar station with built-in graph tracing capabilities"""
    
    def __init__(self, position: np.ndarray = np.array([0, 0, 0])):
        super().__init__(position)
        self.tracing_enabled = False
    
    def enable_graph_tracing(self):
        """Enable computational graph tracing"""
        self.tracing_enabled = True
        start_kalman_tracing()
        print("Graph tracing enabled")
    
    def disable_graph_tracing(self):
        """Disable computational graph tracing"""
        if self.tracing_enabled:
            stop_kalman_tracing()
            self.tracing_enabled = False
            print("Graph tracing disabled")
    
    def export_graphs(self, base_filename: str = "radar_graphs"):
        """Export computational graphs with timestamp"""
        if self.tracing_enabled:
            timestamp = int(time.time())
            json_file = f"{base_filename}_{timestamp}.json"
            dot_file = f"{base_filename}_{timestamp}.dot"
            export_kalman_graph(json_file, dot_file)
            return json_file, dot_file
        else:
            print("No graphs to export - tracing not enabled")
            return None, None
    
    def get_tracing_stats(self):
        """Get current tracing statistics"""
        if self.tracing_enabled:
            return get_graph_statistics()
        else:
            return {"message": "Tracing not enabled"}

# Usage example with the enhanced radar station
def demo_enhanced_tracing():
    """Demonstrate enhanced tracing capabilities"""
    print("\n=== Enhanced Tracing Demo ===")
    
    radar = InstrumentedRadarStation()
    
    # Enable tracing for specific operations
    radar.enable_graph_tracing()
    
    # Run a few operations
    # ... (simulate some radar scans)
    
    # Check stats mid-operation
    stats = radar.get_tracing_stats()
    print(f"Operations so far: {stats['total_operations']}")
    
    # Export intermediate results
    files = radar.export_graphs("intermediate_graphs")
    
    # Continue processing...
    # ... (more radar scans)
    
    # Final export
    radar.export_graphs("final_graphs")
    radar.disable_graph_tracing()

if __name__ == "__main__":
    # Run the main demo
    pass  # Main demo code above
    
    # Optionally run enhanced demo
    # demo_enhanced_tracing()
