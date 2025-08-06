# Decorator-based instrumentation

Decorator-Based Method Instrumentation for the radar detection application. 

## Key Benefits

1. Minimal Code Changes: Only need to add decorators to existing methods:
```python
@trace_kalman_method("kalman_predict")  # <-- Only change
def predict(self, dt: float):
    # Original Kalman filter code completely unchanged
    F = np.eye(6)
    F[0:3, 3:6] = np.eye(3) * dt
    # ... rest of original logic
```
2. Automatic Operation Detection: The decorator uses frame inspection to:

   - Capture local variables before/after method execution
   - Detect new numpy arrays created during execution
   - Infer operation types based on Kalman filter patterns
   - Build dependency graphs showing data flow

3. Smart Kalman Filter Awareness: Recognizes common patterns:

   - predicted_state = F @ state → "state_prediction"
   - K = covariance @ H.T @ np.linalg.inv(S) → "kalman_gain"
   - y = z - predicted_measurement → "innovation_calculation"

## How to Use

Step 1: Add decorators to your existing radar_detectionv2.py:
```python
class AircraftTrack:
    @trace_kalman_method("kalman_predict")
    def predict(self, dt: float):
        # Your original code unchanged

class RadarStation:  
    @trace_kalman_method("kalman_update")
    def update_track_kalman(self, track, detection):
        # Your original code unchanged
```
Step 2: Wrap radar processing with tracing:
```python
start_kalman_tracing()
# Your existing radar processing loop
stop_kalman_tracing()
export_kalman_graph()
```

## What Gets Captured

The instrumentation automatically captures:

 - Matrix Operations: F @ state, H.T, np.linalg.inv(S)
 - Variable Dependencies: Which arrays are inputs to create new arrays
 - Operation Sequence: Order of computations within methods
 - Shapes & Types: Dimensions and data types of all intermediate results
 - Method Context: Whether the operation occurred in predict vs update

## Output Formats

 - JSON: Machine-readable computational graph with full metadata
 - DOT: Graphviz visualization showing operation flow and dependencies
 - Statistics: Operation counts, method coverage, variable tracking

The decorator approach provides an FX-style computational graph while preserving the original Kalman filter implementation completely unchanged. The overhead is minimal when tracing is disabled, and you get comprehensive insight into the mathematical operations when tracing is enabled.
