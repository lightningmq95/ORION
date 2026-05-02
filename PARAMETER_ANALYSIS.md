# Parameter Configuration Analysis

## Summary
- ✅ **EKF Node**: Fixed - parameter names now match code
- ✅ **LiDAR Odom**: Fixed - created dedicated params file  
- ✅ **All parameters accounted for**

---

## EKF Node (`extended_kalman_filter.py`)

### Parameters Required by Code:
| Parameter | Default | Type | Purpose |
|-----------|---------|------|---------|
| `process_noise_vx` | 0.1 | float | Linear velocity process noise (motion model uncertainty) |
| `process_noise_omega` | 0.5 | float | Angular velocity process noise (rotation model uncertainty) |
| `imu_heading_noise` | 0.05 | float | IMU yaw measurement noise (how much to trust IMU) |

### File: `ekf_params.yaml` ✅ FIXED
**Before (WRONG):**
```yaml
process_noise_vy: 0.1          # ❌ Not used in code
process_noise_yaw: 0.4         # ❌ Should be process_noise_omega
```

**After (CORRECT):**
```yaml
process_noise_vx: 0.25
process_noise_omega: 0.4       # ✅ Matches code
imu_heading_noise: 0.05
```

### Code Verification:
```python
process_vx    = self.get_parameter('process_noise_vx').value    ✅
process_omega = self.get_parameter('process_noise_omega').value  ✅
self.R_imu    = self.get_parameter('imu_heading_noise').value    ✅
```

---

## LiDAR Odom Node (`lidar_odom.py`)

### 12 Parameters Declared (NO params file existed):
| Parameter | Default | Type | Purpose |
|-----------|---------|------|---------|
| `max_range` | 8.0 | float | Max LiDAR range in meters |
| `voxel_size_map` | 0.10 | float | Fine voxel size for map storage |
| `voxel_size_icp` | 0.20 | float | Coarser voxel size for ICP |
| `min_points` | 20 | int | Min points required per scan |
| `max_map_points` | 20000 | int | Map storage limit |
| `odom_frame` | 'odom' | str | Odometry frame ID |
| `base_frame` | 'base_footprint' | str | Robot frame ID |
| `publish_tf` | True | bool | Publish TF transforms |
| `cov_x` | 0.05 | float | Pose X uncertainty |
| `cov_y` | 0.05 | float | Pose Y uncertainty |
| `cov_yaw` | 0.02 | float | Pose yaw uncertainty (rad²) |
| `cov_vx` | 0.10 | float | Velocity X uncertainty |
| `cov_vyaw` | 0.05 | float | Angular velocity uncertainty (rad/s²) |

### File: `lidar_odom_params.yaml` ✅ CREATED
New file created with all 12 parameters, defaults, and documentation.

---

## Parameter Propagation

### How to use the params files in launch file:
```python
# Include the param files when spawning nodes
params = [
    os.path.join(config_dir, 'ekf_params.yaml'),
    os.path.join(config_dir, 'lidar_odom_params.yaml')
]

lidar_node = Node(
    package='simple_robot_control',
    executable='lidar_odom',
    name='lidar_odom_node',
    parameters=params
)

ekf_node = Node(
    package='simple_robot_control',
    executable='ekf_node',
    name='ekf_node',
    parameters=params
)
```

---

## Physical Meaning of Parameters

### EKF Process Noise (Motion Model)
- **Higher values** = Trust prediction less, rely more on measurements
- **Lower values** = Trust constant-velocity model
- Current: `process_vx=0.25`, `process_omega=0.4` = Moderate trust in motion

### EKF Measurement Noise (Sensor Trust)
- **Higher values** = Trust sensor less
- **Lower values** = Trust sensor more
- Current: `imu_heading_noise=0.05` = High trust in IMU

### LiDAR Odom Covariances
- Uncertainty in final odometry estimate
- Used by robot_localization EKF for weighting measurements
- Current: Low uncertainty (0.05m, 0.02 rad) = High confidence
