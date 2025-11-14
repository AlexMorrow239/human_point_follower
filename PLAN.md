# HSR Human Pointing Follower - Implementation Plan

## Context
- **Environment**: HSR robot in Omniverse simulation
- **OpenPose**: Successfully installed and tested at `/home/csc752/hsr_robocanes_omniverse/src/openpose/`
- **Package**: `human_point_follower` in catkin workspace
- **Working Camera Topic**: `/hsrb/head_rgbd_sensor/rgb/image_rect_color`
- **Working Depth Topic**: `/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw`

## Completed Steps
**Phase 1.1**: OpenPose Installation
- OpenPose installed and working with BODY_25 model
- Python bindings functional

**Phase 1.2**: Basic ROS-OpenPose Bridge
- Simple bridge created and tested (`openpose_bridge.py`)
- Publishing `/openpose/human_detected` and `/openpose/skeleton_image`
- Configuration via `params.yaml`

## Remaining Implementation Steps

### Phase 2: Human Detection & Tracking

#### Step 2.1: Human Presence Detection
- Already detecting humans with minimum keypoint threshold
- Publishing boolean to `/openpose/human_detected`

#### Step 2.2: Human Tracking with Head Movement
- Calculate human torso center from OpenPose keypoints (indices 1=Neck, 8=MidHip)
- Implement PID controller for HSR head joints (`head_pan_joint`, `head_tilt_joint`)
- Use `/hsrb/head_trajectory_controller/follow_joint_trajectory` action server
- **Benchmark**: Robot maintains human in center of camera frame as person moves

### Phase 3: Pointing Detection

#### Step 3.1: Arm Pose Recognition
- Detect extended arm using BODY_25 keypoints:
  - Right arm: indices 2 (RShoulder), 3 (RElbow), 4 (RWrist)
  - Left arm: indices 5 (LShoulder), 6 (LElbow), 7 (LWrist)
- Calculate arm straightness (angle between upper and lower arm vectors)
- **Benchmark**: Publish `/pointing_detected` boolean when arm is extended

#### Step 3.2: 2D Pointing Direction
- Extract 2D pointing vector from shoulder → elbow → wrist
- Convert from image coordinates to camera frame
- **Benchmark**: Visualize pointing direction in RViz as arrow marker

### Phase 4: Ground Point Calculation

#### Step 4.1: 3D Point Reconstruction
- Use depth image from `/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw`
- Get 3D position of wrist using depth value at wrist pixel location
- Extend pointing vector from wrist position
- **Benchmark**: Publish 3D pointing ray in camera frame

#### Step 4.2: Ray-Ground Intersection
- Transform pointing ray from camera frame to robot base frame (`base_link`)
- Calculate intersection with ground plane (z=0 in base frame)
- Use TF2 for coordinate transformations
- **Benchmark**: Place marker in RViz at calculated ground point

### Phase 5: Robot Movement

#### Step 5.1: Simple Direct Movement
- Subscribe to target ground point
- Use `/hsrb/command_velocity` for direct velocity commands
- Implement simple proportional controller for linear and angular velocity
- Stop at 0.5m from target
- **Benchmark**: Robot moves to within 0.5m of target point

#### Step 5.2: Safety Features
- Add timeout (10 seconds)
- Stop if human is no longer detected
- Stop if pointing gesture ends
- Maximum velocity limits (0.3 m/s linear)
- **Benchmark**: Robot stops safely when conditions change

## File Structure
```
human_point_follower/
├── CMakeLists.txt
├── package.xml
├── config/
│   └── params.yaml (COMPLETED)
├── launch/
│   ├── openpose.launch (COMPLETED)
│   ├── human_tracker.launch
│   └── full_system.launch
├── scripts/
│   ├── openpose_omniverse_bridge.py (COMPLETED)
│   ├── human_tracker.py
│   ├── pointing_detector.py
│   ├── ground_point_calculator.py
│   └── point_follower.py
└── msg/
    └── PointingInfo.msg (optional)
```

## Key Technical Details

### OpenPose BODY_25 Keypoints
- 0: Nose, 1: Neck, 2: RShoulder, 3: RElbow, 4: RWrist
- 5: LShoulder, 6: LElbow, 7: LWrist, 8: MidHip
- Each keypoint has [x, y, confidence] where x,y are pixel coordinates

### HSR Omniverse Topics
- RGB Camera: `/hsrb/head_rgbd_sensor/rgb/image_rect_color`
- Depth: `/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw`
- Camera Info: `/hsrb/head_rgbd_sensor/rgb/camera_info`
- Velocity Commands: `/hsrb/command_velocity`

### Current params.yaml Structure
```yaml
camera:
  rgb_topic: "/hsrb/head_rgbd_sensor/rgb/image_rect_color"
  depth_topic: "/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw"

openpose:
  model_folder: "/home/csc752/hsr_robocanes_omniverse/src/openpose/models/"
  model_pose: "BODY_25"
  net_resolution: "320x176"
  min_confidence: 0.3

human_detection:
  min_keypoints: 10
```

## Testing Strategy
1. Test each component independently before integration
2. Use RViz for visualization of all 3D data
3. Log key metrics (detection rate, pointing accuracy, movement success)
4. Start with stationary human, then add movement
5. Test in controlled Omniverse environment with single person