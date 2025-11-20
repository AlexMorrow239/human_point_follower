#!/usr/bin/env python3

import rospy
from std_msgs.msg import Bool, Float64
from geometry_msgs.msg import PoseArray, Twist
from sensor_msgs.msg import CameraInfo, JointState
import math


class PIDController:
    """Simple PID controller for smooth tracking"""
    def __init__(self, kp, ki, kd, max_output=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output
        
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None
        
    def compute(self, error, current_time):
        """Compute PID output"""
        if self.last_time is None:
            self.last_time = current_time
            return 0.0
            
        dt = (current_time - self.last_time).to_sec()
        if dt <= 0:
            return 0.0
            
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self.integral += error * dt
        i_term = self.ki * self.integral
        
        # Derivative term
        d_term = self.kd * (error - self.last_error) / dt if dt > 0 else 0.0
        
        # Compute output
        output = p_term + i_term + d_term
        
        # Apply limits
        if self.max_output:
            output = max(-self.max_output, min(output, self.max_output))
            
        self.last_error = error
        self.last_time = current_time
        
        return output
    
    def reset(self):
        """Reset controller state"""
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None


class HumanTracker:
    """Scan for humans, then track them continuously once found"""

    def __init__(self):
        rospy.init_node('human_tracker')

        # State management - Added TRACKING state
        self.state = "INITIALIZING"  # States: INITIALIZING, SCANNING, HUMAN_FOUND, TRACKING, LOST_HUMAN, SCAN_COMPLETE
        
        # Scanning parameters
        self.scan_angular_speed = rospy.get_param('/scanning/angular_speed', 0.3)  # rad/s for base rotation
        self.head_tilt_angle = rospy.get_param('/scanning/head_tilt_angle', 0.0)  # 0 for level
        self.head_pan_angle = rospy.get_param('/scanning/head_pan_angle', 0.0)  # Center position
        
        # Track rotation for full circle detection
        self.total_rotation = 0.0
        self.last_time = None
        self.full_rotation_threshold = 2 * math.pi  # 360 degrees in radians
        
        # Detection parameters
        self.min_confidence = rospy.get_param('/tracking/min_confidence', 0.3)
        self.detection_count = 0
        self.detection_threshold = 5  # Need 5 consecutive detections to confirm
        
        # State variables
        self.human_detected = False
        self.camera_width = 640  # default
        self.camera_height = 480  # default
        self.camera_info_received = False
        
        # Joint state tracking
        self.current_head_pan = None
        self.current_head_tilt = None
        self.joint_states_received = False
        
        # Tracking parameters from yaml
        self.control_rate = rospy.get_param('/tracking/control_rate', 20)
        self.target_x = rospy.get_param('/tracking/target_x', 320)
        self.target_y = rospy.get_param('/tracking/target_y', 240)
        self.deadband_x = rospy.get_param('/tracking/deadband_x', 10)
        self.deadband_y = rospy.get_param('/tracking/deadband_y', 10)
        self.max_pan_velocity = rospy.get_param('/tracking/max_pan_velocity', 0.5)
        self.max_tilt_velocity = rospy.get_param('/tracking/max_tilt_velocity', 0.5)
        
        # PID controllers for tracking
        pan_kp = rospy.get_param('/tracking/pid_pan/kp', 0.001)
        pan_ki = rospy.get_param('/tracking/pid_pan/ki', 0.0001)
        pan_kd = rospy.get_param('/tracking/pid_pan/kd', 0.0005)
        
        tilt_kp = rospy.get_param('/tracking/pid_tilt/kp', 0.001)
        tilt_ki = rospy.get_param('/tracking/pid_tilt/ki', 0.0001)
        tilt_kd = rospy.get_param('/tracking/pid_tilt/kd', 0.0005)
        
        self.pan_pid = PIDController(pan_kp, pan_ki, pan_kd, self.max_pan_velocity)
        self.tilt_pid = PIDController(tilt_kp, tilt_ki, tilt_kd, self.max_tilt_velocity)
        
        # Base rotation PID (stronger gains for base rotation)
        self.base_pid = PIDController(0.003, 0.0001, 0.001, max_output=0.8)
        
        # Tracking state
        self.current_keypoints = None
        self.lost_human_time = None
        self.lost_human_timeout = 2.0  # seconds before terminating after losing human
        
        # Head joint limits (typical HSR limits)
        self.head_pan_min = -3.839  # ~-220 degrees
        self.head_pan_max = 1.745   # ~100 degrees
        self.head_tilt_min = -0.524  # ~-30 degrees
        self.head_tilt_max = 1.047   # ~60 degrees

        # Subscribe to topics
        self.detection_sub = rospy.Subscriber(
            '/openpose/human_detected',
            Bool,
            self.detection_callback,
            queue_size=1
        )

        self.keypoints_sub = rospy.Subscriber(
            '/openpose/keypoints',
            PoseArray,
            self.keypoints_callback,
            queue_size=1
        )

        self.camera_info_sub = rospy.Subscriber(
            '/hsrb/head_rgbd_sensor/rgb/camera_info',
            CameraInfo,
            self.camera_info_callback,
            queue_size=1
        )
        
        # Subscribe to joint states to get current head position
        # Try both possible joint state topics
        joint_state_topics = [
            '/hsrb/joint_states',
            '/joint_states',
            '/hsrb/robot_state/joint_states'
        ]
        
        for topic in joint_state_topics:
            try:
                msg = rospy.wait_for_message(topic, JointState, timeout=1.0)
                self.joint_state_sub = rospy.Subscriber(topic, JointState, self.joint_state_callback, queue_size=1)
                rospy.loginfo(f"Subscribed to joint states on: {topic}")
                break
            except:
                continue

        # Publisher for base velocity commands
        self.cmd_vel_pub = rospy.Publisher(
            '/hsrb/command_velocity',
            Twist,
            queue_size=1
        )
        
        # Publishers for direct head joint control (simpler approach for Isaac Sim)
        self.head_pan_pub = rospy.Publisher('/hsrb/head_pan_joint/command', Float64, queue_size=1)
        self.head_tilt_pub = rospy.Publisher('/hsrb/head_tilt_joint/command', Float64, queue_size=1)
        
        rospy.loginfo("Waiting for head joint command publishers to connect...")
        rospy.sleep(1.0)  # Give publishers time to establish connections

    def joint_state_callback(self, msg):
        """Track current head joint positions"""
        try:
            # Find head joint indices
            if 'head_pan_joint' in msg.name:
                pan_idx = msg.name.index('head_pan_joint')
                self.current_head_pan = msg.position[pan_idx]
            
            if 'head_tilt_joint' in msg.name:
                tilt_idx = msg.name.index('head_tilt_joint')
                self.current_head_tilt = msg.position[tilt_idx]
                
            # Log current position once
            if not self.joint_states_received and self.current_head_pan is not None and self.current_head_tilt is not None:
                rospy.loginfo(f"Current head position - Pan: {math.degrees(self.current_head_pan):.1f}°, Tilt: {math.degrees(self.current_head_tilt):.1f}°")
                self.joint_states_received = True
                
        except (ValueError, IndexError):
            pass

    def camera_info_callback(self, msg):
        """Update camera dimensions from camera info"""
        if not self.camera_info_received:
            self.camera_width = msg.width
            self.camera_height = msg.height
            # Update target to actual image center
            self.target_x = msg.width / 2
            self.target_y = msg.height / 2
            self.camera_info_received = True
            rospy.loginfo(f"Camera info received: {msg.width}x{msg.height}")
            rospy.loginfo(f"Target center updated to: ({self.target_x}, {self.target_y})")

    def detection_callback(self, msg):
        """Update human detection status"""
        if self.state == "SCANNING":
            if msg.data:
                self.detection_count += 1
                rospy.loginfo(f"Human detected! Count: {self.detection_count}/{self.detection_threshold}")
                
                # Confirm detection with threshold
                if self.detection_count >= self.detection_threshold:
                    self.state = "HUMAN_FOUND"
                    self.stop_base()
                    rospy.loginfo("HUMAN CONFIRMED - Transitioning to tracking")
            else:
                # Reset counter if we lose detection
                if self.detection_count > 0:
                    rospy.loginfo("Lost human detection, resetting counter")
                self.detection_count = 0
        
        elif self.state == "TRACKING":
            if not msg.data:
                # Lost human during tracking
                if self.lost_human_time is None:
                    self.lost_human_time = rospy.Time.now()
                    rospy.logwarn("Lost human during tracking!")
                else:
                    # Check timeout
                    elapsed = (rospy.Time.now() - self.lost_human_time).to_sec()
                    if elapsed > self.lost_human_timeout:
                        self.state = "LOST_HUMAN"
                        rospy.logerr(f"Human lost for {elapsed:.1f}s - terminating tracking")
            else:
                # Human reacquired
                if self.lost_human_time is not None:
                    rospy.loginfo("Human reacquired!")
                    self.lost_human_time = None
        
        self.human_detected = msg.data

    def keypoints_callback(self, msg):
        """Process keypoints for tracking"""
        # Store keypoints for tracking
        if len(msg.poses) > 0:
            self.current_keypoints = msg.poses
        else:
            self.current_keypoints = None

    def get_human_center(self):
        """Get the center position of the human from keypoints"""
        if self.current_keypoints is None or len(self.current_keypoints) == 0:
            return None, None
            
        # Average available keypoints (Neck and MidHip from OpenPose)
        x_sum = 0
        y_sum = 0
        count = 0
        
        for pose in self.current_keypoints:
            if pose.position.z > self.min_confidence:  # z stores confidence
                x_sum += pose.position.x
                y_sum += pose.position.y
                count += 1
        
        if count > 0:
            center_x = x_sum / count
            center_y = y_sum / count
            return center_x, center_y
            
        return None, None

    def track_human(self, current_time):
        """Perform tracking control using only base rotation"""
        center_x, center_y = self.get_human_center()
        
        if center_x is None:
            # No valid keypoints
            return
        
        # Calculate horizontal error (pixels from center)
        error_x = center_x - self.target_x
        
        # Apply deadband to reduce jitter
        if abs(error_x) < self.deadband_x:
            error_x = 0
        
        # Check if we need to move
        if error_x == 0:
            # Stop base if centered
            self.stop_base()
            rospy.loginfo_throttle(2.0, f"Human centered at x={center_x:.0f}")
            return
        
        # Use base rotation for all tracking
        # Simple proportional control
        base_gain = 0.003  # radians per pixel error - adjust this for responsiveness
        base_angular_vel = -error_x * base_gain  # negative for correct direction
        
        # Limit base velocity for safety
        max_base_velocity = 0.5  # rad/s
        base_angular_vel = max(-max_base_velocity, min(max_base_velocity, base_angular_vel))
        
        # Send base rotation command
        cmd_twist = Twist()
        cmd_twist.linear.x = 0.0
        cmd_twist.linear.y = 0.0
        cmd_twist.angular.z = base_angular_vel
        self.cmd_vel_pub.publish(cmd_twist)
        
        # Log tracking status
        rospy.loginfo_throttle(
            0.5,
            f"Tracking - Error: {error_x:.0f}px | "
            f"Human at: {center_x:.0f} (target: {self.target_x:.0f}) | "
            f"Base vel: {base_angular_vel:.3f} rad/s"
        )

    def set_head_position(self, pan, tilt):
        """Set head position using direct joint commands (works better in Isaac Sim)"""
        rospy.loginfo(f"Setting head to Pan: {math.degrees(pan):.1f}°, Tilt: {math.degrees(tilt):.1f}°")
        
        # Publish commands
        pan_msg = Float64()
        pan_msg.data = pan
        tilt_msg = Float64()
        tilt_msg.data = tilt
        
        # Send commands multiple times to ensure they're received
        for _ in range(3):
            self.head_pan_pub.publish(pan_msg)
            self.head_tilt_pub.publish(tilt_msg)
            rospy.sleep(0.1)
        
        # Wait for joints to reach position
        rospy.sleep(1.5)
        
        # Check if we reached the target position (with tolerance)
        if self.current_head_pan is not None and self.current_head_tilt is not None:
            pan_error = abs(self.current_head_pan - pan)
            tilt_error = abs(self.current_head_tilt - tilt)
            
            if pan_error < 0.1 and tilt_error < 0.1:  # Within ~5 degrees
                rospy.loginfo("Head positioned successfully")
                return True
            else:
                rospy.logwarn(f"Head positioning incomplete - Pan error: {math.degrees(pan_error):.1f}°, Tilt error: {math.degrees(tilt_error):.1f}°")
                rospy.logwarn("Continuing anyway...")
                return False
        else:
            rospy.logwarn("Cannot verify head position - no joint state feedback")
            return False

    def stop_base(self):
        """Stop base rotation"""
        stop_twist = Twist()
        stop_twist.linear.x = 0.0
        stop_twist.linear.y = 0.0
        stop_twist.angular.z = 0.0
        self.cmd_vel_pub.publish(stop_twist)

    def rotate_base(self, angular_velocity):
        """Send rotation command to base"""
        cmd_twist = Twist()
        cmd_twist.linear.x = 0.0
        cmd_twist.linear.y = 0.0
        cmd_twist.angular.z = angular_velocity
        self.cmd_vel_pub.publish(cmd_twist)

    def update_rotation_tracking(self, angular_velocity, dt):
        """Track total rotation amount"""
        self.total_rotation += abs(angular_velocity * dt)
        
        if self.total_rotation >= self.full_rotation_threshold:
            rospy.loginfo(f"Completed full rotation ({math.degrees(self.total_rotation):.1f}°)")
            self.state = "SCAN_COMPLETE"
            self.stop_base()
            return True
        return False

    def run(self):
        """Main control loop with scanning and tracking"""
        rate = rospy.Rate(self.control_rate)
        
        # Wait a moment for all connections to establish
        rospy.sleep(2.0)
        
        # Step 1: Position head to scanning position
        rospy.loginfo("Positioning head to scanning position...")
        self.set_head_position(self.head_pan_angle, self.head_tilt_angle)
        
        # Step 2: Start base rotation scan
        rospy.loginfo("Starting 360° base rotation scan for humans...")
        self.state = "SCANNING"
        self.last_time = rospy.Time.now()
        
        while not rospy.is_shutdown():
            current_time = rospy.Time.now()
            
            if self.state == "SCANNING":
                # Calculate time delta
                if self.last_time:
                    dt = (current_time - self.last_time).to_sec()
                    
                    # Update rotation tracking
                    full_rotation_complete = self.update_rotation_tracking(
                        self.scan_angular_speed, dt
                    )
                    
                    if full_rotation_complete:
                        break
                
                # Continue rotating
                self.rotate_base(self.scan_angular_speed)
                
                # Log progress
                rospy.loginfo_throttle(
                    2.0,
                    f"Scanning... Rotated {math.degrees(self.total_rotation):.1f}° / 360°"
                )
                
            elif self.state == "HUMAN_FOUND":
                # Transition to tracking
                rospy.loginfo("=" * 50)
                rospy.loginfo("HUMAN DETECTED - Starting tracking mode")
                rospy.loginfo("=" * 50)
                
                # Center the head for tracking (only base will move)
                rospy.loginfo("Centering head for base-only tracking...")
                self.set_head_position(0.0, 0.0)  # Center position
                
                # Reset PIDs for smooth tracking start
                self.pan_pid.reset()
                self.tilt_pid.reset()
                self.base_pid.reset()
                
                self.state = "TRACKING"
                # Continue to next iteration to start tracking
                
            elif self.state == "TRACKING":
                # Perform tracking control
                self.track_human(current_time)
                
                # Check if still tracking or lost
                if self.state == "LOST_HUMAN":
                    # Human lost - terminate
                    rospy.logerr("Human lost - terminating tracking")
                    break
                
            elif self.state == "SCAN_COMPLETE":
                rospy.loginfo("Full rotation completed without finding human. Terminating.")
                break
            
            self.last_time = current_time
            rate.sleep()
        
        # Ensure base is stopped
        self.stop_base()
        
        # Final status
        if self.state == "LOST_HUMAN":
            rospy.loginfo("=" * 50)
            rospy.loginfo("TRACKING TERMINATED: Human lost for too long")
            rospy.loginfo("=" * 50)
        elif self.state == "SCAN_COMPLETE":
            rospy.loginfo("=" * 50)
            rospy.loginfo("SCAN RESULT: No human found after full rotation")
            rospy.loginfo("=" * 50)
        
        rospy.loginfo("Human tracker node terminating...")


if __name__ == '__main__':
    try:
        tracker = HumanTracker()
        tracker.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        # Ensure we stop the base on exit
        if 'tracker' in locals():
            tracker.stop_base()