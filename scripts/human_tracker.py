#!/usr/bin/env python3

import rospy
import actionlib
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import CameraInfo
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint
import math


class PIDController:
    """Simple PID controller for tracking"""
    def __init__(self, kp, ki, kd, max_output=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output

        self.prev_error = 0.0
        self.integral = 0.0

    def reset(self):
        """Reset controller state"""
        self.prev_error = 0.0
        self.integral = 0.0

    def update(self, error, dt):
        """Calculate PID output"""
        # Proportional term
        p_term = self.kp * error

        # Integral term
        self.integral += error * dt
        i_term = self.ki * self.integral

        # Derivative term
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        d_term = self.kd * derivative

        # Calculate total output
        output = p_term + i_term + d_term

        # Apply output limits
        if self.max_output is not None:
            output = max(-self.max_output, min(self.max_output, output))

        self.prev_error = error

        return output


class HumanTracker:
    """Track human with HSR head using PID control"""

    def __init__(self):
        rospy.init_node('human_tracker')

        # Load parameters
        self.control_rate = rospy.get_param('/tracking/control_rate', 20)
        self.target_x = rospy.get_param('/tracking/target_x', 320)
        self.target_y = rospy.get_param('/tracking/target_y', 240)

        # PID parameters
        pan_kp = rospy.get_param('/tracking/pid_pan/kp', 0.001)
        pan_ki = rospy.get_param('/tracking/pid_pan/ki', 0.0001)
        pan_kd = rospy.get_param('/tracking/pid_pan/kd', 0.0005)

        tilt_kp = rospy.get_param('/tracking/pid_tilt/kp', 0.001)
        tilt_ki = rospy.get_param('/tracking/pid_tilt/ki', 0.0001)
        tilt_kd = rospy.get_param('/tracking/pid_tilt/kd', 0.0005)

        # Safety limits
        self.max_pan_vel = rospy.get_param('/tracking/max_pan_velocity', 0.5)
        self.max_tilt_vel = rospy.get_param('/tracking/max_tilt_velocity', 0.5)
        self.min_confidence = rospy.get_param('/tracking/min_confidence', 0.3)

        # Deadband to prevent jitter
        self.deadband_x = rospy.get_param('/tracking/deadband_x', 10)
        self.deadband_y = rospy.get_param('/tracking/deadband_y', 10)

        # Initialize PID controllers
        self.pan_controller = PIDController(pan_kp, pan_ki, pan_kd, self.max_pan_vel)
        self.tilt_controller = PIDController(tilt_kp, tilt_ki, tilt_kd, self.max_tilt_vel)

        # State variables
        self.human_detected = False
        self.torso_center_x = None
        self.torso_center_y = None
        self.camera_width = 640  # default
        self.camera_height = 480  # default
        self.camera_info_received = False

        # Current joint positions
        self.current_pan = 0.0
        self.current_tilt = 0.0

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

        # Initialize action client for head control
        rospy.loginfo("Connecting to head trajectory controller...")
        self.head_client = actionlib.SimpleActionClient(
            '/hsrb/head_trajectory_controller/follow_joint_trajectory',
            FollowJointTrajectoryAction
        )

        # Wait for action server with timeout
        if not self.head_client.wait_for_server(timeout=rospy.Duration(10.0)):
            rospy.logwarn("Head trajectory action server not available. Tracking will not work.")
            self.head_client = None
        else:
            rospy.loginfo("Connected to head trajectory controller!")

        rospy.loginfo(f"Human Tracker initialized at {self.control_rate} Hz")
        rospy.loginfo(f"Target image center: ({self.target_x}, {self.target_y})")
        rospy.loginfo(f"PID Pan: Kp={pan_kp}, Ki={pan_ki}, Kd={pan_kd}")
        rospy.loginfo(f"PID Tilt: Kp={tilt_kp}, Ki={tilt_ki}, Kd={tilt_kd}")

    def camera_info_callback(self, msg):
        """Update camera dimensions from camera info"""
        if not self.camera_info_received:
            self.camera_width = msg.width
            self.camera_height = msg.height
            self.target_x = msg.width / 2.0
            self.target_y = msg.height / 2.0
            self.camera_info_received = True
            rospy.loginfo(f"Camera info received: {msg.width}x{msg.height}")
            rospy.loginfo(f"Updated target center: ({self.target_x}, {self.target_y})")

    def detection_callback(self, msg):
        """Update human detection status"""
        was_detected = self.human_detected
        self.human_detected = msg.data

        if not self.human_detected and was_detected:
            # Human lost - reset controllers
            self.pan_controller.reset()
            self.tilt_controller.reset()
            self.torso_center_x = None
            self.torso_center_y = None
            rospy.loginfo("Human lost - resetting controllers")

    def keypoints_callback(self, msg):
        """Process keypoints and calculate torso center"""
        if not self.human_detected:
            return

        # Extract Neck and MidHip from PoseArray
        # Format: poses[0] = Neck, poses[1] = MidHip
        # position.x = pixel_x, position.y = pixel_y, position.z = confidence

        neck = None
        midhip = None

        if len(msg.poses) >= 1:
            # First pose is Neck
            if msg.poses[0].position.z >= self.min_confidence:
                neck = (msg.poses[0].position.x, msg.poses[0].position.y)

        if len(msg.poses) >= 2:
            # Second pose is MidHip
            if msg.poses[1].position.z >= self.min_confidence:
                midhip = (msg.poses[1].position.x, msg.poses[1].position.y)

        # Calculate torso center (average of neck and midhip)
        if neck is not None and midhip is not None:
            self.torso_center_x = (neck[0] + midhip[0]) / 2.0
            self.torso_center_y = (neck[1] + midhip[1]) / 2.0
        elif neck is not None:
            # Use only neck if midhip not available
            self.torso_center_x = neck[0]
            self.torso_center_y = neck[1]
        elif midhip is not None:
            # Use only midhip if neck not available
            self.torso_center_x = midhip[0]
            self.torso_center_y = midhip[1]
        else:
            # No valid keypoints
            self.torso_center_x = None
            self.torso_center_y = None

    def send_head_command(self, pan_position, tilt_position, duration=0.1):
        """Send head trajectory command via action client"""
        if self.head_client is None:
            return

        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = ["head_pan_joint", "head_tilt_joint"]

        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = [pan_position, tilt_position]
        point.velocities = [0.0, 0.0]
        point.time_from_start = rospy.Duration(duration)

        goal.trajectory.points.append(point)
        goal.trajectory.header.stamp = rospy.Time.now()

        # Send goal (non-blocking)
        self.head_client.send_goal(goal)

    def control_loop(self):
        """Main control loop"""
        rate = rospy.Rate(self.control_rate)
        dt = 1.0 / self.control_rate

        rospy.loginfo("Starting control loop...")

        while not rospy.is_shutdown():
            if self.human_detected and self.torso_center_x is not None:
                # Calculate error in pixels
                error_x = self.torso_center_x - self.target_x
                error_y = self.torso_center_y - self.target_y

                # Apply deadband to prevent jitter
                if abs(error_x) < self.deadband_x:
                    error_x = 0.0
                if abs(error_y) < self.deadband_y:
                    error_y = 0.0

                # Calculate PID outputs (angular velocities)
                pan_velocity = -self.pan_controller.update(error_x, dt)  # Negative for correct direction
                tilt_velocity = self.tilt_controller.update(error_y, dt)

                # Update joint positions
                self.current_pan += pan_velocity * dt
                self.current_tilt += tilt_velocity * dt

                # Apply joint limits (HSR typical limits)
                self.current_pan = max(-3.84, min(1.75, self.current_pan))
                self.current_tilt = max(-1.57, min(0.52, self.current_tilt))

                # Send command to head
                self.send_head_command(self.current_pan, self.current_tilt, duration=dt*2)

                # Log status (throttled)
                if abs(error_x) > 0 or abs(error_y) > 0:
                    rospy.loginfo_throttle(
                        2.0,
                        f"Tracking - Error: ({error_x:.1f}, {error_y:.1f})px | "
                        f"Torso: ({self.torso_center_x:.1f}, {self.torso_center_y:.1f})px | "
                        f"Head: pan={math.degrees(self.current_pan):.1f}°, tilt={math.degrees(self.current_tilt):.1f}°"
                    )
                else:
                    rospy.loginfo_throttle(5.0, "Human centered in frame")

            rate.sleep()


if __name__ == '__main__':
    try:
        tracker = HumanTracker()
        tracker.control_loop()
    except rospy.ROSInterruptException:
        pass
