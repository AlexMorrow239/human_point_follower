#!/usr/bin/env python3

import sys
import os
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge

# Add OpenPose Python path
sys.path.append('/home/csc752/hsr_robocanes_omniverse/src/openpose/build/python')
from openpose import pyopenpose as op

class OpenPoseOmniverseBridge:
    def __init__(self):
        rospy.init_node('openpose_omniverse_bridge')
        
        # Load parameters from yaml
        model_folder = rospy.get_param('/openpose/model_folder')
        model_pose = rospy.get_param('/openpose/model_pose', 'BODY_25')
        net_resolution = rospy.get_param('/openpose/net_resolution', '320x176')
        
        # Setup OpenPose
        params = dict()
        params["model_folder"] = model_folder
        params["model_pose"] = model_pose
        params["net_resolution"] = net_resolution
        
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()
        rospy.loginfo("OpenPose initialized!")
        
        # Setup ROS
        self.bridge = CvBridge()
        self.min_keypoints = rospy.get_param('/human_detection/min_keypoints', 10)
        
        # Subscribe to camera
        camera_topic = rospy.get_param('/camera/rgb_topic')
        self.image_sub = rospy.Subscriber(camera_topic, Image, self.image_callback, queue_size=1)
        
        # Publishers
        self.detection_pub = rospy.Publisher('/openpose/human_detected', Bool, queue_size=1)
        self.skeleton_image_pub = rospy.Publisher('/openpose/skeleton_image', Image, queue_size=1)
        
        rospy.loginfo(f"Subscribed to: {camera_topic}")
        
    def image_callback(self, msg):
        try:
            # Convert and process
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            datum = op.Datum()
            datum.cvInputData = cv_image
            self.opWrapper.emplaceAndPop(op.VectorDatum([datum]))
            
            # Check detection
            human_detected = False
            if datum.poseKeypoints is not None and len(datum.poseKeypoints) > 0:
                valid_keypoints = sum(1 for kp in datum.poseKeypoints[0] if kp[2] > 0.3)
                human_detected = valid_keypoints >= self.min_keypoints
                
                if human_detected:
                    rospy.loginfo_throttle(1.0, f"Human detected with {valid_keypoints} keypoints")
            
            # Publish results
            self.detection_pub.publish(Bool(human_detected))
            
            if datum.cvOutputData is not None:
                skeleton_msg = self.bridge.cv2_to_imgmsg(datum.cvOutputData, "bgr8")
                self.skeleton_image_pub.publish(skeleton_msg)
                
        except Exception as e:
            rospy.logerr(f"Error: {e}")

if __name__ == '__main__':
    try:
        bridge = OpenPoseOmniverseBridge()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass