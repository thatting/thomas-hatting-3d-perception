#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(PointCloud2_msg):


# Exercise-2:

    # Convert ROS msg to PCL data
    cloud = ros_to_pcl(PointCloud2_msg)


    # Statistical Outlier Filtering:

    filename = 'initial_point_cloud.pcd'
    pcl.save(cloud,filename)

    # Much like the previous filters, we start by creating a filter object:
    outlier_filter = cloud.make_statistical_outlier_filter()

    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(50)

    # Set threshold scale factor
    x = 1.0

    # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(x)

    # Finally call the filter function for magic
    cloud = outlier_filter.filter()

    filename = 'statistical_filter_inliers.pcd'
    pcl.save(cloud, filename)


    outlier_filter.set_negative(True)
    filename = 'statistical_filter_outliers.pcd'
    pcl.save(outlier_filter.filter(), filename)


    # Voxel Grid Downsampling:

    # Create a VoxelGrid filter object for our input point cloud
    vox = cloud.make_voxel_grid_filter()

    # Choose a voxel (also known as leaf) size
    # Note: this (1) is a poor choice of leaf size
    # Experiment and find the appropriate size!
    LEAF_SIZE = 0.01    #0.01

    # Set the voxel (or leaf) size
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

    # Call the filter function to obtain the resultant downsampled point cloud
    cloud_filtered = vox.filter()
    filename = 'voxel_downsampled.pcd'
    pcl.save(cloud_filtered, filename)

    # PassThrough filter
    # Create a PassThrough filter object.
    passthrough = cloud_filtered.make_passthrough_filter()

    # Assign axis and range to the passthrough filter object.
    filter_axis = 'z'
    passthrough.set_filter_field_name (filter_axis)
    axis_min = 0.6      #Set to 0.6
    axis_max = 1.0      #Set to 1.0 
    passthrough.set_filter_limits (axis_min, axis_max)

    # Finally use the filter function to obtain the resultant point cloud
    cloud_filtered = passthrough.filter()
    filename = 'pass_through_filtered.pcd'
    pcl.save(cloud_filtered, filename)


    # RANSAC Plane Segmentation:

    # Create the segmentation object
    seg = cloud_filtered.make_segmenter()

    # Set the model you wish to fit
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    # Max distance for a point to be considered fitting the model
    # Experiment with different values for max_distance
    # for segmenting the table
    max_distance = 0.005     #Set to 0.005
    seg.set_distance_threshold(max_distance)

    # Call the segment function to obtain set of inlier indices and model coefficients
    inliers, coefficients = seg.segment()


    # Extract inliers and outliers:

    # Extract inliers
    extracted_inliers = cloud_filtered.extract(inliers, negative=False)
    filename = 'extracted_inliers.pcd'
    pcl.save(extracted_inliers, filename)

    cloud_table = extracted_inliers       #Assign inliers to cloud_table

    # Extract outliers
    extracted_outliers = cloud_filtered.extract(inliers, negative=True)
    filename = 'extracted_outliers.pcd'
    pcl.save(extracted_outliers, filename)

    cloud_objects = extracted_outliers    #Assign outliers to cloud_objects


    # Euclidean Clustering:

    white_cloud = XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()

    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold
    # as well as minimum and maximum cluster size (in points)
    # NOTE: These are poor choices of clustering parameters
    # Your task is to experiment and find values that work for segmenting objects.
    ec.set_ClusterTolerance(0.02)     #Set to 0.02
    ec.set_MinClusterSize(50)         #Set to 50
    ec.set_MaxClusterSize(20000)      #Set to 20000
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()

    # Create Cluster-Mask Point Cloud to visualize each cluster separately

    #Assign a color corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                            white_cloud[indice][1],
                                            white_cloud[indice][2],
                                            rgb_to_float(cluster_color[j])])

    #Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # Convert PCL data to ROS messages

    ros_cloud_objects = pcl_to_ros(cloud_objects)
    ros_cloud_table   = pcl_to_ros(cloud_table)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    # Publish ROS messages

    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cluster_cloud)



# Exercise-3:

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []
    for index, pts_list in enumerate(cluster_indices):

        # Grab the points for the cluster
        pcl_cluster = cloud_objects.extract(pts_list)

        # Convert the cluster from pcl to ROS using helper function
        ros_cluster=pcl_to_ros(pcl_cluster)

        # Compute the associated feature vector
        # Extract histogram features
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction
        # retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)


        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))


        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)


    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    # Publish the list of detected objects
    # This is the output you'll need to complete the upcoming project!
    detected_objects_pub.publish(detected_objects)

    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass


# function to load parameters and request PickPlace service
def pr2_mover(detected_object_list):

    # Initialize variables

    TEST_SCENE_NUM = Int32()   #Create object of class Int32()  - message type std_msgs/Int
    OBJECT_NAME = String()     #Create object of class String() - message type std_msgs/String
    WHICH_ARM = String()       #Create object of class String() - message type std_msgs/String
    PICK_POSE = Pose()         #Create object of class Pose()   - message type geometry_msgs/Pose
    PLACE_POSE = Pose()        #Create object of class Pose()   - message type geometry_msgs/Pose


    # Get/Read parameters

    object_list_param = rospy.get_param('/object_list')      #Get object_list_param from ROS parameter server
    dropbox_param = rospy.get_param('/dropbox')              #Get dropbox_param from ROS parameter server


    # Parse parameters into individual variables

    object_name =   []                #Initialize list of object names of pick list
    object_group =  []                #Initialize list of object groups of pick list
    dropbox_position = []             #Initialize list of dropbox positions


    for i in range(0, len(object_list_param)):
        object_name.append(object_list_param[i]['name'])               #Parse through object names
        object_group.append(object_list_param[i]['group'])             #Parse through object groups

    for i in range(0, len(dropbox_param)):
        dropbox_position.append(dropbox_param[i]['position'])          #Parse through dropbox positions


    # Rotate PR2 in place to capture side tables for the collision map
    # ** To be done at later stage **


    # Loop through the pick list:

    labels = []          #Initiate list of labels for detected objecsts
    centroids = []       #Initiate list of centroids for detected objects
    dict_list = []       #Initiate list of dictionaries for ROS messages for yaml files
    point_clouds = []    #Initiate list of point clouds for detected objects

    for object in detected_object_list:                        #Loop through detected objects
        labels.append(object.label)
        points_arr = ros_to_pcl(object.cloud).to_array()
        centroids.append(np.mean(points_arr, axis=0)[:3])      #Create list of individual centroids
        point_clouds.append(object.cloud)                      #Create list of individual point clouds 


    for i in range(0, len(object_list_param)):


        # Get the PointCloud for a given object and obtain it's centroid

        for j in range(0, len(detected_object_list)):

            if object_name[i] == labels[j] :
               pointcloud_target_object = point_clouds[j]                 #Get point cloud of target object
               centroid_target_object_x = np.asscalar(centroids[j][0])    #Convert centroid's x-coordinate to 'float' (scalar notation)
               centroid_target_object_y = np.asscalar(centroids[j][1])    #Convert centroid's y-coordinate to 'float' (scalar notation)
               centroid_target_object_z = np.asscalar(centroids[j][2])    #Convert centroid's z-coordinate to 'float' (scalar notation)


        # Create 'place_pose' for the object

               if object_group[i] == "red":                               #If object group is "red" choose left-hand box
                   PLACE_POSE.position.x = dropbox_position[0][0]
                   PLACE_POSE.position.y = dropbox_position[0][1]
                   PLACE_POSE.position.z = dropbox_position[0][2]
               else:                                                      #If object group is anything else (ie. "green") choose right-hand box
                   PLACE_POSE.position.x = dropbox_position[1][0]
                   PLACE_POSE.position.y = dropbox_position[1][1]
                   PLACE_POSE.position.z = dropbox_position[1][2]


        # Assign the arm to be used for pick_place

               if object_group[i] == "red":                               #If object group is "red" choose left arm of robot
                   WHICH_ARM.data = "left"
	       else:                                                      #If object group is anything else (ie. "green") choose right arm of robot
                   WHICH_ARM.data = "right"


        # Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format

               TEST_SCENE_NUM.data = 1                                #Assign test_scene_number (1,2 or 3)

               OBJECT_NAME.data = object_list_param[i]['name']        #Assign object_name

               PICK_POSE.position.x = centroid_target_object_x        #Assign x-coordinate of target object's centroid
               PICK_POSE.position.y = centroid_target_object_y        #Assign y-coordinate of target object's centroid
               PICK_POSE.position.z = centroid_target_object_z        #Assign z-coordinate of target object's centroid

               yaml_dict = make_yaml_dict(TEST_SCENE_NUM, WHICH_ARM, OBJECT_NAME, PICK_POSE, PLACE_POSE)      #For each target object call function make_yaml_dict() to create dictionary
               dict_list.append(yaml_dict)                                                                    #Append to list of dictionaries


        # Wait for 'pick_place_routine' service to come up

        # **To be done as part of "PR2 Collision Avoidance" and "PR2 Motion" at a later stage**

               #rospy.wait_for_service('pick_place_routine')

               #try:
                   #pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # Insert your message variables to be sent as a service request
                   #resp = pick_place_routine(TEST_SCENE_NUM, OBJECT_NAME, WHICH_ARM, PICK_POSE, PLACE_POSE)

                   #print ("Response: ",resp.success)

               #except rospy.ServiceException, e:
                   #print ("Service call failed")       #Format error corrected in print statement


    # Output your request parameters into output .yaml file

    yaml_filename = 'output_1.yaml'          #Define file name (output_1.yaml, output_2.yaml or output_3.yaml)
    send_to_yaml(yaml_filename, dict_list)   #Function call to save dict_list to .yaml file


if __name__ == '__main__':

    # ROS node initialization
    rospy.init_node('object_recognition', anonymous=True)    #Create ROS node 'object_recognition'

    # Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)     #Create subscriber to camera point cloud data

    # Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)                     #Create publisher for objects (after RANSAC filtering)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)                         #Create publisher for table (after RANSAC filtering)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)                     #Create publisher for clusters (after Euclidean Clustering)
    object_markers_pub   = rospy.Publisher("/object_markers", Marker, queue_size=1)                  #Create publisher for object markers
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)  #Create publisher for detected objects


    # Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))  #Load file 'model.sav' created by feature capturing and SVM training
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']


    # Initialize color_list
    get_color_list.color_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()

