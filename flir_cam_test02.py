"""
Test converting datatype from camera SDK (Spinnaker) to Numpy/OpenCV datatype

Exit codes:
100 - Exited at end of program successfully
99 - Not enough cameras, no cameras found.
"""

import cv2
import numpy as np
import PySpin


def print_node_value(nodemap, node_name):
    """
    Print the value of an enumeration node

    :param nodemap:
    :param node_name:
    :return:
    """

    result = True

    node = PySpin.CEnumerationPtr(nodemap.GetNode(node_name))

    if not PySpin.IsAvailable(node) or not PySpin.IsReadable(node):
        print('Unable to read node: {}'.format(node_name))
        return False

    display_name = node.GetName()
    value = node.ToString()
    print('{}: {}'.format(display_name, value))

    return result


def print_device_info(nodemap):
    """
    This function prints the device information of the camera from the transport
    layer; please see NodeMapInfo example for more in-depth comments on printing
    device information from the nodemap.

    :param nodemap: Transport layer device nodemap.
    :type nodemap: INodeMap
    :returns: True if successful, False otherwise.
    :rtype: bool
    """

    print("*** DEVICE INFORMATION ***\n")

    try:
        result = True
        node_device_information = PySpin.CCategoryPtr(nodemap.GetNode('DeviceInformation'))

        if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
            features = node_device_information.GetFeatures()
            for feature in features:
                node_feature = PySpin.CValuePtr(feature)
                print('{0}: {1}'.format(node_feature.GetName(),
                                        node_feature.ToString() if PySpin.IsReadable(node_feature) else 'Node not available'))

        else:
            print('Device control information not available.')

    except PySpin.SpinnakerException as ex:
        print('Error: {}'.format(ex))
        return False

    return result


def acquire_images(cam, nodemap, nodemap_tldevice):
    """
    This function acquires and saves a single image from a device.

    :param cam: Camera to acquire images from.
    :param nodemap: Device nodemap.
    :param nodemap_tldevice: Transport layer device nodemap.
    :type cam: CameraPtr
    :type nodemap: INodeMap
    :type nodemap_tldevice: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """

    print("*** IMAGE ACQUISITION ***\n")

    try:
        result = True

        # Print the PixelColor format

        print_node_value(nodemap, "PixelFormat")

        # In order to access the node entries, they have to be casted to a pointer type (CEnumerationPtr here)
        node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
            print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
            return False

        # Retrieve entry node from enumeration node
        node_acquisition_mode_singleframe = node_acquisition_mode.GetEntryByName('SingleFrame')
        if not PySpin.IsAvailable(node_acquisition_mode_singleframe) or not PySpin.IsReadable(node_acquisition_mode_singleframe):
            print('Unable to set acquisition mode to single frame (entry retrieval). Aborting...')
            return False

        # Retrieve integer value from entry node
        acquisition_mode_singleframe = node_acquisition_mode_singleframe.GetValue()

        # Set integer value from entry node as new value of enumeration node
        node_acquisition_mode.SetIntValue(acquisition_mode_singleframe)
        print('Acquisition mode set to single frame.')

        # Begin acquiring images
        #  *** LATER ***
        #  Image acquisition must be ended when no more images are needed.
        cam.BeginAcquisition()
        print('Acquiring images...')

        # Retrieve next received image
        image_result = cam.GetNextImage()

        if image_result.IsIncomplete():
            print('Image incomplete with image status {} ...'.format(image_result.GetImageStatus()))
        else:
            width = image_result.GetWidth()
            height = image_result.GetHeight()
            print('Grabbed Image, width = {0}, height = {1}'.format(width, height))

            # Convert ImagePtr datatype to numpy ndarray for use with OpenCV
            image_array = image_result.GetNDArray()

            # OpenCV to convert from BayerGB8 to BGR for display
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BAYER_GB2BGR)

            image_result.Release()

        #  End acquisition
        #
        #  *** NOTES ***
        #  Ending acquisition appropriately helps ensure that devices clean up
        #  properly and do not need to be power-cycled to maintain integrity.
        cam.EndAcquisition()

    except PySpin.SpinnakerException as ex:
        print('Error: {}'.format(ex))
        return False

    return result, image_array


def run_single_camera(cam):
    """
    This function acts as the body of the example; please see NodeMapInfo example
    for more in-depth comments on setting up cameras.

    :param cam: Camera to run on.
    :type cam: CameraPtr
    :return: True if successful, False otherwise.
    :rtype: bool
    """

    try:
        result = True

        # Retrieve (transport layer) TL device nodemap and print device information
        nodemap_tldevice = cam.GetTLDeviceNodeMap()

        # print device info
        result &= print_device_info(nodemap_tldevice)

        # Initialize the camera
        cam.Init()

        # Retrieve GenICam nodemap
        nodemap = cam.GetNodeMap()

        # Acquire images
        ret, img = acquire_images(cam, nodemap, nodemap_tldevice)
        result &= ret

        # Deinitialize camera
        cam.DeInit()

    except PySpin.SpinnakerException as ex:
        print('Error: {}'.format(ex))
        result = False

    return result, img


cv2.namedWindow('Image', cv2.WINDOW_KEEPRATIO)

# Retrieve singleton reference to system object
system = PySpin.System.GetInstance()

# Retrieve list of cameras from the system
cam_list = system.GetCameras()

num_cameras = cam_list.GetSize()
print('Number of Cameras: {}'.format(num_cameras))

# Finish if there are no cameras
if num_cameras == 0:
    # Clear camera list before releasing system
    cam_list.Clear()

    # Release system
    system.ReleaseInstance()

    print('Not enough cameras!')
    exit(99)

# Get first camera object
cam = cam_list.GetByIndex(0)

# Call the function to run the camera object
result, img = run_single_camera(cam)
print('Camera example complete...')

# Clean up camera object
del cam

# Clear camera list before releasing system
cam_list.Clear()

# Release system
system.ReleaseInstance()

# Display Image with OpenCV
cv2.imshow('Image', img)
cv2.waitKey(0)
# Clear all cv2 windows
cv2.destroyAllWindows()

# Exit the program
exit(100)
