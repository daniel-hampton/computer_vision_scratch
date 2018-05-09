"""
Test playing video from camera SDK (Spinnaker) with contour detection code.
Based on successful flir_cam_test_camera.py

Exit codes:
100 - Exited at end of program successfully
99 - Not enough cameras, no cameras found.
"""

import cv2
import numpy as np
import PySpin

trackbar_mode = True
default_thresh = 85
default_high_thresh = 255


def nothing():
    """ Empty function to supply to opencv trackbar object"""

    pass


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
    This function acquires begins acquiring images from the device and cleans up afterward.

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

        # In order to access the node entries, they have to be casted to a pointer type (CEnumerationPtr here)
        node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
            print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
            return False

        # Retrieve entry node from enumeration node
        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
        if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(node_acquisition_mode_continuous):
            print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
            return False

        # Retrieve integer value from entry node
        acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

        # Set integer value from entry node as new value of enumeration node
        node_acquisition_mode.SetIntValue(acquisition_mode_continuous)
        print('Acquisition mode set to single frame.')

        # Begin acquiring images
        #  *** LATER ***
        #  Image acquisition must be ended when no more images are needed.
        cam.BeginAcquisition()
        print('Acquiring video...')

        # Display the images/video
        result &= display_video(cam)

        print('Ending acquisition...')

        #  End acquisition
        #
        #  *** NOTES ***
        #  Ending acquisition appropriately helps ensure that devices clean up
        #  properly and do not need to be power-cycled to maintain integrity.
        cam.EndAcquisition()

    except PySpin.SpinnakerException as ex:
        print('Error: {}'.format(ex))
        return False

    return result


def display_video(cam):

    result = True

    while True:
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
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BAYER_GB2RGB)

            image_array = analyze_contours(image_array)

            cv2.imshow('Video', image_array)

            image_result.Release()

        k = cv2.waitKey(1) & 0xFF

        if k == 27:
            break

    return result


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

        # Print node data for the cam
        print_node_value(nodemap, "PixelFormat")  # Print the PixelColor format

        # Acquire images
        result &= acquire_images(cam, nodemap, nodemap_tldevice)

        # Deinitialize camera
        cam.DeInit()

    except PySpin.SpinnakerException as ex:
        print('Error: {}'.format(ex))
        result = False

    return result


def analyze_contours(img):
    # Get height, width and area of whole image.
    # img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    # img = cv2.resize(img, (1090, 1920), interpolation=cv2.INTER_AREA)

    # Get height, width and area of whole image.
    height, width = img.shape[:2]

    # initialize mask as all zeroes (black)
    mask = np.zeros((height, width), np.uint8)  # mask must be np.unint8 not the default np.float65

    # Define region of interest polygon
    # roi of polygon points
    offset = 0.20  # fractional offset from edge of  screen
    # Box ROI
    myROI = np.array([[width * offset, height * offset],
                      [(1 - offset) * width, height * offset],
                      [(1 - offset) * width, (1 - offset) * height],
                      [width * offset, (1 - offset) * height]], np.int32)

    # Define ROI polygon bounding box
    # myROI = np.array([[727, 711],
    #                   [713, 543],
    #                   [1553, 89],
    #                   [1657, 1001]], np.int32)
    cv2.fillPoly(mask, [myROI], 1)

    # Get total area of region of interest
    total_area = cv2.contourArea(myROI)

    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply the mask to the grayscale image
    gray_roi = cv2.bitwise_and(img_gray, img_gray, mask=mask)

    # Create threshold binary image
    if trackbar_mode is True:
        thresh_value = cv2.getTrackbarPos('Threshold', 'Video')
        high_thresh = cv2.getTrackbarPos('HighThreshold', 'Video')
    else:
        thresh_value = default_thresh
        high_thresh = default_high_thresh

    # retval, thresh = cv2.threshold(gray_roi, thresh_value, 255, cv2.THRESH_BINARY)
    thresh = cv2.inRange(gray_roi, thresh_value, high_thresh)

    # Remove noise through morphological transforms, opening and closing
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    ret_image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Area of ROI
    print('ROI total area: {}'.format(total_area))

    try:
        # Get the level-0 contours indices. Those with Parent hierarchy values of -1
        # hierarchy format: [NEXT, PREVIOUS, FIRST_CHILD, PARENT]
        # Value of -1 for none. (i.e. -1 for NEXT if there are no next
        # contours in that level of the hierarchy
        h0 = np.where(hierarchy[0][:, 3] == -1)

        # Create a list of contours from the h0 contour indices.
        contours_white = [contours[i] for i in h0[0]]

        # Get the level_1 contours indices. The children of the level_0 contours.
        h1 = np.where(hierarchy[0][:, 3] != -1)

        # Create a list of contours from the h1 contour indices.
        contours_black = [contours[i] for i in h1[0]]
        areas_black = [cv2.contourArea(x) for x in contours_black]

        # Create list of contour areas sorted descending
        areas = [cv2.contourArea(x) for x in contours_white]
        con_areas = list(zip(contours_white, areas))  # list of tuples of (contour, contour_area)
        con_areas = sorted(con_areas, key=lambda x: x[1], reverse=True)

        # Calc total area of white contours
        white_contour_area_total = 0
        for cont, area in con_areas:
            white_contour_area_total += area

        # Calc total area of black (holes) contours
        holes_area_total = 0
        for area in areas_black:
            holes_area_total += area

        contour_area_total = white_contour_area_total - holes_area_total

        print('Total Contour Area: {:.0f}'.format(white_contour_area_total))
        print('Number of contours total: {}'.format(len(contours_white)))

    except TypeError as err:
        print('No contours found.')
        print(err)

        # When there are no contours set the area to zero
        contour_area_total = 0

    print('{} x {}'.format(width, height))

    # Draw contours
    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

    # Draw ROI polygon
    cv2.polylines(img, [myROI], True, (0, 0, 255), 2, cv2.LINE_AA)  # must be list of polygon points

    # Resize video for display on monitor
    # [WARNING] DON'T DETERMINE AREA VALUES AFTER THIS POINT. NUMBER OF PIXELS HAS CHANGED.
    # img = cv2.resize(img, (540, 960), interpolation=cv2.INTER_AREA)

    # Rotate image 90 degrees clockwise
    # img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    # Recalc height, width for writing text
    height, width = img.shape[:2]

    # Write text to image
    dry_area_percent = contour_area_total / total_area * 100
    text = 'Dry Area: {:02.0f}%'.format(dry_area_percent)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Draw black background for text
    cv2.rectangle(img, (0, int(0.97 * height)), (width - 1, int(0.96 * height - 65)), (0, 0, 0), -1, cv2.LINE_AA)
    img = cv2.putText(img, text, (int(0.05 * width), int(0.95 * height)), font, 2, (0, 0, 255), 2, cv2.LINE_AA)

    # covert BGR to RGB colors
    # img = img[:, :, ::-1]

    return img

# Create gui window to display images/video
cv2.namedWindow('Video', cv2.WINDOW_KEEPRATIO)

# Create trackbars with inital values if set to true at beginning of file.
if trackbar_mode is True:
    cv2.createTrackbar('Threshold', 'Video', default_thresh, 255, nothing)
    cv2.createTrackbar('HighThreshold', 'Video', default_high_thresh, 255, nothing)

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
result = run_single_camera(cam)
print('Camera example complete...')

# Clean up camera object
del cam

# Clear camera list before releasing system
cam_list.Clear()

# Release system
system.ReleaseInstance()

# Clear all cv2 windows
cv2.destroyAllWindows()

# Exit the program
exit(100)
