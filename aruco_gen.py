import cv2


# Load the Aruco dictionary
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

# Generate the marker image
marker_size = 200
marker_id = 23
marker_image = cv2.aruco.drawMarker(aruco_dict, marker_id, marker_size)

# Save the marker image as a PNG file
cv2.imwrite('marker_23.png', marker_image)
