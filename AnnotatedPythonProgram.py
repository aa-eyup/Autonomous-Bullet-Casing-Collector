
'''
# Program (with the exception of modules/packages) Developed by Team 2 (Automated Bullet Casing Collector)
# Design II Fall 2017
# developped in Python 3.5.2

#The Following Information is important:
 Initial image is taken by camera ON the robot's linear actuator,
 perspective is changed to bird's eye view
 and then resized to 385 by 430 so that 12" in any direction on the picture is
 62 pixels +/- 2 pixels


# Purpose and functions of this Python Program are as follows
 at a high level (chronologically listed):
 -command the camera to take a picture
 -change the perspective of the image to bird's eye
 -resize the image to obtain inches to pixel scale
 -apply color masking to the image to identify pixel coordinates of bullets & the start_end point
 -add these coordinates to a list, remove double detections within a range, remove extra zeros from initialized list
 -cluster these coordinates based on robot's range of collection
 -create a distance matrix for the clustered coordinates (distance i to j)
 -use the distance matrix as input to the TSP/Dijkstra Algos
 -select an alo based on shortest distance traveled
 -use cluster order (order of visitation) to extract coordinates of clusters (in the same order)
 -use those coordinates to find distances and angles from i to j
 -convert pixels to inches to steps
 -prepare and send the '<direction,step>' strings for instructing the arduino
'''

#Open CV
import cv2
# arithmatic
import numpy as np
from matplotlib import pyplot as plt
from math import *
from scipy.spatial.distance import cdist as dist
from scipy.spatial.distance import pdist as pdist
# will use euclidean method cluster nodes of casing detections
from cluster import KMeansClustering
from scipy.cluster.hierarchy import *
import collections
# shortest distance algorithms 
import tsp
from tsp_custom import solve
from tsp_solver.greedy import solve_tsp
# for perspective change
import imutils
# for communicating with arduino
import serial
import time
# establish font for printing on images
font = cv2.FONT_HERSHEY_SIMPLEX

#____________________________TAKING THE IMAGE WITH USB CAMERA_____________________________________________

# TAKE AND READ THE IMAGE
# full size image

# Camera port = 0/1 for usb cam
camera_port = 0
 
#throw away 60 frames to adjust to light levels
ramp_frames = 60
 
#Initialize camera capture
camera = cv2.VideoCapture(camera_port)
#set dimensions of image taken to 1280x720
camera.set(3, 1280)
camera.set(4, 1024) 
# Captures a single image from the camera and returns it in PIL format
def get_image():
 # read is the easiest way to get a full image out of a VideoCapture object.
 retval, im = camera.read()
 return im

# make the program wait for 'ramp_frames' until get_image is activated
for i in range(ramp_frames):
 throwing_frames = get_image()
print("Taking image...")

# Capture image to keep
camera_capture = get_image()
# create a path for the image which will be kept for analysis
my_capture = 'C:\\path\\my_capture.png'
# write the image to the path created above
cv2.imwrite(my_capture, camera_capture)


 
# delete the camera so that script does not have to be exited to take a new picture (if need be)
del(camera)

# read the image from the path which was written to path after camera capture
image = cv2.imread('C:\\path\\my_capture.png')

#_______________________INITIALIZATION_________________________________

# create a list of 0s which will be replaced by x,y pixel coordinates of casing contour detections
xs = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
ys = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

# create a list of 0s which will be replaced by x,y pixel coordinates of start/end point
a_s = [0,0,0,0,0,0,0,0,0,0]
b_s = [0,0,0,0,0,0,0,0,0,0]

# color masking parameters for bullet casings
# the 3 numbers are [B, G, R] values 
#upper_bound = np.array([50, 50, 150])
#lower_bound = np.array([20, 15, 80])
#upper_bound = np.array([100, 80, 85])
#lower_bound = np.array([5, 5, 5])
# color mask below was selected to room which demonstrations of project was shown
upper_bound = np.array([85, 85, 85])
lower_bound = np.array([5, 5, 5])

tape_upper_bound = np.array([110, 160, 220])
tape_lower_bound = np.array([40, 120, 170])

# kernels used to enhance true detections & deminish false detections
kernel_erode = np.ones((1,1), np.uint8)
kernel_dilate = np.ones((1,1), np.uint8)
kernel_open = np.ones((1, 1), np.uint8)
kernel_close = np.ones((1,1), np.uint8)


#________________________PERSPECTIVE CHANGE TO BIRD'S EYE VIEW______________________________________

# resize reads integers, not floats. place the size in an integer argument
# define heigh of image which will be used for perspective change
imsize = int(600)
# looking for shapes with 4 corners
no_corners = 4
# ratio of (original image heigh divided by new height) saved for later use
ratio = image.shape[0]/(imsize)
orig = image.copy()
# resize the original image
image = imutils.resize(image, height = imsize)

#cv2.imshow('image', image)

# convert the image to grayscale to allow edge detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# use parameters to blur the image slightly to
# remove noise which might trigger as an "edge"
#bullets detected well at 11,20,20 (fyi)
gray = cv2.bilateralFilter(gray, 25, 25, 55)
edged_image = cv2.Canny(gray, 200, 120)


#cv2.imshow('edged', edged_image)

# find the contours of the image (where pixel value is = 1 which means considered as an edge (white)
cnts = cv2.findContours(edged_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
# sort the contours list from greatest area
# to least area and then retreive the first ten (0 through 9)
# the enclosed square area consumes a large area portion of the image
# which is why we want the largest values
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
# initialize the contour of the tape to none
turfCnt = None
# identify the contours which belong to the turf
for i in cnts:
    p = cv2.arcLength(i, True)
    # approximate the number of corners of each contour using 5% of the contours perimeter
    # because the bottom corners of the turf
    approximation = cv2.approxPolyDP(i, 0.05 * p, True)
    if len(approximation) == no_corners:
        turfCnt = approximation
        break
# convert contour list into a numpy array so the drawContours function can iterate through
turfCtr = np.array(turfCnt).reshape((-1,1,2)).astype(np.int32)
cv2.drawContours(image, [turfCtr], -1, (0, 255, 0), 2)
cv2.imshow('turf', image)


# Need to find the 4 corners of the turf (tl, tr, br, bl)
turf_pts = turfCtr.reshape(4,2)
# initialize a 4 (4 corners) by 2 (x and y) array of zeros, to be replaced later
turf_rect = np.zeros((4,2), dtype ='float32')

# top left: small x value, small y value
# top right: large x value, small y value
# bottom right: large x value, large y value
# bottom left: small x value, large y value

# sum the points vertically using axis = 1
pts_sum = turf_pts.sum(axis = 1)
pts_diff = np.diff(turf_pts, axis = 1)

# replace the zeros in the turf_rect array
# top left
turf_rect[0] = turf_pts[np.argmin(pts_sum)]
# top right
turf_rect[1] = turf_pts[np.argmin(pts_diff)]
# bottom right
turf_rect[2] = turf_pts[np.argmax(pts_sum)]
# bottom left
turf_rect[3] = turf_pts[np.argmax(pts_diff)]

# size the perspective shape to stay proportional with the original image
fin_turf_rect = turf_rect * ratio

# perform the transformation with the 4 corners
(tl, tr, br, bl) = fin_turf_rect
# pythag theroem the width of the birds eye image to horizontally level the top & bottom edges of the image
# sqrt(delta(x)^2 + delta(y)^2) of bottom right and bottom left corners
width1 = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
# sqrt(delta(x)^2 + delta(y)^2) of top right and top left corners
width2 = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

# pythag theroem the height of the birds eye image to vertically level the left & right edges of the image
# sqrt(delta(x)^2 + delta(y)^2) of top right and bottom right corners
height1 = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
# sqrt(delta(x)^2 + delta(y)^2) of top left and bottom left corners
height2 = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
 
# final dimensions of the turf in birds eye
maxWidth = max(int(width1), int(width2))
maxHeight = max(int(height1), int(height2))

# map the birds eye image
be_dist = np.array([[0,0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype = 'float32')

#transform matrix of new perspective
p = cv2.getPerspectiveTransform(fin_turf_rect, be_dist)
be_img = cv2.warpPerspective(orig, p, (maxWidth, maxHeight))

#cv2.imshow('turf', image)
#cv2.imshow('turf BE', be_img)

#resize the final perspective image to 385x430 pixels so that 12" = 61 pixels in any direction
be_img_sq = cv2.resize(be_img, (385, 430))
fin_img = cv2.resize(be_img, (385, 430))
fin_img_clust = cv2.resize(be_img, (385, 430))

# write the perspective image if visual is needed
cv2.imwrite('C:\\path\\perspective.png', be_img_sq)

#___________________________BULLET CASING/START_END POINT DETECTION___________________________________

while True:

    # convert the image to HSV colors for color masking
    HSV_img = cv2.cvtColor(be_img_sq, cv2.COLOR_BGR2HSV)

    # in the HSV image, look for color B, G, R values in the interval defined above
    mask_tape = cv2.inRange(HSV_img, tape_lower_bound, tape_upper_bound)
    open_tape = cv2.dilate(mask_tape, kernel_open, iterations = 1)
    #cv2.imshow('tape mask', mask_tape)
    # define best mask to use
    mask_tape = mask_tape

    mask = cv2.inRange(HSV_img, lower_bound, upper_bound)
    dilate = cv2.erode(mask, kernel_dilate, iterations = 1)
    mask_open = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, kernel_open)
    mask_close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    #erosion = cv2.erode(mask_open, kernel_open, iterations = 1)
    #ero_dil = cv2.dilate(erosion, kernel_close, iterations = 1)
    # define best mask to use
    mask_final = mask
    #show the mask to see where detections occur (will show as white)
    #cv2.imshow('mask', mask_final)

    # contour of the bullet detections
    con = cv2.findContours(mask_final.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]

    #contour of the initial/final point detection
    con_tape = cv2.findContours(mask_tape.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
    print(con_tape)
    # for loop below finds length of the number of contours (number of detections)- range starts at 0


    # save the x,y values of the contours 
    tape_count = 0
    for i in range(len(con_tape)):
        tape_count += 1
        if tape_count < len(con_tape) + 1:
            a, b, c, d = cv2.boundingRect(con_tape[i])
#            cv2.rectangle(fin_img, (a,b), (a + c, b + d), (255,255,0), 2)
            a_s[i] = a
            b_s[i] = b
            
    cas_count = 0
    for i in range(len(con)):
        cas_count += 1
        if cas_count < len(con) + 1:
            x, y, w, h = cv2.boundingRect(con[i])
            #cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
            xs[i] = x
            ys[i] = y
    
    break

# create new arrays of x/y coordinates with removed zeros and double detections (w/i 30 pixels)
x_zad = []
y_zad = []
# these for loops save the index of the conditions being asked
for i in list(range(0, len(xs))):
    if xs[i] == 0 and ys[i] == 0:
        x_zad.append(i)
        y_zad.append(i)
for i in list(range(0, len(xs) - 1)):
    # remove double detection of same bullet if detections happen with in 15 pixels of e/o
    if abs(xs[i] - xs[i+1]) < 30 and abs(ys[i] - ys[i+1]) < 30:
        x_zad.append(i)
        y_zad.append(i)

a_zad = []
b_zad = []
for i in list(range(0, len(a_s))):
    if a_s[i] == 0 and b_s[i] == 0:
        a_zad.append(i)
        b_zad.append(i)
for i in list(range(0, len(a_s))):
    if abs(a_s[i] - a_s[i-1]) < 18 and abs(b_s[i] - b_s[i-1]) < 18:
        a_zad.append(i)
        b_zad.append(i)

# from the x and y lists, delete the index locations saved above
x_loc = np.delete(xs, x_zad)
y_loc = np.delete(ys, y_zad)

a_loc = np.delete(a_s, a_zad)
b_loc = np.delete(b_s, b_zad)

# find index of y coordinate detection which are below the initial/final point (blue post-it)
pts_below_inifin = []
for i in list(range(0,len(x_loc))):
    if y_loc[i] > np.mean(b_loc):
        pts_below_inifin.append(i)
# use indeces of points below the initial/final to delete from the x & y loc arrays
x_loc = np.delete(x_loc, pts_below_inifin)
y_loc = np.delete(y_loc, pts_below_inifin)


#Initial and final point. Select 1 point from the a_loc and b_loc list
#ini_fin_point_x, ini_fin_point_y = (a_loc[0], b_loc[0])
#ini_fin_point = [ini_fin_point_x, ini_fin_point_y]
# use a fixed coordinate if the detection mask is not working for the start/end point
ini_fin_point_x, ini_fin_point_y = (192, 395)
ini_fin_point = [ini_fin_point_x, ini_fin_point_y]


# save the coordinates (x and y arrays) together and list them verticaly with '.T'
coords = np.vstack([x_loc, y_loc]).T

# create a distance matrix using the coordinates of the bullet casing detections
dist_df = dist(coords, coords, metric = 'euclidean')

# use statement below if scatter plotting x and y coordinates
#coords_plt = np.vstack([x_loc, -y_loc+600]).T

# _______________________________CLUSTERING FUNCTION_______________________________________________

# call the coordinates which will be clustered and save as Z
Z = linkage(coords, 'ward')
# DISTANCE BETWEEB CASINGS IN ORDER TO GET CLUSTERED (SUBJECT TO CHANGE)
# cluster bullets with in 40 pixels of eachother (equivalent to about 4" (half the width of the collection roller)
max_d = 25
clusters = fcluster(Z, max_d, criterion = 'distance')
#print(clusters)

# create a distance matrix for the clusters

coords_matched = []
for i in list(range(0, len(clusters))):
    coords_matched.append(list(np.where(clusters == clusters[i])[0]))
# remove duplicate lists with in the list
unique_coords = [list(x) for x in set(tuple(x) for x in coords_matched)]
#save number of indeces in each cluster
len_clusters = []
for i in unique_coords:
    len_clusters.append(len(i))


# create a dictionary with lists of index values (to retrieve coordinates) of each cluster
dict_keys = list(range(1, len(unique_coords)+1))
# initialize the keys for the dictionary (equivalent to the number of unique cluster coordinates
dict_indices = {}
for j in dict_keys:
    dict_indices['%s' % j] = [unique_coords[j-1]]

dict_coords = collections.defaultdict(list)
for i in dict_keys:
    # access the element related to each key in the dictionary
    for j in dict_indices[str(i)]:
        # append the coordinates being clustered together to the element under each key of the dictionary
        for k in list(range(0, len(j))):
            #appends the coordinates from coords into the a list in the dictionary
            dict_coords['%s' % i].append(coords[j[k]])

# find centroid of the coordinates clustered together (the singular coordinate of each cluster)
# add the coordinate of each cluster to a dictionary with # keys = to number of clusters
my_centroids = {}
for i in list(range(1, len(dict_coords)+1)):
    v_stack = np.vstack(tuple(dict_coords[str(i)]))
    centroid = np.mean(v_stack, axis = 0)
    my_centroids['%s' % i] = [centroid]



# append the centroid coordinates in the dictionary to an array (like coords)
l_coords = []
# order of the items in the array matter -> first append the initial/final point so that it is i & j = 0
# in the distance matrix about to be made below
l_coords.append(ini_fin_point)

for i in list(range(1, len(dict_coords)+1)):
    # keys of the dictionary are strings -> index with strings of i
    l_coords.append(my_centroids[str(i)])
centroid_coords = np.vstack(tuple(l_coords))
#print(centroid_coords)

# Create a distance matrix using the cluster coordinates and the initial/final point
centroid_df = dist(centroid_coords, centroid_coords, metric = 'euclidean')
#print(centroid_df)


# PRINT IMAGE WITH & W/O CLUSTER LABELLING
# image with each detection show
for i in list(range(0, len(x_loc))):
    cv2.rectangle(fin_img, (x_loc[i], y_loc[i]), (x_loc[i]+ 15, y_loc[i]+15), (0,0,255), 1)
    cv2.rectangle(fin_img, (ini_fin_point_x, ini_fin_point_y), (ini_fin_point_x+15, ini_fin_point_y-15), (255, 255, 255), 3)
    #cv2.putText(fin_img, str(i), (x_loc[i] - 25, y_loc[i] + 20), font, 1, (0,255,255), 1, cv2.LINE_AA)
# image with clustered detections
for j in list(range(1, len(my_centroids)+1)):
    cv2.putText(fin_img_clust, str(j), (int(my_centroids[str(j)][0][0]), int(my_centroids[str(j)][0][1])), font, 1, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.rectangle(fin_img_clust, (ini_fin_point_x, ini_fin_point_y), (ini_fin_point_x+15, ini_fin_point_y-15), (255, 255, 255), 3)
    cv2.rectangle(fin_img_clust, (int(my_centroids[str(j)][0][0]), int(my_centroids[str(j)][0][1])), (int(my_centroids[str(j)][0][0])+5, int(my_centroids[str(j)][0][1])+5), (255,255,255), 4)
#cv2.imshow('clust_img', fin_img_clust)
#cv2.imshow('fin_img', fin_img)
#cv2.waitKey(10)


# _________________________USING CLUSTER COORDINATES AND DISTANCE MATRIX FOR OPTIMAL PATH ALGORITHMS____________________
# the centroid distance matrix will be passed through the algo functions because all the nodes in the cluster will be visitied

# ****greedy TSP****
path_greedy = solve_tsp(centroid_df)

# move the initial/final point to the start and end of the path
path_greedy.insert(0, path_greedy.pop(path_greedy.index(0)))
path_greedy.append(0)
path_greedy_final = path_greedy
#print('greedy path cluster point')
#print(path_greedy_final)
coord_path_greedy = []
for i in path_greedy_final:
    coord_path_greedy.append(centroid_coords[i])
#print('greedy path coordinates')
#print(coord_path_greedy)

# ****custom TSP algo****
(path_custom, length_custom) = solve(centroid_df)

# move the initial/final point to the start and end of the path
path_custom.insert(0, path_custom.pop(path_custom.index(0)))
path_custom.append(0)
path_custom_final = path_custom
#print('custom path cluster points')
#print(path_custom_final)
coord_path_custom = []
for i in path_custom_final:
    coord_path_custom.append(centroid_coords[i])
#print('custom path coordinates')
#print(coord_path_custom)

#________________________COMPARING THE TOTAL DISTANCE TRAVELED OF EACH OPTIMAL PATH ALGORITHM_________
# save and sum the distance traveled from node i to node j (j = i+1) in order of the optimal path
# distances obtained from the centroid distances matrix

# ***custom algo distances***
#print('custom algo distances')
custom_distance_n2n = []
for i in list(range(0,len(centroid_df))):
    custom_distance_n2n.append(centroid_df[path_custom_final[i]][path_custom_final[i+1]])
#print(custom_distance_n2n)

custom_sum_distances = 0
for i in custom_distance_n2n:
    custom_sum_distances += i

# ****greedy algo distances****
print('greedy algo distances')
greedy_distance_n2n = []
for i in list(range(0,len(centroid_df))):
    greedy_distance_n2n.append(centroid_df[path_greedy_final[i]][path_greedy_final[i+1]])
print(greedy_distance_n2n)

greedy_sum_distances = 0
for i in greedy_distance_n2n:
    greedy_sum_distances += i

#print('------------------------Total Distance Traveled----------------------')
#print('custom algo')
#print(round(custom_sum_distances,2))
#print('greedy algo')
#print(round(greedy_sum_distances, 2))

#---------------------------------------------------------------------------------

# pick a path for the robot to follow based on which yielded the shortest path
# save all accompanying values as the "final" set

if custom_sum_distances < greedy_sum_distances:
    my_finalpath_dist = custom_sum_distances
    my_finalcoords = coord_path_custom
    my_dist_n2n = custom_distance_n2n
    my_finalpath = path_custom_final
else:
    my_finalpath_dist = greedy_sum_distances
    my_finalcoords = coord_path_greedy
    my_dist_n2n = greedy_distance_n2n
    my_finalpath = path_greedy_final
    
my_finaldist = my_dist_n2n

#print(my_finalcoords)
#print('FINAL PATH ORDER---------------------------------------------')
#print(my_finalpath)

# Calculating the turn angles from node to node ( + = right, - = left)
# these turn angles are relevant to the current node
# (think of it as the robot having the unit circle on it where (90 degress always points up)

# current theta = to -90 because y value increase as go from top to bottom of image
# essentially, -90 means looking UP 
current_heading = -90
# initialize a list of turn angles which will be listed in order corresponding to the path
angles_n2n = []
for i in list(range(0, len(my_finalcoords)-1)):
    dest_angle = (180/3.1415)*(atan2(((my_finalcoords[i+1][1] - my_finalcoords[i][1])), (my_finalcoords[i+1][0] - my_finalcoords[i][0])))
    turn_angle = dest_angle - current_heading
    angles_n2n.append(turn_angle)
    current_heading = dest_angle
my_finalangles = angles_n2n


# altering angles so that minimal rotation is used
for i in list(range(0, len(my_finalangles))):
    # if the turn angle is greater than 180, turn angle is too far right
    # change turn angle to negative (left turns) but greater than -180
    if my_finalangles[i]>180:
        # subtract 374 instead of 360 due to non-linear step-to-turning relationship
        my_finalangles[i] = my_finalangles[i]-374
        
    # if the turn angle is  less than -180, turn angle is too far left
    # change angle to positive (right turn) but less than +180
    elif my_finalangles[i]<-180:
        # add 374 instead of 360 due to non-linear step-to-turning relationship
        my_finalangles[i] = my_finalangles[i]+374

# account for left jolt of the motors at the end of each 'Forward' direction
# and right jolt and the end of each 'Right' direction
for i in list(range(0, len(my_finalangles))):
    # if turning 
    if my_finalangles[i] > 0:
        # the value subtracted is subject to change due to inconsistency of motors
        my_finalangles[i] -= 5
    if my_finalangles[i] < 0:
        # the value added is subject to change due to inconsistency of motor
        my_finalangles[i] += 4

#print('my_finaldist')
#print(my_finaldist)
#print('my_finalangles')
#print(my_finalangles)

# __________________________________________PREP DATA FOR ARDUINO ____________________________
list1 = [item for sublist in zip(my_finalangles,my_finaldist) for item in sublist]
#print(list1)

# set factor constants of # of steps to move to turn a certain number of degrees
#interval of 0-50
steps_in_1_degree_50 = 380/90
# interval of 50-90
steps_in_1_degree_50_90 = 465/90
#steps_in_1_degree_180 = 865/180
#interval for left and right turn between 90-180
steps_in_1_degree_180_Left = 745/180
steps_in_1_degree_180_Right = 760/180
#steps_in_1_inch = 340/12
# factor for 'Forward' direction
steps_in_1_inch = 320/12

# initialize the direction list and strings
converted_list = []
forward = "<Forward,"
left = "<Left,"
right = "<Right,"


# segment the turning angles in order to multiple by a factor to account
# for non-linear step-to-angleturned relationship

for i in range(0,len(list1)):
    #if the index of the step value is even (or 0), this value represents a turn -> multiply with corresponding factor
    if i % 2 == 0:
        if abs(list1[i]) < 50:
            converted_value = round(list1[i] * steps_in_1_degree_50)
            converted_list.append(int(converted_value))
        elif abs(list1[i]) >= 50 and abs(list1[i]) <= 90:
            converted_value = round(list1[i] * steps_in_1_degree_50_90)
            converted_list.append(int(converted_value))
        elif list1[i] >= -180 and list1[i] < -90: #Left turn
            converted_value = round(list1[i] * steps_in_1_degree_180_Left)
            converted_list.append(int(converted_value))
        elif list1[i] <= 180 and list1[i] > 90: #Right turn
            converted_value = round(list1[i] * steps_in_1_degree_180_Right)
            converted_list.append(int(converted_value))
         
    # if the index of the step value is not even -> these are step values to go straight
    elif i % 2 != 0:
        # divide the value by 61.5 because there are about 61.5 pixels per foot in a 385 by 430 pixel img
        converted_value = round(((list1[i] / 61.5) *12) * steps_in_1_inch)
        converted_list.append(int(converted_value))


# finish the list of lists consisting of directions to be passed to the arduino
for i in range(0,len(converted_list)):
    # if index number of the step value is an even number (or 0), add the step value to the list under a turn direction
    if i % 2 == 0:
        if converted_list[i] < 0:
            converted_list[i] = left + str(abs(converted_list[i])) + ">"
        elif converted_list[i] > 0:
            converted_list[i] = right + str(abs(converted_list[i])) + ">"
        # if turning angle is 0, move forward 0 steps (to prevent jolts after turns)
        else:
            converted_list[i] = forward + str(abs(converted_list[i])) + ">"
    # if index number of the step value is an odd number, add the step value to the list under a 'Forward' direction
    else:
        converted_list[i] = forward + str(converted_list[i]) + ">"

final_converted_list.extend(converted_list)
#print(final_converted_list)


# _____________________________SENDING DATA TO THE ARDUINO___________________________________________


def sendToArduino(sendStr):
  ser.write(sendStr.encode())

def recvFromArduino():
  global startMarker, endMarker
  
  ck = ""
  x = "z"  # any value that is not an end- or startMarker
  byteCount = -1 # to allow for the fact that the last increment will be one too many
  
  # wait for the start character
  while  ord(x) != startMarker: 
    x = ser.read()
  
  # save data until the end marker is found
  while ord(x) != endMarker:
    if ord(x) != startMarker:
        print(x)
        ck = ck + x.decode()
        byteCount += 1
    x = ser.read()
  
  return(ck)


def waitForArduino():

   # wait until the Arduino sends 'Arduino Ready' - allows time for Arduino reset
   # it also ensures that any bytes left over from a previous message are discarded
   
    global startMarker, endMarker
    
    msg = ""
    while msg.find("Arduino is ready") == -1:

      while ser.inWaiting() == 0:
        pass        
      msg = recvFromArduino()
      print(msg)


# define the function created to send data to the arduino 1 character at a time
def runTest(td):
  numLoops = len(td)
  waitingForReply = False

  n = 0
  while n < numLoops:

    teststr = td[n]

    if waitingForReply == False:
      sendToArduino(teststr)
      print("Sent from PC -- LOOP NUM " + str(n) + " TEST STR " + teststr)
      waitingForReply = True

    if waitingForReply == True:

      while ser.inWaiting() == 0:
        pass
        
      dataRecvd = recvFromArduino()
      print("Reply Received  " + dataRecvd)
      n += 1
      waitingForReply = False
        # mark the end of the reception of an instruction
      print("===========")

    time.sleep(5)



# NOTE the user must ensure that the serial port and baudrate are correct
# check port name of arduino by going to arduino IDE -> tools -> port
# change X to the value shown under the port in arduino IDE
serPort = "COMX"
baudRate = 9600
ser = serial.Serial(serPort, baudRate)
print ("Serial port " + serPort + " opened  Baudrate " + str(baudRate))


startMarker = 60 # represent the '<' symbol
endMarker = 62 # represents the '>' symbol

waitForArduino()

# used testData to check if singular directions worked properly
#testData = []
#testData.append("<Forward,640>")

# send the final_converted_listed created above to the arduino by passing the list through the runTest function
runTest(final_converted_list)

# close the serial port, if not, program will not execute in arduino
ser.close()



