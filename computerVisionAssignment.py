# TO RUN CODE:
# carMask.png and master_path_to_dataset file must be in same file as the python code
# First go on the command line, I have tested with python3 using linux after running  "opencv3-1.init" and this seems to run my code
# Must have the TBB-durham-02-10-17-sub10 folder inside the same folder as the python script
# so run: python3 ComputerVisionAssignment.py

import os
import numpy as np
import cv2
import random
import math
import time


master_path_to_dataset = "./TTBB-durham-02-10-17-sub10"
directory_to_cycle_left = "left-images"
directory_to_cycle_right = "right-images"

windowName = "first Image"
windowName2 = "second Image"
windowName3 = "edges"

full_path_directory_left =  os.path.join(master_path_to_dataset, directory_to_cycle_left);
full_path_directory_right =  os.path.join(master_path_to_dataset, directory_to_cycle_right);

full_path_filename_left = os.path.join(full_path_directory_left, "1506942480.483420_L.png");
full_path_filename_right = (full_path_filename_left.replace("left", "right")).replace("_L", "_R");

left_file_list = sorted(os.listdir(full_path_directory_left));

camera_focal_length_px = 399.9745178222656
camera_focal_length_m = 4.8 / 1000
stereo_camera_baseline_m = 0.2090607502

image_centre_h = 262.0;
image_centre_w = 474.5;

#####################################################################

def project_disparity_to_3d(disparity, max_disparity):

    points = [];

    f = camera_focal_length_px;
    B = stereo_camera_baseline_m;

    height, width = disparity.shape[:2];

    for y in range(height):
        for x in range(width):

            if (disparity[y,x] > 0):

                Z = (f * B) / disparity[y,x];
                X = (x * Z) / f;
                Y = (y * Z) / f;

                points.append([X,Y,Z]);

    return points;

#####################################################################

def project_3D_points_to_2D_image_points(points):

    points2 = [];

    for i1 in range(len(points)):
        x = (points[i1][0] * camera_focal_length_px) / points[i1][2];
        y = (points[i1][1] * camera_focal_length_px) / points[i1][2];
        points2.append([x,y]);

    return points2;

# Pre-Image filtering - GAMMA TRANSFORM
def gammaTransform8(img, gamma):
    g = 1.0 / gamma
    img = (np.power((img[:,:] / 255), g) * 255)
    return img

# Pre-Image filtering - LOGARITHM TRANSFORM
def logarithmTransform8(img, sigma):
    c = 255 / math.log(1 + 255)
    img = c * np.log(1 + (math.exp(sigma - 1))*img[:,:])
    return img

# Main RANSAC Algorithm
def ransac(imgL, imgR, V, trials):
    imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR);
    imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR);

    # Car Mask that filters away the front of the car
    carMask = cv2.imread("./carMask.png", cv2.IMREAD_COLOR)

    images = [imgL, imgR]
    imageL = images[0]

    # ------- HSV COLOUR SPACE ----
    # Pre-Image filtering - Thresholds the value and saturation level of the images
    hsvL = cv2.cvtColor(imageL, cv2.COLOR_BGR2HSV);
    hL, sL, vL = cv2.split(hsvL)
    vL[(vL >= 200)] = 170
    sL[(sL >= 200)] = 170
    images[0] = cv2.merge([hL,sL,vL])
    images[0] = cv2.cvtColor(images[0], cv2.COLOR_HSV2BGR)
    images[0][(hL >= 30) & (hL <= 65) & (sL >= 30)] = [0, 0, 0]
    # cv2.imshow("HSV", images[0])
    # cv2.waitKey(0)


    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # grayL = clahe.apply(grayL)
    # grayR = clahe.apply(grayR)

    grayL = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY);
    grayR = cv2.cvtColor(images[1], cv2.COLOR_BGR2GRAY);

    # --- HARRIS DETECTION -----
    # harr = cv2.cornerHarris(grayL,3,5,0.04)
    # harr = cv2.dilate(harr, None)
    # images[0][harr>0.005*harr.max()]=[0,0,255]
    # cv2.imshow("Harris", images[0])
    # cv2.waitKey(0);


    # --- FAST DETECTION ------
    # fast = cv2.FastFeatureDetector_create(cv2.FAST_FEATURE_DETECTOR_THRESHOLD)
    # fastImagePoints = fast.detect(images[0], None)
    # cv2.drawKeypoints(images[0], fastImagePoints, color=(255,0,0), outImage = images[0])
    # cv2.imshow("Fast", images[0])
    # cv2.waitKey(0)


    # --- Canny Detection ----
    canny = cv2.Canny(images[0], 1, 200)
    canny2 = canny


    # --- ORB DETECTION ------
    orb = cv2.ORB_create(nfeatures=100000, patchSize=2)
    orbImagePoints = orb.detect(images[0], None)
    orbImagePoints, _ = orb.compute(images[0], orbImagePoints)
    canny = cv2.drawKeypoints(canny, orbImagePoints,None, color = (255, 255,255))
    canny[::3, ::3, :] = 255
    canny[:280,:,:] = 0
    # cv2.imshow("ORB", canny)
    # cv2.waitKey(0)


    # Pre-Image Filtering - Histogram Equalisation
    grayL = cv2.equalizeHist(grayL)
    grayR = cv2.equalizeHist(grayR)

    # grayL = cv2.adaptiveThreshold(grayL, 127, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    # grayR = cv2.adaptiveThreshold(grayR, 127, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    max_disparity = 128;
    stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21);


    # ---- LIGHTING INVARIANCE ----
    #
    # alpha = 0.4
    # images[0] = np.array(images[0])
    # images[1] = np.array(images[1])
    #
    # ii_imageL = 0.5 + np.log10(images[0][:, :, 1]) - alpha * np.log10(images[0][:, :, 0]) - (1 - alpha) * np.log10(images[0][:, :, 2]);
    # ii_imageL = np.array(ii_imageL * 255, dtype=np.uint8)
    # ii_imageR = 0.5 + np.log10(images[1][:, :, 1]) - alpha * np.log10(images[1][:, :, 0]) - (1 - alpha) * np.log10(images[1][:, :, 2]);
    # ii_imageR = np.array(ii_imageR * 255, dtype=np.uint8)
    # cv2.imshow("lightInvL", ii_imageL)
    # cv2.imshow("lightInvR", ii_imageR)
    # cv2.waitKey(0)
    # ---------------------------------


    # ---- Logarithmic Transform ------------
    #
    # logTransformL = logarithmTransform8(grayL, 3.0)
    # logTransformR = logarithmTransform8(grayR, 3.0)
    #
    # grayL = logTransformL.astype('uint8')
    # grayR = logTransformR.astype('uint8')
    #
    # cv2.imshow("log", grayL)
    # cv2.imshow("log2", grayR)

    # ----------------------------------------

    # --- Gamma Transform ---
    gammaTransformL = gammaTransform8(grayL, 17.5)
    gammaTransformR = gammaTransform8(grayR, 17.5)
    grayL = gammaTransformL.astype('uint8')
    grayR  = gammaTransformR.astype('uint8')
    # cv2.imshow("gamma", grayL)
    # cv2.imshow("gamma2", grayR)


    disparity = stereoProcessor.compute(grayL, grayR);

    dispNoiseFilter = 3;  # increase for more agressive filtering
    cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter);

    _, disparity = cv2.threshold(disparity, 0, max_disparity * 16, cv2.THRESH_TOZERO);
    disparity_scaled = (disparity / 16.).astype(np.uint8);

    # --- Filtering Disparity - REDUCE FEATURE POINT SPACE
    canny = canny[:, :, 0]
    disparity_scaled[canny == 0] = 0

    points_filtered = project_disparity_to_3d(disparity_scaled, max_disparity);
    points_filtered = np.array(points_filtered)
    points = np.array(points_filtered)

    count = 0
    for i in range(0, int(trials)):
        count += 1
        cross_product_check = np.array([0, 0, 0]);
        while cross_product_check[0] == 0 and cross_product_check[1] == 0 and cross_product_check[2] == 0:
            [P1, P2, P3] = points_filtered[random.sample(range(len(points_filtered)), 3)];
            # make sure they are non-collinear
            cross_product_check = np.cross(np.subtract(P1, P2), np.subtract(P2, P3));

        #estimate feature parameters using model
        coefficients_abc = np.dot(np.linalg.inv(np.array([P1, P2, P3])), np.ones([3, 1]))
        coefficient_d = math.sqrt(
            coefficients_abc[0] * coefficients_abc[0] + coefficients_abc[1] * coefficients_abc[1] + coefficients_abc[2] * coefficients_abc[2])

        # Calculate array of distance from plane of feature points
        nearbyPoints = 0
        planePoints = []
        planePointsLeft = []
        planePointsRight = []
        dist = abs((np.dot(points, coefficients_abc) - 1) / coefficient_d)

        # Calculate the number of feature points close enough to the plane
        nearbyPoints = (dist <= 0.005).sum()
        boolMask = np.array([dist<=0.005]).flatten()
        planePoints = points[boolMask]

        planeAngle = (np.degrees(np.arccos(coefficients_abc / coefficient_d)))

        #Check that data supports the feature
        #The check involves the enough points close to the plane and that the plane angle is not too verticle
        if nearbyPoints > V and planeAngle[0] <= 120 and planeAngle[0] >= 60 and planeAngle[2] <= 120 and planeAngle[2] >= 60 :

            planePointsTemp = [planePoints[x: x + 100] for x in range(0, len(planePoints), 100)]
            planePoints = []
            for x in planePointsTemp:
                max = [0, 0, 0]
                for y in x:
                    if y[0] > max[0]:
                        max = y
                planePointsLeft.append(max)

            for x in range(len(planePointsTemp) - 1, 0, -1):
                mini = [1000000000, 0, 0]
                for y in planePointsTemp[x]:
                    if y[0] < mini[0]:
                        mini = y
                planePointsRight.append(mini)

            planePointsTemp = [planePointsLeft[x: x + 3] for x in range(0, len(planePointsLeft), 3)]
            for x in planePointsTemp:
                max = [0, 0, 0]
                for y in x:
                    if y[0] > max[0]:
                        max = y
                planePoints.append(max)

            planePointsTemp = [planePointsRight[x: x + 3] for x in range(0, len(planePointsRight), 3)]
            normal = [0, 0, 0]
            for x in planePointsTemp:
                min = [1000000, 0, 0]
                for y in x:
                    if y[0] < min[0]:
                        min = y
                        normal = y
                planePoints.append(min)

            pts = project_3D_points_to_2D_image_points(planePoints);
            pts = np.array(pts, np.int32);
            pts = pts.reshape((-1,1,2));


            # Calculating the Normal of the plane
            f = camera_focal_length_px;
            B = stereo_camera_baseline_m;
            Z = normal[2]
            Zmax = ((f * B) / 2)

            xOrigin = 500
            yOrigin = 267
            XOrigin = ((500-image_centre_w) * Zmax) / f
            YOrigin = ((267-image_centre_h) * Zmax) / f

            xValue = ((((-(coefficients_abc[0] / coefficient_d)*10+XOrigin) * f) / Zmax) + image_centre_w)
            yValue = ((((-(coefficients_abc[1] / coefficient_d)*10 + YOrigin) * f) / Zmax) + image_centre_h)


            # points as circles
            #################
            # pts = pts.ravel()
            # for x in range(0, len(pts), 2):
            # 	print(pts[x])
            # 	cv2.circle(images[0],(pts[x],pts[x+1]),3,(0,255,0),1)
            #################

            # Draw plane on Coloured Image
            cv2.polylines(imageL,[pts],True,(0,0,255), 3);

            # ---- OBJECT DETECTION - TEST -----
            # Car mask removes front of car as object
            # Implementation detects feature points which has Y value above or below the plane by a margin
            mask = np.zeros((544, 1024, 3), dtype=np.uint16)
            mask2 = np.zeros((544, 1024, 3), dtype=np.uint16)
            cv2.fillPoly(mask, [pts], (255, 255, 255))
            disparity_scaled[(mask[:,:,0] == 0)] = 0
            points_filtered = points_filtered[(points_filtered[:,1] > 1.7 * ((coefficient_d - (points_filtered[:,0]*coefficients_abc[0]) - (points_filtered[:,2]*coefficients_abc[2]))/coefficients_abc[1])) | (points_filtered[:,1] < 0.1 * ((coefficient_d - (points_filtered[:,0]*coefficients_abc[0]) - (points_filtered[:,2]*coefficients_abc[2]))/coefficients_abc[1]))]
            ptsFiltered = project_3D_points_to_2D_image_points(points_filtered);
            ptsFiltered = np.array(ptsFiltered, np.int32);
            ptsFiltered = ptsFiltered.reshape((-1, 1, 2));
            for x in ptsFiltered:
                cv2.circle(mask2, (x[0][0], x[0][1]), color=(255,255,255), radius=1)
            mask2[mask[:,:,0] != 255] = 0
            mask2[carMask[:,:,0] != 255] = 0
            kernel = np.ones((5, 5), np.uint8)
            mask2 = cv2.dilate(mask2, kernel, iterations=1)
            mask2 = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)
            mask2 = np.array(mask2, dtype=np.uint8)
            mask2[mask2 != 0] = 255
            (_, contours, _) = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mask2, contours, -1, (0, 255, 0), 20)
            mask2 = cv2.cvtColor(mask2, cv2.COLOR_GRAY2BGR)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(mask2, (x, y), (x + w, y + h), (255, 0, 0), thickness=3)
            imageL[(mask2[:, :] != 0)] = 255
            cv2.arrowedLine(imageL, (int(xOrigin), int(yOrigin)), (int(xValue), int(yValue)), color = (0, 255,0), thickness = 2, tipLength=0.3)
            print(full_path_filename_left)
            print(full_path_filename_right + " : road surface normal ({}, {}, {})".format(coefficients_abc[0][0], coefficients_abc[1][0], coefficients_abc[2][0]))
            cv2.imshow("RESULT", imageL)
            key = cv2.waitKey(40 * (not(pause_playback))) & 0xFF;
            return True
        else:
            cv2.destroyAllWindows()
    if count == int(trials):
        print(full_path_filename_left)
        print(full_path_filename_right + " : road surface normal (0, 0, 0)")


for filename_left in left_file_list:

    filename_right = filename_left.replace("_L", "_R");
    full_path_filename_left = os.path.join(full_path_directory_left, filename_left);
    full_path_filename_right = os.path.join(full_path_directory_right, filename_right);

    ransac(full_path_filename_left, full_path_filename_right, 2000, 3000)
