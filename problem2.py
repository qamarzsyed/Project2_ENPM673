import numpy as np
import cv2


vid = cv2.VideoCapture('whiteline.mp4')

# getting video dimensions to make a mask that only shows a desired height triangle starting from the bottom corners
# ideally narrows down the FOV to only the two lanes the car is in
width = int(vid.get(3))
height = int(vid.get(4))
center = 3*height//7
slope = -center/(width/2)

problem_2 = cv2.VideoWriter('Problem_2.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

# makes mask from dimensions given above
region_mask = [[1 if height-y <= int(center + slope*abs(x-width//2)) else 0 for x in range(0, width)] for y in range(0, height)]
region_mask = (np.array(region_mask)).astype(np.uint8)

# start loop to go through each frame of video
while True:
    stat, frame = vid.read()
    if not stat:
        break

    # add the mask made above to the video
    region = cv2.bitwise_and(frame, frame, mask=region_mask)

    # use HLS color space and filter for lightness to create a mask for the white color of the lanes and apply it
    hls = cv2.cvtColor(region, cv2.COLOR_BGR2HLS)
    lower = np.array([0, 200, 0])
    upper = np.array([255, 255, 255])
    mask = cv2.inRange(hls, lower, upper)
    # ended up filtering out some more height that isn't needed and detects too much
    mask[0:175 * mask.shape[0] // 300, :] = 0
    masked = cv2.bitwise_and(region, region, mask=mask)

    # turn the masked image to grayscale and threshold it to just get a binary image with only the lanes
    # then do edge detection on the binary image to get the outside edges of the lanes
    grey = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grey, 200, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(thresh, 100, 50)

    # apply a dilation to the lanes to increase their size, this will be used later when making the visualization
    # and makes the lanes from the thresh easier to see
    kernel = np.ones((3,3))
    thresh = cv2.dilate(thresh, kernel=kernel, iterations=2)

    # use probabilistic Hough lines transform to find the line segments on the image in order of the 'best' lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 25, None, 250, 500)

    if lines is not None:
        # finding coords of best line, weight added to fill in undetected portions of solid line later
        solid = lines[0][0]
        weight = 70
        x1 = solid[0] - weight
        y1 = solid[1] - weight
        x2 = solid[2] + weight
        y2 = solid[3] + weight

        # apply red onto the initial frame around the coordinates for the 'best' solid line and apply green elsewhere
        solid_color = thresh[:, x1:x2]
        dashed_color = thresh[:, :]
        dashed_indexes = np.nonzero(dashed_color)
        solid_indexes = np.nonzero(solid_color)

    frame[dashed_indexes[0], dashed_indexes[1]] = [0, 255, 0]
    frame[solid_indexes[0], solid_indexes[1] + x1] = [0, 0, 255]

    problem_2.write(frame)

problem_2.release()

