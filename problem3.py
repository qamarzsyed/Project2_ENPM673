import numpy as np
import cv2

vid = cv2.VideoCapture('challenge.mp4')

# get initial video dimensions to make region mask
width = int(vid.get(3))
height = int(vid.get(4))
center = (height - 100) // 2
slope = -center / ((width - 0) / 2)

problem_3 = cv2.VideoWriter('Problem_3.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

# mask except for a certain triangle to only see lanes next to car
region_mask = [[1 if 720 - y <= int(center + slope * abs(x - width // 2)) else 0 for x in range(0, width)] for y in
               range(0, height)]
region_mask = (np.array(region_mask)).astype(np.uint8)

# initializing a few variables
lw = None
ly = None

frames = 0
bad = 0
left = 0
right = 0

# loop across frames
while True:
    stat, frame = vid.read()
    if not stat:
        break

    # apply mask to video on triangle region
    region = cv2.bitwise_and(frame, frame, mask=region_mask)

    # use hsv filtering to find the yellow lane and make a masked video with it
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([10, 70, 180])
    upper_yellow = np.array([70, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    my = cv2.bitwise_and(region, region, mask=yellow_mask)

    # use hls to filter for white and find white lanes and make masked video
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    lower = np.array([0, 200, 0])
    upper = np.array([255, 255, 255])
    mask_white = cv2.inRange(hls, lower, upper)
    mw = cv2.bitwise_and(region, region, mask=mask_white)

    # make a video with both lanes using the masks
    # also use edge detect on the individual lane masks to find the edges of each lane
    masked = cv2.bitwise_and(region, region, mask=yellow_mask + mask_white)
    canny_y = cv2.Canny(my, 200, 50)
    canny_w = cv2.Canny(mw, 200, 50)

    # cut off interrupting white colors such as the sky from the video
    canny_w[0: 195 * mask_white.shape[0] // 300, :] = 0

    # find the yellow and white lane lines via Probabilistic Hough Transform
    line_y = cv2.HoughLinesP(canny_y, 1, np.pi / 180, 25, None, 250, 500)
    line_w = cv2.HoughLinesP(canny_w, 1, np.pi / 180, 40, None, 250, 750)

    # get coords if lines exist
    # NOTE: the hough transform here was made very light to ensure the lines are detected, the proper line
    # analysis is done later in the code and has a backup if it does not succeed
    if line_y is not None:
        for i in range(0, 1):
            ly = line_y[i][0]
            x1_y, y1_y, x2_y, y2_y = ly[0:4]
        # cv2.line(frame, (ly[0], ly[1]), (ly[2], ly[3]), (0,0,255), 3, cv2.LINE_AA)

    if line_w is not None:
        for i in range(0, 1):
            lw = line_w[i][0]
            x1_w, y1_w, x2_w, y2_w = lw[0:4]
            # cv2.line(frame, (lw[0], lw[1]), (lw[2], lw[3]), (0,0,255), 3, cv2.LINE_AA)

    if ly is not None and lw is not None:
        # get homography matrix from the lanes to a rectangle to get a top down view of the lanes
        # apply this to get the top down
        image_loc = np.float32([[x2_y, y2_y], [x1_y, y1_y], [x1_w, y1_w], [x2_w, y2_w]])
        h_size = np.float32([[100, 0], [100, 1000], [900, 0], [900, 1000]])
        H = cv2.getPerspectiveTransform(image_loc, h_size)
        topdown = cv2.warpPerspective(frame, H, (1000, 1000))

        # using lab coloring to filter for yellow channel for yellow lane
        LAB_td = cv2.cvtColor(topdown, cv2.COLOR_BGR2LAB)
        yellow_td = LAB_td[:, :, 2]
        yellow_mask_td = cv2.inRange(yellow_td, 150, 255)
        td_masked = cv2.bitwise_and(topdown, topdown, mask=yellow_mask_td)

        # make video overlay with the yellow lane that goes on top of the lane the car is in
        # uses masks and then does homography back to the original image and makes it colored where the masks are
        midlane = np.zeros((1000, 1500))
        lane_pts = np.nonzero(yellow_mask_td)
        lane = np.zeros((1000, 1500))
        for i in range(0, 750):
            other = lane_pts[1] + i
            other_pts = (lane_pts[0], other)
            midlane[other_pts] = 255
        midlane[lane_pts] = 0
        other = lane_pts[1] + 800
        other_pts = (lane_pts[0], other)
        lane[lane_pts] = 255
        lane[other_pts] = 255

        H2 = cv2.getPerspectiveTransform(h_size, image_loc)
        midlane_overlay = cv2.warpPerspective(midlane, H2, (width, height))
        lane_overlay = cv2.warpPerspective(lane, H2, (width, height))

        over_pts = np.nonzero(midlane_overlay)
        lane_over_pts = np.nonzero(lane_overlay)
        frame[over_pts[0], over_pts[1], 2] = 255
        frame[lane_over_pts[0], lane_over_pts[1]] = [0, 0, 255]
        # also making the lane overlay for the corner
        lane_overlay = cv2.resize(lane, dsize=(250, 250))
        lane_overlay = np.stack((lane_overlay, lane_overlay, lane_overlay), axis=-1)

        # erode the image to get thinner lines for the lane
        kernel = (np.ones((3, 3))).astype(np.uint8)
        kernel[1, 1] = 0
        yellow_mask_td = cv2.erode(yellow_mask_td, kernel, iterations=5)

        # find nonzero points in the mask of the lane to find all the point values of the lane
        points = np.nonzero(yellow_mask_td)

        # find a corresponding x, y pair; this is the location to determine the radius of curvature
        x_deriv = 0
        j = 0
        while x_deriv == 0:
            if len(np.nonzero(yellow_mask_td[250 + j, :])[0]):
                x_deriv = np.nonzero(yellow_mask_td[250 + j, :])[0][0]
            j += 1

        # find the curve of best fit with polyfit for a quadratic equation and calculate the residual per point
        best_fit_curve = np.polyfit(points[1], points[0], 2, full=True)
        resid = best_fit_curve[1][0] / len(points[0])

        # if the residual is low enough, this is an accurate line and its equation will be calculated to apply
        # if it is not accurate, the image statistics will continue to be calculated off of the old equation
        if resid < 50000 or frames == 0:
            x = np.linspace(0, 1000, num=1000)
            y = np.polyval(best_fit_curve[0], x)
            # determining right or left using first derivative at point in the lane to see slope
            r_l = 2 * best_fit_curve[0][0] * x_deriv + best_fit_curve[0][1]
        else:
            bad += 1

        # draw_points = np.asarray([x, y]).T.astype(np.int32)
        # cv2.polylines(yellow_mask_td, [draw_points], False, 125, thickness=5)

        # calculate the radius of curvature with the derivatives
        d1 = 2 * best_fit_curve[0][0] * x_deriv + best_fit_curve[0][1]
        d2 = abs(2 * best_fit_curve[0][0])
        radius = pow(1 + pow(d1, 2), 1.5) / d2

        # determine if line is straight, right, or left based on derivative of the lane
        rls = 0
        if r_l < -20:
            rls = 1
            turn = 'Right'
            right += 1
        elif r_l > 20:
            turn = 'Left'
            rls = 2
            left += 1
        else:
            turn = 'Straight'
            rls = 0

        # apply overlays and text
        radius_txt = 'Radius of Curvature: ' + str(radius)
        cv2.putText(frame, radius_txt, (0, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
        cv2.putText(frame, turn, (width//2 - 200, height//2), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255))
        frame[-250:, -250:] = lane_overlay
        problem_3.write(frame)
    frames += 1
problem_3.release()


