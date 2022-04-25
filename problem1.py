import numpy as np
import cv2

# list and iteration through every image in the folder
images = np.arange(0, 25, 1, dtype=int)

for number in images:
    if number >= 10:
        select_image_number = str(number)
    else:
        select_image_number = '0' + str(number)
    img = cv2.imread('adaptive_hist_data/00000000' + select_image_number + '.png')

    # start by splitting up each color channel
    blue = img[:, :, 0]
    green = img[:, :, 1]
    red = img[:, :, 2]

    # starting regular histogram equalization
    # function to normalize each color by creating CFD and multiplying that by the color value
    def equalized(color):
        cfd = np.zeros(256)
        for sub in color:
            for pix in sub:
                cfd[pix:] += 1/color.size

        return (np.round(color*cfd[color])).astype(np.uint8)

    # using function to equalize all colors and join into bgr image
    blue_eq = equalized(blue)
    green_eq = equalized(green)
    red_eq = equalized(red)
    img_eq = np.stack((blue_eq, green_eq, red_eq), axis=-1)

    # starting adaptive equalization
    # it doesn't seem like the assignment mentions contrast limiting so that was not attempted

    # function to equalize by 8x8 bins across a color channel
    def adaptive_equalized(color):
        tile_size = 8
        final = np.zeros(color.shape)
        for i in range(0, color.shape[0], tile_size):
            for j in range(0, color.shape[1], tile_size):
                # determine tile size by looking if the full area is possible or it is at the edge and should be clipped
                if i + tile_size > color.shape[0] and j + tile_size > color.shape[1]:
                    i_max = color.shape[0]
                    j_max = color.shape[1]
                elif i + tile_size > color.shape[0]:
                    i_max = color.shape[0]
                    j_max = j + tile_size
                elif j + tile_size > color.shape[1]:
                    i_max = i + tile_size
                    j_max = color.shape[1]
                else:
                    i_max = i + tile_size
                    j_max = j + tile_size

                # use the histogram equalization function to get new tile and insert into return image
                tile = color[i:i_max, j:j_max]
                final[i:i_max, j:j_max] = equalized(tile)

        # return image with equalized tiles
        return (np.round(final)).astype(np.uint8)

    # use function to make all the equalized images and combine into one image
    blue_aeq = adaptive_equalized(blue)
    green_aeq = adaptive_equalized(green)
    red_aeq = adaptive_equalized(red)
    img_aeq = np.stack((blue_aeq, green_aeq, red_aeq), axis=-1)

    # save images into respective folders
    cv2.imwrite('equalized_images/00000000' + select_image_number + '.png', img_eq)
    cv2.imwrite('adaptive_equalized_images/00000000' + select_image_number + '.png', img_aeq)




