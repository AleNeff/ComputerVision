import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from skimage import filters, feature, img_as_int
from skimage.measure import regionprops
from nonmax import non_max_suppression

def plot_interest_points(image, x, y):
    '''
    Plot interest points for the input image. 
    
    Show the interest points given on the input image. Be sure to add the images you make to your writeup. 

    Useful functions: Some helpful (not necessarily required) functions may include
        - matplotlib.pyplot.imshow, matplotlib.pyplot.scatter, matplotlib.pyplot.show, matplotlib.pyplot.savefig
    
    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    '''

    # TODO: Your implementation here! See block comments and the handout pdf for instructions
    plt.scatter(x,y,zorder=1, color = 'none', edgecolor='red')
    plt.imshow(image, zorder=0, cmap='gray') # may need to use extent param
    plt.show()

def get_interest_points(image, feature_width):
    '''
    Returns interest points for the input image

    (Please note that we recommend implementing this function last and using cheat_interest_points()
    to test your implementation of get_features() and match_features())

    Implement the Harris corner detector (See Szeliski 7.1.1) to start with.
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious (false/fake) interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.feature.peak_local_max (experiment with different min_distance values to get good results)
        - skimage.measure.regionprops
          
    Note: You may decide it is unnecessary to use feature_width in get_interest_points, or you may also decide to 
    use this parameter to exclude the points near image edges.

    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width: the width and height of each local feature in pixels

    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point

    '''

    # TODO: Your implementation here! See block comments and the handout pdf for instructions

    # These are placeholders - replace with the coordinates of your interest points!

    xs = np.zeros(1)
    ys = np.zeros(1)
    # Note that xs and ys represent the coordinates of the image. Thus, xs actually denote the columns
    # of the respective points and ys denote the rows of the respective points.

    # STEP 1: Calculate the gradient (partial derivatives on two directions).
    # STEP 2: Apply Gaussian filter with appropriate sigma.
    # STEP 3: Calculate Harris cornerness score for all pixels.
    # STEP 4: Peak local max to eliminate clusters. (Try different parameters.)
    
    # BONUS: There are some ways to improve:
    # 1. Making interest point detection multi-scaled.
    # 2. Use adaptive non-maximum suppression.

    sobelx = cv2.Sobel(image,ddepth=-1,dx=1,dy=0,ksize=3)
    sobely = cv2.Sobel(image,ddepth=-1,dx=0,dy=1,ksize=3)
    img_dx = cv2.filter2D(image, ddepth=-1, kernel = sobelx)
    img_dy = cv2.filter2D(image, ddepth=-1, kernel = sobely)
    img_ddx = cv2.filter2D(img_dx, ddepth=-1, kernel = sobelx)
    img_ddy = cv2.filter2D(img_dy, ddepth=-1, kernel = sobely)
    img_dxdy = cv2.filter2D(img_dy, ddepth=-1, kernel = sobelx)

    # error here trying to use spatialGradient

    # image = (image * 255).astype(np.uint8)
    # img_dx, img_dy = cv2.spatialGradient(image, borderType=cv2.BORDER_REPLICATE)
    # img_ddx = img_dx**2
    # img_dxdy = img_dx * img_dy
    # img_ddy = img_dy**2

    # img_ddx, img_dxdy = cv2.spatialGradient(img_dx, borderType=cv2.BORDER_REPLICATE)
    # img_dxdy, img_ddy = cv2.spatialGradient(img_dy, borderType=cv2.BORDER_REPLICATE)

    M_mat = np.ndarray(shape = [image.shape[0], image.shape[1], 2, 2])
    alpha = 0.04
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            M_mat[i][j][0][0] = img_ddx[i][j]
            M_mat[i][j][0][1] = img_dxdy[i][j]
            M_mat[i][j][1][0] = img_dxdy[i][j]
            M_mat[i][j][1][1] = img_ddy[i][j]
    
    img_ddx = cv2.GaussianBlur(src=img_ddx, ksize=[3,3], sigmaX=1, sigmaY=1)
    img_dxdy = cv2.GaussianBlur(src=img_dxdy, ksize=[3,3], sigmaX=1, sigmaY=1)
    img_ddy = cv2.GaussianBlur(src=img_ddy, ksize=[3,3], sigmaX=1, sigmaY=1)
    # plt.imshow(img_dxdy, cmap='gray')
    # plt.show()

    M_det = np.ndarray(image.shape)
    M_trace = np.ndarray(image.shape)
    C_mat = np.ndarray(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            M_det[i][j] = np.linalg.det(M_mat[i][j])
            M_trace[i][j] = np.trace(M_mat[i][j])
            C_mat[i][j] = M_det[i][j] - alpha * M_trace[i][j]

    t = 0.0005
    C_mat = C_mat * (C_mat > t)
    peaks = feature.peak_local_max(C_mat, min_distance=int(feature_width/2), num_peaks = 1500, exclude_border=True)
    #  threshold_abs=C_mat_med, threshold_rel=C_mat_med, 
    peaks = np.array(peaks)
    xs = peaks[:,1]
    ys = peaks[:,0]

    return xs, ys


def get_features(image, x, y, feature_width):
    '''
    Returns features for a given set of interest points.

    To start with, you might want to simply use normalized patches as your
    local feature descriptor. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT-like feature descriptor
    (See Szeliski 7.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) feature descriptor should have:
    (1) a 4x4 grid of cells, each feature_width / 4 pixels square.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like features can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.filters (library)


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments. Make sure input arguments 
    are optional or the autograder will break.

    :returns:
    :features: np array of computed features. It should be of size
            [len(x) * feature dimensionality] (for standard SIFT feature
            dimensionality is 128)

    '''

    # TODO: Your implementation here! See block comments and the handout pdf for instructions
    
    # STEP 1: Calculate the gradient (partial derivatives on two directions) on all pixels.
    # STEP 2: Decompose the gradient vectors to magnitude and direction.
    # STEP 3: For each interest point, calculate the local histogram based on related 4x4 grid cells.
    #         Each cell is a square with feature_width / 4 pixels length of side.
    #         For each cell, we assign these gradient vectors corresponding to these pixels to 8 bins
    #         based on the direction (angle) of the gradient vectors. 
    # STEP 4: Now for each cell, we have a 8-dimensional vector. Appending the vectors in the 4x4 cells,
    #         we have a 128-dimensional feature.
    # STEP 5: Don't forget to normalize your feature.
    
    # BONUS: There are some ways to improve:
    # 1. Use a multi-scaled feature descriptor.
    # 2. Borrow ideas from GLOH or other type of feature descriptors.

    gray_img = image
    # applying a gaussian filter seemed to help accuracy
    g_filter = cv2.getGaussianKernel(3, 1)
    gray_img = cv2.filter2D(gray_img, -1, g_filter)
    features = np.ndarray(shape=[len(x), 256]).astype(np.float32)
    s_x = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.int64)
    s_y = np.asarray([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.int64)
    angles = [0, 45, 90, 135, 180, 225, 270, 315]

    filter_set = np.ndarray(shape=[len(angles), s_x.shape[0], s_x.shape[1]])
    for i in range(len(angles)):
        filter_set[i] = np.asarray(np.cos(np.deg2rad(angles[i])) * s_x + np.sin(np.deg2rad(angles[i])) * s_y)

    oriented_imgs = [cv2.filter2D(gray_img, -1, filt) for filt in filter_set]
    oriented_imgs = np.array(oriented_imgs)
    # set of angle yielding highest gradient in image from oriented filters
    grad_ori = np.ndarray(shape=gray_img.shape)
    grad_mag = np.ndarray(shape=gray_img.shape)
    for i in range(oriented_imgs.shape[1]):
        for j in range(oriented_imgs.shape[2]):
            max_grad_angle_arg = np.argmax(oriented_imgs[:,i,j])
            grad_ori[i][j] = max_grad_angle_arg
            grad_mag[i][j] = oriented_imgs[max_grad_angle_arg][i][j]

# using approach #1 from Dr. Wang, where POI pixel is offset from center of feature box
    for k in range(len(x)):
        for i in range(-1, 3):
            for j in range(-1, 3):
                start_pixel_x = x[k] + 4*j
                start_pixel_y = y[k] + 4*i
                for u in range(0, 4):
                    for v in range(0, 4):
                        # compute bin, [0 - 7], for orientation
                        chunk_start = (32 * (i+1)) + (8 * (j+1))
                        # avoid out-of-bounds
                        if start_pixel_y+u >= gray_img.shape[0] or start_pixel_x+v >= len(gray_img[1]): continue
                        # non-SIFT intensity histograms approach that I used initially
                        # features[k][chunk_start+v+4*u] = gray_img[start_pixel_y+u][start_pixel_x+v]
                        bin_index = int(grad_ori[start_pixel_y + u][start_pixel_x + v])
                        chunk_start = (32 * (i+1)) + (8 * (j+1))
                        # approach if I wanted to add magnitude to bin instead of just increment by 1
                        # features[k][chunk_start + bin_index] += grad_mag[start_pixel_y + u][start_pixel_x + v]
                        features[k][chunk_start + bin_index] += 1
        # features[k] = features[k] / np.linalg.norm(features[k])
        # illumination-invariant normalization
        features[k] = features[k] / np.max(features[k])
        for f in range(len(features[k])):
            if features[k][f] > 0.3:
                features[k][f] = 0.3
        features[k] = features[k] * (1/np.max(features[k]))

    return features


def match_features(im1_features, im2_features):
    '''
    Implements the Nearest Neighbor Distance Ratio Test to assign matches between interest points
    in two images.

    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 7.18 in Section 7.1.3 of Szeliski.

    For extra credit you can implement spatial verification of matches.

    Please assign a confidence, else the evaluation function will not work. Remember that
    the NNDR test will return a number close to 1 for feature points with similar distances.
    Think about how confidence relates to NNDR.

    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).

    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features

    :params:
    :im1_features: an np array of features returned from get_features() for interest points in image1
    :im2_features: an np array of features returned from get_features() for interest points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    '''

    # TODO: Your implementation here! See block comments and the handout pdf for instructions

    # These are placeholders - replace with your matches and confidences!
    
    # STEP 1: Calculate the distances between each pairs of features between im1_features and im2_features.
    #         HINT: check match_features.pdf
    # STEP 2: Sort and find closest features for each feature, then performs NNDR test.
    
    # BONUS: Using PCA might help the speed (but maybe not the accuracy).

    matches = []
    confidences = []

    mat_B = np.matmul(im1_features, np.transpose(im2_features)) * 2

    mat_f1 = np.sum((im1_features*im1_features), axis=1)
    mat_f2 = np.sum((im2_features*im2_features), axis=1)
    mat_f1 = np.expand_dims(mat_f1, axis=1)
    mat_f2 = np.expand_dims(mat_f2, axis=0)

    mat_A = mat_f1 + mat_f2
    mat_D = np.sqrt(mat_A-mat_B + 1e-9)
    # print(mat_A.shape)
    # print(mat_B.shape)
    # print(mat_D)
    
    feat_dists_arg_sorted = np.argsort(mat_D, axis=1)
    # now perform ratio test for each of n feature to the first two of the m features
    RATIO_CUTOFF = 1

    for i in range(feat_dists_arg_sorted.shape[0]):
        ratio = mat_D[i][feat_dists_arg_sorted[i][0]] / mat_D[i][feat_dists_arg_sorted[i][1]]
        if ratio < RATIO_CUTOFF:
            coord_pair = [i, feat_dists_arg_sorted[i][0]]
            matches.append(coord_pair)
            # print(coord_pair)
            confidences.append(1-ratio)
    # matches_temp = [match for _, match in sorted(zip(confidences, matches), reverse=True)]
    # confidences_temp = [conf for conf,_ in sorted(zip(confidences, matches), reverse=True)]
    # matches = matches_temp
    # confidences = confidences_temp

    confidences = np.array(confidences)
    conf_inds = np.flip(confidences.argsort())
    matches = np.array(matches)[conf_inds]
    confidences = np.array(confidences)[conf_inds]
    # print(matches.shape)

    return matches, confidences
