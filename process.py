import numpy as np
import cv2
import math
import time
from scipy.interpolate import UnivariateSpline, interp1d
from scipy import interpolate
from scipy import optimize
from scipy import integrate
from scipy import signal
from scipy import stats as st


def integrateImage(image_list, spline_x, spline_y, t_vals, arc_points):
        """
        Compute numerical integral of pixel intensity in an image along the given spline curve.
        Input:
        image_list (list): python list containing np.arrays of each image channel to be integrated
        spline_x (scipy.interpolate.UnivariateSpline): spline curve of x pixel positions (parameterized by t)
        spline_y (scipy.interpolate.UnivariateSpline): spline curve of y pixel positions (parameterized by t)
        t_vals (np.array): values of t that define points to take samples from
        arc_points (np.array): arc-length values of curve at points defined from spline(t_vals)

        Output:
        [total_val, ROI_length, width]
        total_val (np.array): array representing integrated values corresponding to  all points in arc_points
        ROI_length (np.array): image representing the straightened region of interest that we integrate. Useful for alignment.
        width (np.array): width distribution of food (deprecated)
        """
        x_deriv = spline_x.derivative()
        y_deriv = spline_y.derivative()

        average_val = []
        t_max = max(t_vals)
        ##Create sub-pixel sample points along skeleton
        radii = np.linspace(-25,25, 300)


        total_val = np.zeros((len(arc_points),len(image_list)))
        total_green_val = np.zeros(len(arc_points))
        global ROI, z, w
        ROI = np.zeros((300,len(arc_points)))
        ROI_green = np.zeros((300,len(arc_points)))
        i = 0
        theta_prev = 0
        for point_num in range(len(arc_points)): #for every sample point along arc
            t_current = t_vals[point_num]
            x = spline_x(t_current)
            y = spline_y(t_current)
            dy_dx = y_deriv(t_current)/x_deriv(t_current) #xy derivative gained from paramaterization... could result in error if denom is zero.
            #test.append(dy_dx)



            theta = np.arctan(dy_dx)
            theta_normal = theta + np.pi/2


            if point_num > 0:
                if theta_normal - theta_prev >= np.pi/2:
                    theta_normal = theta_normal - np.pi
                elif theta_prev - theta_normal >= np.pi/2:
                    theta_normal = theta_normal + np.pi
            theta_prev = theta_normal
            #want 150 sample points along normal line
            #15 px in both directions
            xs = radii * np.cos(theta_normal) + x
            ys = radii * np.sin(theta_normal) + y
            #plt.plot(xs,ys,"blue")
            #return
            floor_x = np.floor(xs-0.5)
            ceil_x = np.ceil(xs-0.5)
            floor_y = np.floor(ys-0.5)
            ceil_y = np.ceil(ys-0.5)

            #create block diagonal
            length = len(xs) * 2
            diag_mat = np.zeros((length, length,len(image_list)))
            #diag_mat_green = np.zeros((length, length))

            #global evens, lin
            evens = np.linspace(0,length-2, int(length/2))
            odds = evens + 1

            evens = evens.astype(int)
            odds = odds.astype(int)

            floor_x = floor_x.astype(int)
            floor_y = floor_y.astype(int)
            ceil_x = ceil_x.astype(int)
            ceil_y = ceil_y.astype(int)
            for image_idx in range(len(image_list)):
                diag_mat[evens, evens,image_idx] = image_list[image_idx][floor_y,floor_x]
                diag_mat[evens, odds,image_idx] = image_list[image_idx][ceil_y,floor_x]
                diag_mat[odds, evens,image_idx] = image_list[image_idx][floor_y,ceil_x]
                diag_mat[odds, odds,image_idx] = image_list[image_idx][ceil_y,ceil_x]
            #create flattened array
            xs = xs-0.5-floor_x
            ys = ys -0.5-floor_y
            one_minus_xs = 1-xs
            one_minus_ys = 1-ys
            interp_xs = np.array((one_minus_xs, xs)).T.flatten()#is two stacked on top of eo
            interp_ys = np.array((one_minus_ys, ys)).T.flatten()
            for image_idx in range(len(image_list)):
                total_val[point_num,image_idx] = (np.dot(interp_xs @ diag_mat[:,:,image_idx], interp_ys))
            #total_green_val[point_num] = (np.dot(interp_xs @ diag_mat_green, interp_ys))

            ##For visualizing spread of intensity
            diag_y_mat = np.zeros((length, int(length/2)))
            lin = np.linspace(0,int(length/2)-1, int(length/2))
            lin = lin.astype(int)
            diag_y_mat[evens,lin] = interp_ys[evens]
            diag_y_mat[odds,lin] = interp_ys[odds]
            #smoothed_signal = signal.savgol_filter(interp_xs @ diag_mat[:,:,1] @ diag_y_mat, 50, 3)

            ROI[:,i] = interp_xs @ diag_mat[:,:,0] @ diag_y_mat
            #ROI_green[:,i] = interp_xs @ diag_mat_green @ diag_y_mat

            i+=1

        SROI = ROI.copy()
        SROIG = ROI_green.copy()
        flatten = ROI.flatten()
        mode = st.mode(np.rint(flatten[flatten > 20]))[0][0]
        w = np.count_nonzero(ROI, axis = 0)
        #ROI[ROI == 0] = -mode
        width = countWidth(ROI)
        ROI_length = np.zeros((SROI.shape[0], math.floor(arc_points.max())+1))
        #ROI_green_length = np.zeros((SROIG.shape[0], math.floor(arc_points.max())+1))
        #plt.imshow(SROI)
        #plt.show()
        for row in range(SROI.shape[0]):
            #plt.plot(arc_points, ROI[row,:])
            #plt.show()
            func = interp1d(arc_points, SROI[row,:])
            ROI_length[row,:] = func(np.linspace(0,math.floor(arc_points.max()), math.floor(arc_points.max())+1))

            func_green = interp1d(arc_points, SROIG[row,:])

        return [total_val, ROI_length, width]


def arcLengthSpline(spline_x, spline_y, x_vals, y_vals):
    """
    Given splines of x and y return np array containing (x,y) of equally spaced points along curve
    """
    arc_points = np.zeros(len(x_vals)) #arc_points is list of cumulative arc lengts
    radii = np.linspace(-25,25, 300)
    i = 0
    theta_prev = 0
    for point_num in range(1,len(x_vals)): #for every sample point along arc
        x = x_vals[point_num] # current x coordinate
        y = y_vals[point_num] # current y coordinate

        arc_points[point_num] = arc_points[point_num-1] + math.sqrt((x-x_vals[point_num-1])**2 + (y-y_vals[point_num-1])**2) # populate cumulative arc length for each point
    return arc_points



def pointsOnSpline(spline_x, spline_y, t_max, resolution):
    """
    Given scipy spline objects of the t-parameterized curve of both x and y, and query t values, return numpy array representing points along body
    Input:
        spline_x (scipy.interpolate.UnivariateSpline) spline fit of x
        spline_y (scipy.interpolate.UnivariateSpline) spline fit of y
        resolution (int) number of points between pixels
        t_max (int) t-value to interpolate to
    Output:
        pts_on_curve (np.array)
    """
    t_on_curve = np.linspace(0,int(t_max), int(t_max*resolution))
    pts_on_curve = np.array([spline_x(t_on_curve), spline_y(t_on_curve)]).T
    return t_on_curve, pts_on_curve


def fitSpline(skeleton_points):
    '''
    Given a list of points representing skeleton ordered from head to tail or tail to head, return scipy spline fit
    Input:
        skeleton_points (np.array): Ordered array containing (x,y) of points along skeleton.

    Output:
        list: [spline_x, spline_y, t_vals, x_vals, y_vals]
        spline_x (scipy.interpolate.UnivariateSpline) spline fit of x
        spline_y (scipy.interpolate.UnivariateSpline) spline fit of y
    '''
    x_vals = []
    y_vals = []
    t_vals = []

##Fit skeleton to spline curve
    for point_num in range(len(skeleton_points)-1): #Arrange x, y values in increasing order
        x_vals.append(skeleton_points[point_num][1]) #look here for flipped x,y
        y_vals.append(skeleton_points[point_num][0])
        t_vals.append(point_num)
    t_vals.append(len(skeleton_points)) # assign parameter t to each point
    x_vals.append(skeleton_points[-1][1])
    y_vals.append(skeleton_points[-1][0])
    spline_x = UnivariateSpline(t_vals, x_vals) # fit x values to spline
    #global x_deriv, y_deriv
    x_deriv = spline_x.derivative() # x derivative
    spline_y = UnivariateSpline(t_vals,y_vals) #fit y values to spline
    y_deriv = spline_y.derivative() #y derivative

    #ax2.plot(spline_x(t_vals), spline_y(t_vals))

    return [spline_x, spline_y, t_vals, x_vals, y_vals]



#intestine_start_pos = 470
#intestine_max_pos = 570


def correlation_lags(in1_len, in2_len, mode='full'):
    # calculate lag ranges in different modes of operation
    if mode == "full":
        # the output is the full discrete linear convolution
        # of the inputs. (Default)
        lags = np.arange(-in2_len + 1, in1_len)
    elif mode == "same":
        # the output is the same size as `in1`, centeresshs
        # with respect to the 'full' output.
        # calculate the full output
        lags = np.arange(-in2_len + 1, in1_len)
        # determine the midpoint in the full output
        mid = lags.size // 2
        # determine lag_bound to be used with respect
        # to the midpoint
        lag_bound = in1_len // 2
        # calculate lag ranges for even and odd scenarios
        if in1_len % 2 == 0:
            lags = lags[(mid-lag_bound):(mid+lag_bound)]
        else:
            lags = lags[(mid-lag_bound):(mid+lag_bound)+1]
    elif mode == "valid":
        # the output consists only of those elements that do not
        # rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
        # must be at least as large as the other in every dimension.

        # the lag_bound will be either negative or positive
        # this let's us infer how to present the lag range
        lag_bound = in1_len - in2_len
        if lag_bound >= 0:
            lags = np.arange(lag_bound + 1)
        else:
            lags = np.arange(lag_bound, 1)
    return lags



def bilinear_interpolation(x, y, points):
    '''Interpolate (x,y) from values associated with four points.

    The four points are a list of four triplets:  (x, y, value).
    The four points can be in any order.  They should form a rectangle.

        >>> bilinear_interpolation(12, 5.5,
        ...                        [(10, 4, 100),
        ...                         (20, 4, 200),
        ...                         (10, 6, 150),
        ...                         (20, 6, 300)])
        165.0

    '''
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

    points = sorted(points)               # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError('(x, y) not within the rectangle')

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1) + 0.0)


def interpolateIntensity(values, xpoints, max_arc_length,spacing): #only for first frame now
    """
    Interpolate the array "values" at integer arc length inputs.
    Input:
    values (np.array): array representing the food distribution function evaluated at non equispaced intervals (non equispaced due to how we step along the spline curve of worm)
    xpoints (np.array): array representing the arc length values of each corresponding value
    max_arc_length (int): integer representing the rounded max domain of sequence. We create an array of length "max_arc_length" for the output interpolated sequence.
    spacing (int): integer representing how many inter-integer points we want to output.

    Output:
    interpolated_vals (np.array): array representing the food distribution at standardized domain values. Useful for kymograph representation.
    """

    row_vec = np.zeros((max_arc_length*spacing+400))
    if xpoints[0] > 0:
        initial_l = xpoints[0]
        xpoints = np.pad(xpoints, [(math.ceil(initial_l),0)],mode = 'linear_ramp')
        for val in range(len(values)):
            values[val] = np.pad(values[val], [(math.ceil(initial_l),0)],mode='constant')

    diff = math.ceil(max_arc_length - xpoints.max()) #max arc length for first frame is always gonna be lower by 10%. Diff is positive
    if diff > 0: #fill with zeros
        xpoints = np.pad(xpoints, [(0,diff)], mode='linear_ramp',end_values = ((0,xpoints.max() + diff)))
        for val in range(len(values)):
            values[val] = np.pad(values[val], [(0,int(diff))],mode = 'constant')

        #print("change?")
    #plt.plot(xpoints,total_val)
    #plt.show()
    dict_func = {}
    for val in range(len(values)):
        dict_func[val] = interp1d(xpoints, values[val])

    '''Change based on expected length'''
    xnew = np.linspace(1,max_arc_length,max_arc_length*spacing)
    interpolated_vals = []
    for val in range(len(values)):
        row_vec[200:max_arc_length*spacing+200] = dict_func[val](xnew)
        interpolated_vals.append(row_vec.copy())
    return interpolated_vals.copy()


def countWidth(FI):
    #split down middle,
    FI = FI.copy()
    width = np.zeros(FI.shape[1])
    half1 = FI.shape[0] // 2
    half2 = FI.shape[0] - half1
    for col_num in range(FI.shape[1]):
        col = FI[:,col_num]
        col[col < 0] = -1
        col[col > 0] = 0
        col_first_half = col[:half1]
        num_zeros_first_half = len(col_first_half) - len(np.trim_zeros(col_first_half,'b'))
        col_second_half = col[half2:]
        num_zeros_second_half = len(col_second_half) - len(np.trim_zeros(col_second_half,'f'))
        width[col_num] = num_zeros_first_half + num_zeros_second_half
    return width

def displayThreshold(image, spline_x, spline_y, t_vals, x_vals,y_vals):
        x_deriv = spline_x.derivative()
        y_deriv = spline_y.derivative()

        average_val = []
        t_max = max(t_vals)
        x1_vals = x_vals[:]
        y1_vals = y_vals[:]
        t1_vals = t_vals[:]
        x_vals = [] #x coordinate list of points to be sampled along arc
        y_vals = [] #y coordinate list of points to be sampled along arc
        t_vals = [] #t paramaterization
        ##Create sub-pixel sample points along skeleton
        for sample_pt in range(t_max*4): #currently we have one sample point per pixel gained from the skeleton, but since we have the spline fit we can sample subpixel points along arc.
            x_vals.append(float(spline_x(sample_pt*0.25)))
            y_vals.append(float(spline_y(sample_pt*0.25)))
            t_vals.append(sample_pt*0.25)
        #self.t_vals = t_vals
        #self.x_vals = x_vals
        #self.y_vals = y_vals
        #times = []
        #test = []
        arc_points = np.zeros(len(x_vals)) #arc_points is list of cumulative arc length
        arc_points[0] = 0
        #splx = UnivariateSpline(t_vals,x_vals)
        #sply = UnivariateSpline(t_vals,y_vals)
        #dx = splx.derivative()
        #dy = sply.derivative()
        radii = np.linspace(-25,25, 300)


        total_val = np.zeros(len(x_vals))
        global ROI, z, w
        ROI = np.zeros((300,len(x_vals)))
        i = 0
        switch = True
        for point_num in range(len(x_vals)): #for every sample point along arc
            #print(len(x_vals))
            x = x_vals[point_num] # current x coordinate
            y = y_vals[point_num] # current y coordinate
            #if point_num == 1720:
                #print(x,y)
            if point_num ==0:
                arc_points[point_num] = 0
            else:
                arc_points[point_num] = arc_points[point_num-1] + math.sqrt((x-x_vals[point_num-1])**2 + (y-y_vals[point_num-1])**2) # populate cumulative arc length for each point
            t_current = t_vals[point_num]
            dy_dx = y_deriv(t_current)/x_deriv(t_current) #xy derivative gained from paramaterization... could result in error if denom is zero.
            #test.append(dy_dx)



            theta = np.arctan(dy_dx)
            if switch == True:
                print(theta)
                switch = False
            theta_normal = theta + np.pi/2


            if point_num > 0:
                if theta_normal - theta_prev >= np.pi/2:
                    theta_normal = theta_normal - np.pi
                elif theta_prev - theta_normal >= np.pi/2:
                    theta_normal = theta_normal + np.pi
            theta_prev = theta_normal
            #want 150 sample points along normal line
            #15 px in both directions
            xs = radii * np.cos(theta_normal) + x
            ys = radii * np.sin(theta_normal) + y
            #plt.plot(xs,ys,"blue")
            #return
            floor_x = np.floor(xs-0.5)
            ceil_x = np.ceil(xs-0.5)
            floor_y = np.floor(ys-0.5)
            ceil_y = np.ceil(ys-0.5)

            #create block diagonal
            length = len(xs) * 2
            diag_mat = np.zeros((length, length))

            #global evens, lin
            evens = np.linspace(0,length-2, int(length/2))
            odds = evens + 1

            evens = evens.astype(int)
            odds = odds.astype(int)

            floor_x = floor_x.astype(int)
            floor_y = floor_y.astype(int)
            ceil_x = ceil_x.astype(int)
            ceil_y = ceil_y.astype(int)
            #for i in range(len(xs)):
                #print(i)
                #diag_mat[i*2, i*2] = img[int(floor_y[i]),int(floor_x[i])]
                #diag_mat[i*2, i*2+1] = img[int(ceil_y[i]),int(floor_x[i])]
                #diag_mat[i*2+1, i*2] = img[int(floor_y[i]),int(ceil_x[i])]
                #diag_mat[i*2+1, i*2+1] = img[int(ceil_y[i]),int(ceil_x[i])]
            diag_mat[evens, evens] = image[floor_y,floor_x]
            diag_mat[evens, odds] = image[ceil_y,floor_x]
            diag_mat[odds, evens] = image[floor_y,ceil_x]
            diag_mat[odds, odds] = image[ceil_y,ceil_x]

            #create flattened array
            xs = xs-0.5-floor_x
            ys = ys -0.5-floor_y
            one_minus_xs = 1-xs
            one_minus_ys = 1-ys
            interp_xs = np.array((one_minus_xs, xs)).T.flatten()#is two stacked on top of eo
            interp_ys = np.array((one_minus_ys, ys)).T.flatten()
            total_val[point_num] = (np.dot(interp_xs @ diag_mat, interp_ys))

            ##For visualizing spread of intensity
            diag_y_mat = np.zeros((length, int(length/2)))
            lin = np.linspace(0,int(length/2)-1, int(length/2))
            lin = lin.astype(int)
            diag_y_mat[evens,lin] = interp_ys[evens]
            diag_y_mat[odds,lin] = interp_ys[odds]
            smoothed_signal = signal.savgol_filter(interp_xs @ diag_mat @ diag_y_mat, 50, 3)

            ROI[:,i] = interp_xs @ diag_mat @ diag_y_mat

            i+=1

        flatten = ROI.flatten()
        #mode = st.mode(np.rint(flatten[flatten > 20]))[0][0] + threshold
        #ROI[ROI < mode] = 0 #try no threshold
        return ROI




def correlate2d(mat1,mat2):
    """
    Compute 2d cross correlation between 2 arrays, only along columns. mat2 is sliding on mat1

    """
    #region 1, mat1 is sliding onto mat2
    mat1_length = mat1.shape[1]
    mat2_length = mat2.shape[1]
    mat2 = mat2-mat2.mean()

    if mat1_length < mat2_length:
        z = np.zeros((mat1.shape[0],(mat2_length-mat1_length)*2))
        print(z.shape)
        mat1 = np.concatenate((z,mat1,z),axis = 1)
        #print(mat1.shape,mat2.shape)
        mat1_length = mat1.shape[1]
    #print(mat1_length, mat2_length)

    corr = np.zeros((mat1_length-mat2_length))
    #for i in range(1,mat2_length):
    #    res = np.sum(np.multiply(mat1[:,:i],mat2[:,-i:]))
    #    #print(res)
    #    corr[i-1] = res
    for i in range(mat1_length-mat2_length):
        res = np.sum(np.multiply(mat1[:,i:i+mat2_length]-mat1[:,i:i+mat2_length].mean(), mat2))
        res = res/math.sqrt(np.sum(np.multiply(np.power(mat1[:,i:i+mat2_length]-mat1[:,i:i+mat2_length].mean(),2), np.power(mat2,2))))

        corr[i] = res
    #for i in range(1,mat2_length):
    #    res = np.sum(np.multiply(mat1[:,-i:],mat2[:,:i]))
    #    corr[i+mat1_length-1] = res
    return corr