import cupy as cp
import numpy as np
import torch
from numpy.polynomial import polynomial as poly
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

def remove_outliers_iqr(data, k=1.5):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return data[(data >= lower) & (data <= upper)]

def tric_apex_distance_calculation(window, 
                                   method="triangle",
                                   ):
    """
    Calculate the distance from the apex to the free wall and septum.
    Also calculates the diameter of the right ventricle, and the area of the region
    defined by those three points.
    
    Parameters
    ----------
    window: ndarray (..., 3, 2) 
        Coordinates of the free wall, septum, and apex at each time point.
    method : str, optional
        'triangle' (Heron's formula) or 'spline' (area under spline through points).
    
    Returns
    -------
    rvfac, diast_area, syst_area, rvldfw, rvldsep, rvlsfw, rvlssep,
    rvldmid, rvlsmid, tadd, tasd, rvlsffw, rvlsfsep, rvlsfmid, rvlsfglobal
    """

    free_wall = window[:,0]
    septum = window[:,1]
    apex = window[:,2]

    midpoint = (free_wall + septum) / 2

    # Distances
    length_fw = np.linalg.norm(free_wall - apex, axis=-1)
    length_septum = np.linalg.norm(septum - apex, axis=-1)
    length_mid = np.linalg.norm(midpoint - apex, axis=-1)

    #tv diameter
    diameter = np.linalg.norm(free_wall - septum, axis=-1)

    # calculation of RVEDA and RVESA. It can be done by using the triangle method or the spline method
    if method == 'triangle': # triangle method
        
        semiperimeter = (length_fw + length_septum + diameter) / 2
        area = np.sqrt(
            semiperimeter
            * (semiperimeter - length_fw)
            * (semiperimeter - length_septum)
            * (semiperimeter - diameter)
        )

        area = remove_outliers_iqr(area)
        diast_area = area.max()
        syst_area = area.min()

    elif method == "spline": # calculate the RV area as the area inside the 3rd degree spline that interpolates the three points

        # diastole area method
        area_diastole = []
        for fw, ap, sp in zip(free_wall, apex, septum):
            # Order points (close loop)
            pts = np.vstack([fw, ap, sp, fw])
            t = np.arange(len(pts))

            # Parametric spline
            tck, u = interpolate.splprep([pts[:,0], pts[:,1]], s=0, per=True, k=3)
            u_new = np.linspace(0, 1, 6)
            x_new, y_new = interpolate.splev(u_new, tck)

            # Shoelace formula
            poly = np.vstack([x_new, y_new]).T

            area_i = 0.5*np.abs(np.dot(poly[:,0], np.roll(poly[:,1], -1)) -
                                np.dot(poly[:,1], np.roll(poly[:,0], -1)))
            area_diastole.append(area_i)

        area_diastole = np.array(area_diastole)
        # remove clear outliers
        area_diastole = remove_outliers_iqr(area_diastole)
        # take the maximum value across the heartbeat
        diast_area = area_diastole.max()

        # systole spline method
        area_systole = []
        for fw, ap, sp in zip(free_wall, apex, septum):
            # Order points (close loop)
            pts = np.vstack([fw, ap, sp, fw])
            t = np.arange(len(pts))

            # Parametric spline
            tck, u = interpolate.splprep([pts[:,0], pts[:,1]], s=0, per=True, k=3)
            u_new = np.linspace(0, 1, 5)
            x_new, y_new = interpolate.splev(u_new, tck)

            # Shoelace formula
            poly = np.vstack([x_new, y_new]).T

            area_i = 0.5*np.abs(np.dot(poly[:,0], np.roll(poly[:,1], -1)) -
                                np.dot(poly[:,1], np.roll(poly[:,0], -1)))
            area_systole.append(area_i)
        
        #same type of operations as for the diastolic method above
        area_systole = np.array(area_systole)
        area_systole = remove_outliers_iqr(area_systole)
        syst_area = area_systole.min()

    else: # errors
        raise ValueError("method must be 'triangle' or 'spline'")

    #calculation of rvfac at each heartbeat
    rvfac = (diast_area - syst_area) / diast_area * 100

    #calculation of lengths. 
    # I put them here and not above because they are needed as a whole (whithout removing outliers) to calculate the area with the triangle method  
    #fw
    length_fw = remove_outliers_iqr(length_fw)
    rvldfw = length_fw.max()
    rvlsfw = length_fw.min()

    #septum length
    length_septum = remove_outliers_iqr(length_septum)
    rvldsep = length_septum.max()
    rvlssep = length_septum.min()

    #apex
    length_mid = remove_outliers_iqr(length_mid)
    rvldmid = length_mid.max()
    rvlsmid = length_mid.min()

    #calculation of rv longitudinal strains
    rvlsffw = (rvldfw - rvlsfw)/ rvldfw * 100
    rvlsfsep = (rvldsep - rvlssep)/ rvldsep * 100
    rvlsfmid = (rvldmid - rvlsmid)/ rvldmid * 100
    rvlsfglobal = ((rvldfw+rvldsep)-(rvlsfw+rvlssep))/(rvldfw+rvldsep) * 100

    # diameter related calculations
    diameter = remove_outliers_iqr(diameter)
    tadd = diameter.max()
    tasd = diameter.min()
        
    return (rvfac, 
            diast_area, 
            syst_area, 
            rvldfw, 
            rvldsep,
            rvlsfw, 
            rvlssep, 
            rvldmid, 
            rvlsmid, 
            tadd, 
            tasd,
            rvlsffw, 
            rvlsfsep, 
            rvlsfmid, 
            rvlsfglobal)


def tric_apex_distance_calculation_best(
        window_kalman,
        window_avg,
        window_both,
        window_unfiltered
        ):
    """
    Calculate the distance from the apex to the free wall and septum.
    Also calculates the diameter of the right ventricle, and the area of the region
    defined by those three points.
    
    Parameters
    ----------
    free_wall, septum, apex : ndarray (..., 2)
        Coordinates of the free wall, septum, and apex at each time point.
    method : str, optional
        'triangle' (Heron's formula) or 'spline' (area under spline through points).
    
    Returns
    -------
    rvfac, diast_area, syst_area, rvldfw, rvldsep, rvlsfw, rvlssep,
    rvldmid, rvlsmid, tadd, tasd, rvlsffw, rvlsfsep, rvlsfmid, rvlsfglobal
    """
    # extract the coordinates of all the points at disposal. Each index will need it's specific points for the calculation
    free_wall_kalman = window_kalman[:,0]
    free_wall_both = window_both[:,0]
    free_wall_avg = window_avg[:,0]
    free_wall_none = window_unfiltered[:,0]

    septum_kalman = window_kalman[:,1]
    septum_both = window_both[:,1]
    septum_avg = window_avg[:,1]
    septum_none = window_unfiltered[:,1]

    apex_kalman = window_kalman[:,2]
    apex_both = window_both[:,2]
    apex_avg = window_avg[:,2]
    apex_none = window_unfiltered[:,2]

    midpoint_kalman = (free_wall_kalman + septum_kalman) / 2
    midpoint_both = (free_wall_both + septum_both) / 2
    midpoint_avg = (free_wall_avg + septum_avg) / 2
    midpoint_none = (free_wall_none + septum_none) / 2

    ## Distances from apex (fw)
    # best diast method
    RV_length_fw_diast = np.linalg.norm(free_wall_kalman - apex_kalman, axis=-1)
    RV_length_fw_diast = remove_outliers_iqr(RV_length_fw_diast)
    # take the max out of the heartbeat
    rvldfw = RV_length_fw_diast.max()  
    # best syst method
    RV_length_fw_syst = np.linalg.norm(free_wall_avg - apex_avg, axis=-1)
    RV_length_fw_syst = remove_outliers_iqr(RV_length_fw_syst)
    # take the min out of the heartbeat
    rvlsfw = RV_length_fw_syst.min() 

    # syst method (only for rvlsffw_calculation)
    RV_length_fw_syst_calc = np.linalg.norm(free_wall_none - apex_none, axis=-1)
    RV_length_fw_syst_calc = remove_outliers_iqr(RV_length_fw_syst_calc)
    # take the min out of the heartbeat
    rvlsfw_calc = RV_length_fw_syst_calc.min() 

    # diast method (only for rvlsffw calculation)
    RV_length_fw_diast_calc = np.linalg.norm(free_wall_kalman - apex_kalman, axis=-1)
    RV_length_fw_diast_calc = remove_outliers_iqr(RV_length_fw_diast_calc)
    # take the max out of the heartbeat
    rvldfw_calc = RV_length_fw_diast_calc.max()

    ## distances from apex (septum)
    #best_diast method
    RV_length_sep_diast= np.linalg.norm(septum_kalman - apex_kalman, axis=-1)
    RV_length_sep_diast = remove_outliers_iqr(RV_length_sep_diast)
    rvldsep = RV_length_sep_diast.max()
    #best syst method
    RV_length_sep_syst = np.linalg.norm(septum_avg - apex_avg, axis=-1)
    RV_length_sep_syst = remove_outliers_iqr(RV_length_sep_syst)
    rvlssep = RV_length_sep_syst.min()

    ## distances from apex (TV midpoint)
    #best_diast method
    RV_length_mid_diast= np.linalg.norm(midpoint_both - apex_both, axis=-1)
    RV_length_mid_diast = remove_outliers_iqr(RV_length_mid_diast)
    rvldmid = RV_length_mid_diast.max()
    #best syst method
    RV_length_mid_syst = np.linalg.norm(midpoint_none - apex_none, axis=-1)
    RV_length_mid_syst = remove_outliers_iqr(RV_length_mid_syst)
    rvlsmid = RV_length_mid_syst.min()

    ## distances from apex (TV midpoint) but only to calculate the rvlsfmid
    #best_diast method
    RV_length_mid_diast_calc= np.linalg.norm(midpoint_avg - apex_avg, axis=-1)
    RV_length_mid_diast_calc = remove_outliers_iqr(RV_length_mid_diast_calc)
    rvldmid_calc = RV_length_mid_diast_calc.max()
    #best syst method
    RV_length_mid_syst_calc = np.linalg.norm(midpoint_avg - apex_avg, axis=-1)
    RV_length_mid_syst_calc = remove_outliers_iqr(RV_length_mid_syst_calc)
    rvlsmid_calc = RV_length_mid_syst_calc.min()


    # diameter related indexes
    diameter = np.linalg.norm(free_wall_both - septum_both, axis=-1)
    diameter = remove_outliers_iqr(diameter)
    tadd = diameter.max()
    tasd = diameter.min()



    area_diastole = []
    for fw, ap, sp in zip(free_wall_kalman, apex_kalman, septum_kalman): #loop for diastolic area calculation
        # Order points (close loop)
        pts = np.vstack([fw, ap, sp, fw])
        t = np.arange(len(pts))

        # Parametric spline
        tck, u = interpolate.splprep([pts[:,0], pts[:,1]], s=0, per=True, k=3)
        u_new = np.linspace(0, 1, 6)
        x_new, y_new = interpolate.splev(u_new, tck)

        # Shoelace formula
        poly = np.vstack([x_new, y_new]).T

        area_i = 0.5*np.abs(np.dot(poly[:,0], np.roll(poly[:,1], -1)) -
                            np.dot(poly[:,1], np.roll(poly[:,0], -1)))
        area_diastole.append(area_i)
    area_diastole = np.array(area_diastole)
    #remove values that are outside the distribution
    area_diastole = remove_outliers_iqr(area_diastole)
    # take the max across the heartbeat
    diast_area = area_diastole.max()



    area_diastole_rvfac = [] # for the calculation of rvfac, since I'm underestimating the systolic area, i prefer to underestimate also the diastolic area
    for fw, ap, sp in zip(free_wall_both, apex_both, septum_both): #loop for diastolic area calculation
        # Order points (close loop)
        pts = np.vstack([fw, ap, sp, fw])
        t = np.arange(len(pts))

        # Parametric spline
        tck, u = interpolate.splprep([pts[:,0], pts[:,1]], s=0, per=True, k=3)
        u_new = np.linspace(0, 1, 9) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! only thing that changes is that i use 9 points instead of 6 for the interpolation
        x_new, y_new = interpolate.splev(u_new, tck)

        # Shoelace formula
        poly = np.vstack([x_new, y_new]).T

        area_i = 0.5*np.abs(np.dot(poly[:,0], np.roll(poly[:,1], -1)) -
                            np.dot(poly[:,1], np.roll(poly[:,0], -1)))
        area_diastole_rvfac.append(area_i)
    area_diastole_rvfac = np.array(area_diastole_rvfac)
    diast_area_rvfac = remove_outliers_iqr(area_diastole_rvfac)
    diast_area_rvfac = diast_area_rvfac.max()
    

    area_systole = []
    for fw, ap, sp in zip(free_wall_none, apex_none, septum_none): # loop for systole area calculation
        # Order points (close loop)
        pts = np.vstack([fw, ap, sp, fw])
        t = np.arange(len(pts))

        # Parametric spline
        tck, u = interpolate.splprep([pts[:,0], pts[:,1]], s=0, per=True, k=3)
        u_new = np.linspace(0, 1, 5)
        x_new, y_new = interpolate.splev(u_new, tck)

        # Shoelace formula
        poly = np.vstack([x_new, y_new]).T

        area_i = 0.5*np.abs(np.dot(poly[:,0], np.roll(poly[:,1], -1)) -
                            np.dot(poly[:,1], np.roll(poly[:,0], -1)))
        area_systole.append(area_i)
    area_systole = np.array(area_systole)
    area_systole = remove_outliers_iqr(area_systole)
    syst_area = area_systole.min()

    area_systole_rvfac = []
    for fw, ap, sp in zip(free_wall_avg, apex_avg, septum_avg): # loop for systole area calculation
        # Order points (close loop)
        pts = np.vstack([fw, ap, sp, fw])
        t = np.arange(len(pts))

        # Parametric spline
        tck, u = interpolate.splprep([pts[:,0], pts[:,1]], s=0, per=True, k=3)
        u_new = np.linspace(0, 1, 5)
        x_new, y_new = interpolate.splev(u_new, tck)

        # Shoelace formula
        poly = np.vstack([x_new, y_new]).T

        area_i = 0.5*np.abs(np.dot(poly[:,0], np.roll(poly[:,1], -1)) -
                            np.dot(poly[:,1], np.roll(poly[:,0], -1)))
        area_systole_rvfac.append(area_i)
    area_systole_rvfac = np.array(area_systole_rvfac)
    area_systole_rvfac = remove_outliers_iqr(area_systole_rvfac)
    syst_area_rvfac = area_systole_rvfac.min()

        
    rvfac = (diast_area_rvfac - syst_area_rvfac) / diast_area_rvfac * 100


    rvlsffw = (rvldfw - rvlsfw)/ rvldfw * 100
    rvlsfsep = (rvldsep - rvlssep)/ rvldsep * 100
    rvlsfmid = (rvldmid_calc - rvlsmid_calc)/ rvldmid_calc * 100


    # global calculation (using dummy indexes that are not reported in the results (avg filter ony))
    # diast method (only for rvlsffw calculation)
    RV_length_fw_diast_calc_global = np.linalg.norm(free_wall_avg - apex_avg, axis=-1)
    RV_length_fw_diast_calc_global = remove_outliers_iqr(RV_length_fw_diast_calc_global)
    # take the max out of the heartbeat
    rvldfw_calc_global = RV_length_fw_diast_calc_global.max()

    #best_diast method
    RV_length_sep_diast_global= np.linalg.norm(septum_avg - apex_avg, axis=-1)
    RV_length_sep_diast_global = remove_outliers_iqr(RV_length_sep_diast_global)
    rvldsep_global = RV_length_sep_diast_global.max()

    # the other 2 indices that are needed for the calculation are already using the avg filter, so I use them directly
    rvlsfglobal = ((rvldfw_calc_global+rvldsep_global)-(rvlsfw+rvlssep))/(rvldfw_calc_global+rvldsep_global) * 100

    return (rvfac, 
            diast_area, 
            syst_area, 
            rvldfw, 
            rvldsep,
            rvlsfw, 
            rvlssep, 
            rvldmid, 
            rvlsmid, 
            tadd, 
            tasd,
            rvlsffw, 
            rvlsfsep, 
            rvlsfmid, 
            rvlsfglobal,
            rvldfw_calc,
            rvlsfw_calc)
                

def tapse_calculation(
    window,
    direction = None,
    tapse_calc = 'distance',
):
    """
    Calculate TAPSE (Tricuspid Annular Plane Systolic Excursion) from the coordinates of the septum and free wall.
    The coordinates should be in the format (x, y) and the direction should be a unit vector.
    The pixelsize is a list containing the pixel size in mm for each dimension.

    based on the parameter "tapse_calc", the function will calculate the tapse in two different ways:
    - 'window': np.ndarray (..., 3, 2) containing hte coordinates of fw, septum and apex
    - 'distance': just calculates the maximum distance between the points in the septum and free wall for the time you provide
    - 'projection': projects the points in the direction of the vector and calculates the distance between the maximum and minimum projection for both septum and free wall, then averages them
    """

    # select the coordinates from the window
    coordinates_septum = window[:, 1]
    coordinates_fw = window[:,0]

    if tapse_calc == 'projection' and direction is None:
        raise ValueError("If tapse_calc is 'projection', direction must be provided")
    
    elif tapse_calc == 'distance':
        diff_septum = coordinates_septum[:, np.newaxis, :] - coordinates_septum[np.newaxis, :, :]  # (n, n, 2)
        dist_septum = np.linalg.norm(diff_septum, axis=-1)  # (n, n)
        tapse_septum = dist_septum.max()

        diff_fw = coordinates_fw[:, np.newaxis, :] - coordinates_fw[np.newaxis, :, :]  # (n, n, 2)
        dist_fw = np.linalg.norm(diff_fw, axis=-1)  # (n, n)
        tapse_fw = dist_fw.max()

    elif tapse_calc == 'projection':
        projection_septum = coordinates_septum @ direction
        projection_fw = coordinates_fw @ direction
        tapse_septum = projection_septum.max() - projection_septum.min()
        tapse_fw = projection_fw.max() - projection_fw.min()

    return tapse_septum, tapse_fw


def tapse_calculation_best(
    window_kalman,
    window_avg,
    window_both,
    window_unfiltered
):
    """
    Calculate TAPSE (Tricuspid Annular Plane Systolic Excursion) from the coordinates of the septum and free wall.
    The coordinates should be in the format (x, y) and the direction should be a unit vector.
    The pixelsize is a list containing the pixel size in mm for each dimension.

    based on the parameter "tapse_calc", the function will calculate the tapse in two different ways:
    - 'distance': just calculates the maximum distance between the points in the septum and free wall for the time you provide
    - 'projection': projects the points in the direction of the vector and calculates the distance between the maximum and minimum projection for both septum and free wall, then averages them
    """
    # coordinates to calculate tapsefw (best = both)
    coordinates_fw = window_both[:, 0]

    # coordinates to calculate tapsesep (best = kalman)
    coordinates_septum = window_kalman[:,1]

    # distance_based calculation
    diff_septum = coordinates_septum[:, np.newaxis, :] - coordinates_septum[np.newaxis, :, :]  # forma (n, n, 2)
    dist_septum = np.linalg.norm(diff_septum, axis=-1)  # forma (n, n)
    tapse_septum = dist_septum.max()

    diff_fw = coordinates_fw[:, np.newaxis, :] - coordinates_fw[np.newaxis, :, :]  # forma (n, n, 2)
    dist_fw = np.linalg.norm(diff_fw, axis=-1)  # forma (n, n)
    tapse_fw = dist_fw.max()

    return tapse_septum, tapse_fw

def find_parallel_direction(points):
    """
    Find the direction of the vector that is parallel to the array.
    The array should be a 2D numpy array with shape (n, 2).
    """
    if len(points) < 2:
        raise ValueError("Array must contain at least two points.")
    
    # Fit a linear polynomial: returns [intercept, slope]
    b, m = poly.polyfit(points[0], points[1], 1)

    # Create direction vector using the slope
    v = np.array([1, m])  # dx = 1, dy = m

    # Normalize to get the versor
    versor = v / np.linalg.norm(v)
    
    return versor

def find_es(window):
    '''
    function that finds the es and ed frames from a window of predictions spanning from one r peak to the next one
    args: 
    window: np.ndarray with shape (n, 3, 2) containing the coordinates of the 3 landmarks for each of the frames in an heartbeat

    returns:
    es: index of the frame with the minimum rv length (End Systole). The index is relative to the beginning of the window. It can be also calcu lated from 
    the maximum projected distance from the fw ed point of the fw es point. 
    '''
    fw = window[:,0]
    distances_from_ed = np.linalg.norm(fw - fw[0], axis = 1)
    es = distances_from_ed.argmax()

    return es

def find_ed_es_from_distance(window):
    '''
    function that calculates the es and ed frames on the whole acquisition. It uses the filtered coordinates to calculate the ta midpoint to apex distance.
    the local minima are es frames, the local maxima are the ed frames. Returns the indexes of these frames.
    '''
    fw = window[:,0]
    sep = window[:,1]
    apex = window[:,2]

    midpoint = (fw + sep) / 2

    rv_length = np.linalg.norm(midpoint - apex, axis=-1)

    ed = []
    es = []
    for i in range(1, len(rv_length)-1):
        if rv_length[i] < rv_length[i-1] and rv_length[i] < rv_length[i+1]:
            es.append(i)
        if rv_length[i] > rv_length[i-1] and rv_length[i] > rv_length[i+1]:
            ed.append(i)

    return ed, es

class RVCalculator:
    def __init__(self, ed_frame, es_frame, method="triangle"):
        self.ed_free_wall = ed_frame[0]
        self.ed_septum = ed_frame[1]
        self.ed_apex = ed_frame[2]

        self.es_free_wall = es_frame[0]
        self.es_septum = es_frame[1]
        self.es_apex = es_frame[2]

        self.method = method
        self._compute()

    def _compute(self):
        ed_mid = (self.ed_free_wall + self.ed_septum) / 2
        es_mid = (self.es_free_wall + self.es_septum) / 2

        self.tapse_fw = np.linalg.norm(self.ed_free_wall - self.es_free_wall)
        self.tapse_sep = np.linalg.norm(self.ed_septum - self.es_septum)

        self.ed_len_fw = np.linalg.norm(self.ed_free_wall - self.ed_apex, axis=-1)
        self.ed_len_sep = np.linalg.norm(self.ed_septum - self.ed_apex, axis=-1)
        self.ed_len_mid = np.linalg.norm(ed_mid - self.ed_apex, axis=-1)

        self.es_len_fw = np.linalg.norm(self.es_free_wall - self.es_apex, axis=-1)
        self.es_len_sep = np.linalg.norm(self.es_septum - self.es_apex, axis=-1)
        self.es_len_mid = np.linalg.norm(es_mid - self.es_apex, axis=-1)

        self.ed_diam = np.linalg.norm(self.ed_free_wall - self.ed_septum, axis=-1)
        self.es_diam = np.linalg.norm(self.es_free_wall - self.es_septum, axis=-1)

        if self.method == "triangle":
            ed_s = (self.ed_len_fw + self.ed_len_sep + self.ed_diam) / 2
            self.ed_area = np.sqrt(
                ed_s
                * (ed_s - self.ed_len_fw)
                * (ed_s - self.ed_len_sep)
                * (ed_s - self.ed_diam)
            )

            es_s = (self.es_len_fw + self.es_len_sep + self.es_diam) / 2
            self.es_area = np.sqrt(
                es_s
                * (es_s - self.es_len_fw)
                * (es_s - self.es_len_sep)
                * (es_s - self.es_diam)
            )

        elif self.method == "spline":
            self.ed_area = self._spline_area(self.ed_free_wall, self.ed_apex, self.ed_septum, n_points= 6)
            self.es_area = self._spline_area(self.es_free_wall, self.es_apex, self.es_septum, n_points= 5)

        else:
            raise ValueError("method must be 'triangle' or 'spline'")

        self.rvfac = (self.ed_area - self.es_area) / self.ed_area * 100

        self.strain_fw = (self.ed_len_fw - self.es_len_fw) / self.ed_len_fw * 100
        self.strain_sep = (self.ed_len_sep - self.es_len_sep) / self.ed_len_sep * 100
        self.strain_mid = (self.ed_len_mid - self.es_len_mid) / self.ed_len_mid * 100
        self.strain_global = (
            (self.ed_len_fw + self.ed_len_sep)
            - (self.es_len_fw + self.es_len_sep)
        ) / (self.ed_len_fw + self.ed_len_sep) * 100

    @staticmethod
    def _spline_area(p1, apex, p2, n_points):
        pts = np.vstack([p1, apex, p2, p1])
        tck, u = interpolate.splprep([pts[:, 0], pts[:, 1]], s=0, per=True, k=3)
        u_new = np.linspace(0, 1, n_points)
        x_new, y_new = interpolate.splev(u_new, tck)
        poly = np.column_stack((x_new, y_new))
        return 0.5 * np.abs(
            np.dot(poly[:, 0], np.roll(poly[:, 1], -1))
            - np.dot(poly[:, 1], np.roll(poly[:, 0], -1))
        )


class RVCalculatorBest:
    def __init__(self, window_none, window_kalman, window_avg, window_both, ed, es):
        #extract ed and es frames for each of the types of filtering
        none_ed = window_none[ed]
        none_es = window_none[es]

        kalman_ed = window_kalman[ed]
        kalman_es = window_kalman[es]

        avg_ed = window_avg[ed]
        avg_es = window_avg[es]

        both_ed = window_both[ed]
        both_es = window_both[es]

        # none
        self.none_ed_free_wall = none_ed[0]
        self.none_ed_septum = none_ed[1]
        self.none_ed_apex = none_ed[2]

        self.none_es_free_wall = none_es[0]
        self.none_es_septum = none_es[1]
        self.none_es_apex = none_es[2]

        self.none_ed_mid = (self.none_ed_free_wall + self.none_ed_septum) / 2
        self.none_es_mid = (self.none_es_free_wall + self.none_es_septum) / 2

        # kalman
        self.kalman_ed_free_wall = kalman_ed[0]
        self.kalman_ed_septum = kalman_ed[1]
        self.kalman_ed_apex = kalman_ed[2]

        self.kalman_es_free_wall = kalman_es[0]
        self.kalman_es_septum = kalman_es[1]
        self.kalman_es_apex = kalman_es[2]

        self.kalman_ed_mid = (self.kalman_ed_free_wall + self.kalman_ed_septum) / 2
        self.kalman_es_mid = (self.kalman_es_free_wall + self.kalman_es_septum) / 2

        # avg
        self.avg_ed_free_wall = avg_ed[0]
        self.avg_ed_septum = avg_ed[1]
        self.avg_ed_apex = avg_ed[2]

        self.avg_es_free_wall = avg_es[0]
        self.avg_es_septum = avg_es[1]
        self.avg_es_apex = avg_es[2]

        self.avg_ed_mid = (self.avg_ed_free_wall + self.avg_ed_septum) / 2
        self.avg_es_mid = (self.avg_es_free_wall + self.avg_es_septum) / 2

        # both
        self.both_ed_free_wall = both_ed[0]
        self.both_ed_septum = both_ed[1]
        self.both_ed_apex = both_ed[2]

        self.both_es_free_wall = both_es[0]
        self.both_es_septum = both_es[1]
        self.both_es_apex = both_es[2]

        self.both_ed_mid = (self.both_ed_free_wall + self.both_ed_septum) / 2
        self.both_es_mid = (self.both_es_free_wall + self.both_es_septum) / 2

        self._compute()

    def _compute(self):
        self.tapse_fw = np.linalg.norm(self.kalman_ed_free_wall - self.kalman_es_free_wall)
        self.tapse_sep = np.linalg.norm(self.both_ed_septum - self.both_es_septum)

        self.ed_len_fw = np.linalg.norm(self.none_ed_free_wall - self.none_ed_apex, axis=-1)
        self.ed_len_sep = np.linalg.norm(self.none_ed_septum - self.none_ed_apex, axis=-1)
        self.ed_len_mid = np.linalg.norm(self.none_ed_mid - self.none_ed_apex, axis=-1)

        self.es_len_fw = np.linalg.norm(self.kalman_es_free_wall - self.kalman_es_apex, axis=-1)
        self.es_len_sep = np.linalg.norm(self.both_es_septum - self.both_es_apex, axis=-1)
        self.es_len_mid = np.linalg.norm(self.both_es_mid - self.both_es_apex, axis=-1)

        self.ed_diam = np.linalg.norm(self.kalman_ed_free_wall - self.kalman_ed_septum, axis=-1)
        self.es_diam = np.linalg.norm(self.none_es_free_wall - self.none_es_septum, axis=-1)

        self.ed_area = self._spline_area(self.both_ed_free_wall, self.both_ed_apex, self.both_ed_septum, n_points= 6)
        self.ed_area_rvfac = self._spline_area(self.both_ed_free_wall, self.both_ed_apex, self.both_ed_septum, n_points= 11)
        self.es_area = self._spline_area(self.kalman_es_free_wall, self.kalman_es_apex, self.kalman_es_septum, n_points= 5)
        self.es_area_rvfac = self._spline_area(self.kalman_es_free_wall, self.kalman_es_apex, self.kalman_es_septum, n_points= 5)

        self.rvfac = (self.ed_area_rvfac - self.es_area_rvfac) / self.ed_area_rvfac * 100


        #indexes to calculate rv strains
        self.kalman_ed_len_fw = np.linalg.norm(self.kalman_ed_free_wall - self.kalman_ed_apex, axis=-1)
        self.kalman_es_len_fw = np.linalg.norm(self.kalman_es_free_wall - self.kalman_es_apex, axis=-1)
        self.kalman_ed_len_sep = np.linalg.norm(self.kalman_ed_septum - self.kalman_ed_apex, axis=-1)
        self.kalman_es_len_sep = np.linalg.norm(self.kalman_es_septum - self.kalman_es_apex, axis=-1)
        self.kalman_ed_len_mid = np.linalg.norm(self.kalman_ed_mid - self.kalman_ed_apex, axis=-1)
        self.kalman_es_len_mid = np.linalg.norm(self.kalman_es_mid - self.kalman_es_apex, axis=-1)

        self.strain_fw = (self.kalman_ed_len_fw - self.kalman_es_len_fw) / self.kalman_ed_len_fw * 100
        self.strain_sep = (self.kalman_ed_len_sep - self.kalman_es_len_sep) / self.kalman_ed_len_sep * 100
        self.strain_mid = (self.kalman_ed_len_mid - self.kalman_es_len_mid) / self.kalman_ed_len_mid * 100
        self.strain_global = (
            (self.kalman_ed_len_fw + self.kalman_ed_len_sep)
            - (self.kalman_es_len_fw + self.kalman_es_len_sep)
        ) / (self.kalman_ed_len_fw + self.kalman_ed_len_sep) * 100

    @staticmethod
    def _spline_area(p1, apex, p2, n_points):
        pts = np.vstack([p1, apex, p2, p1])
        tck, u = interpolate.splprep([pts[:, 0], pts[:, 1]], s=0, per=True, k=3)
        u_new = np.linspace(0, 1, n_points)
        x_new, y_new = interpolate.splev(u_new, tck)
        poly = np.column_stack((x_new, y_new))
        return 0.5 * np.abs(
            np.dot(poly[:, 0], np.roll(poly[:, 1], -1))
            - np.dot(poly[:, 1], np.roll(poly[:, 0], -1))
        )