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

def tric_apex_distance_calculation(free_wall_filtered,
                                   septum_filtered, 
                                   apex_filtered,
                                   free_wall,
                                   septum, 
                                   apex, 
                                   method="triangle", 
                                   filter = False,
                                   best_combination = False
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

    if not best_combination:
        if not filter:
            midpoint = (free_wall + septum) / 2

            # Distances from apex
            dist_0 = np.linalg.norm(free_wall - apex, axis=-1)
            dist_1 = np.linalg.norm(septum - apex, axis=-1)
            dist_2 = np.linalg.norm(midpoint - apex, axis=-1)

            diameter = np.linalg.norm(free_wall - septum, axis=-1)

            if method == "triangle": # calculate the RV area as the area inside the triangle
                semiperimeter = (dist_0 + dist_1 + diameter) / 2
                area = np.sqrt(
                    semiperimeter
                    * (semiperimeter - dist_0)
                    * (semiperimeter - dist_1)
                    * (semiperimeter - diameter)
                )

                area = remove_outliers_iqr(area)
                diast_area = area.max()
                syst_area = area.min()

            elif method == "spline": # calculate the RV area as the area inside the spline through the three points
                area_diastole = []
                area_systole = []
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

                    # plt.figure()
                    # plt.plot(poly[:,0], poly[:,1], 'o-', label="Spline polygon")
                    # plt.plot([fw[0], ap[0], sp[0], fw[0]],
                    #         [fw[1], ap[1], sp[1], fw[1]], 'r--', label="Triangle")
                    # plt.scatter([fw[0], ap[0], sp[0]], [fw[1], ap[1], sp[1]], c='k', zorder=5)
                    # plt.axis('equal')
                    # plt.legend()
                    # plt.show()

                    area_i = 0.5*np.abs(np.dot(poly[:,0], np.roll(poly[:,1], -1)) -
                                        np.dot(poly[:,1], np.roll(poly[:,0], -1)))
                    area_diastole.append(area_i)
                area_diastole = np.array(area_diastole)
                
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

                    # plt.figure()
                    # plt.plot(poly[:,0], poly[:,1], 'o-', label="Spline polygon")
                    # plt.plot([fw[0], ap[0], sp[0], fw[0]],
                    #         [fw[1], ap[1], sp[1], fw[1]], 'r--', label="Triangle")
                    # plt.scatter([fw[0], ap[0], sp[0]], [fw[1], ap[1], sp[1]], c='k', zorder=5)
                    # plt.axis('equal')
                    # plt.legend()
                    # plt.show()

                    area_i = 0.5*np.abs(np.dot(poly[:,0], np.roll(poly[:,1], -1)) -
                                        np.dot(poly[:,1], np.roll(poly[:,0], -1)))
                    area_systole.append(area_i)
                area_systole = np.array(area_systole)

                area_diastole = remove_outliers_iqr(area_diastole)
                area_systole = remove_outliers_iqr(area_systole)
                diast_area = area_diastole.max()
                syst_area = area_systole.min()
            else:
                raise ValueError("method must be 'triangle' or 'spline'")

            rvfac = (diast_area - syst_area) / diast_area * 100

            dist_0 = remove_outliers_iqr(dist_0)
            dist_1 = remove_outliers_iqr(dist_1)
            dist_2 = remove_outliers_iqr(dist_2)
            rvldfw = dist_0.max()
            rvldsep = dist_1.max()
            rvlsfw = dist_0.min()
            rvlssep = dist_1.min()
            rvldmid = dist_2.max()
            rvlsmid = dist_2.min()

            rvlsffw = (rvldfw - rvlsfw)/ rvldfw * 100
            rvlsfsep = (rvldsep - rvlssep)/ rvldsep * 100
            rvlsfmid = (rvldmid - rvlsmid)/ rvldmid * 100
            rvlsfglobal = ((rvldfw+rvldsep)-(rvlsfw+rvlssep))/(rvldfw+rvldsep) * 100

            diameter = remove_outliers_iqr(diameter)

            tadd = diameter.max()
            tasd = diameter.min()


        else: # apply filter

            midpoint = (free_wall_filtered + septum_filtered) / 2

            # Distances from apex
            dist_0 = np.linalg.norm(free_wall_filtered - apex_filtered, axis=-1)
            dist_1 = np.linalg.norm(septum_filtered - apex_filtered, axis=-1)
            dist_2 = np.linalg.norm(midpoint - apex_filtered, axis=-1)

            diameter = np.linalg.norm(free_wall_filtered - septum_filtered, axis=-1)

            if method == "triangle": # calculate the RV area as the area inside the triangle
                semiperimeter = (dist_0 + dist_1 + diameter) / 2
                area = np.sqrt(
                    semiperimeter
                    * (semiperimeter - dist_0)
                    * (semiperimeter - dist_1)
                    * (semiperimeter - diameter)
                )

                area = remove_outliers_iqr(area)
                diast_area = area.max()
                syst_area = area.min()

            elif method == "spline": # calculate the RV area as the area inside the spline through the three points
                area_diastole = []
                area_systole = []
                for fw, ap, sp in zip(free_wall_filtered, apex_filtered, septum_filtered): #loop for diastolic area calculation
                    # Order points (close loop)
                    pts = np.vstack([fw, ap, sp, fw])
                    t = np.arange(len(pts))

                    # Parametric spline
                    tck, u = interpolate.splprep([pts[:,0], pts[:,1]], s=0, per=True, k=3)
                    u_new = np.linspace(0, 1, 6)
                    x_new, y_new = interpolate.splev(u_new, tck)

                    # Shoelace formula
                    poly = np.vstack([x_new, y_new]).T

                    # plt.figure()
                    # plt.plot(poly[:,0], poly[:,1], 'o-', label="Spline polygon")
                    # plt.plot([fw[0], ap[0], sp[0], fw[0]],
                    #         [fw[1], ap[1], sp[1], fw[1]], 'r--', label="Triangle")
                    # plt.scatter([fw[0], ap[0], sp[0]], [fw[1], ap[1], sp[1]], c='k', zorder=5)
                    # plt.axis('equal')
                    # plt.legend()
                    # plt.show()

                    area_i = 0.5*np.abs(np.dot(poly[:,0], np.roll(poly[:,1], -1)) -
                                        np.dot(poly[:,1], np.roll(poly[:,0], -1)))
                    area_diastole.append(area_i)
                area_diastole = np.array(area_diastole)
                
                for fw, ap, sp in zip(free_wall_filtered, apex_filtered, septum_filtered): # loop for systole area calculation
                    # Order points (close loop)
                    pts = np.vstack([fw, ap, sp, fw])
                    t = np.arange(len(pts))

                    # Parametric spline
                    tck, u = interpolate.splprep([pts[:,0], pts[:,1]], s=0, per=True, k=3)
                    u_new = np.linspace(0, 1, 5)
                    x_new, y_new = interpolate.splev(u_new, tck)

                    # Shoelace formula
                    poly = np.vstack([x_new, y_new]).T

                    # plt.figure()
                    # plt.plot(poly[:,0], poly[:,1], 'o-', label="Spline polygon")
                    # plt.plot([fw[0], ap[0], sp[0], fw[0]],
                    #         [fw[1], ap[1], sp[1], fw[1]], 'r--', label="Triangle")
                    # plt.scatter([fw[0], ap[0], sp[0]], [fw[1], ap[1], sp[1]], c='k', zorder=5)
                    # plt.axis('equal')
                    # plt.legend()
                    # plt.show()

                    area_i = 0.5*np.abs(np.dot(poly[:,0], np.roll(poly[:,1], -1)) -
                                        np.dot(poly[:,1], np.roll(poly[:,0], -1)))
                    area_systole.append(area_i)
                area_systole = np.array(area_systole)

                area_diastole = remove_outliers_iqr(area_diastole)
                area_systole = remove_outliers_iqr(area_systole)
                diast_area = area_diastole.max()
                syst_area = area_systole.min()
            else:
                raise ValueError("method must be 'triangle' or 'spline'")

            rvfac = (diast_area - syst_area) / diast_area * 100

            dist_0 = remove_outliers_iqr(dist_0)
            dist_1 = remove_outliers_iqr(dist_1)
            dist_2 = remove_outliers_iqr(dist_2)
            rvldfw = dist_0.max()
            rvldsep = dist_1.max()
            rvlsfw = dist_0.min()
            rvlssep = dist_1.min()
            rvldmid = dist_2.max()
            rvlsmid = dist_2.min()

            rvlsffw = (rvldfw - rvlsfw)/ rvldfw * 100
            rvlsfsep = (rvldsep - rvlssep)/ rvldsep * 100
            rvlsfmid = (rvldmid - rvlsmid)/ rvldmid * 100
            rvlsfglobal = ((rvldfw+rvldsep)-(rvlsfw+rvlssep))/(rvldfw+rvldsep) * 100

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
    window_kalman,
    window_avg,
    window_both,
    window_unfiltered,
    direction = None,
    tapse_calc = 'distance',
    filter = 'none',
):
    """
    Calculate TAPSE (Tricuspid Annular Plane Systolic Excursion) from the coordinates of the septum and free wall.
    The coordinates should be in the format (x, y) and the direction should be a unit vector.
    The pixelsize is a list containing the pixel size in mm for each dimension.

    based on the parameter "tapse_calc", the function will calculate the tapse in two different ways:
    - 'distance': just calculates the maximum distance between the points in the septum and free wall for the time you provide
    - 'projection': projects the points in the direction of the vector and calculates the distance between the maximum and minimum projection for both septum and free wall, then averages them
    """

    # select the coordinates from the passed windows
    if filter == 'both':
        coordinates_septum = window_both[:, 1]
        coordinates_fw = window_both[:,0]
    elif filter == 'none':
        coordinates_septum = window_unfiltered[:, 1]
        coordinates_fw = window_unfiltered[:,0]
    elif filter == 'kalman':
        coordinates_septum = window_kalman[:, 1]
        coordinates_fw = window_kalman[:,0]
    elif filter == 'avg':
        coordinates_septum = window_avg[:, 1]
        coordinates_fw = window_avg[:,0]

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