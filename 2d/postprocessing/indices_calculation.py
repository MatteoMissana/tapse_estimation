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


    else: # best combination from experiments
        midpoint = (free_wall_filtered + septum_filtered) / 2

        # Distances from apex
        dist_0 = np.linalg.norm(free_wall_filtered - apex_filtered, axis=-1)
        dist_1 = np.linalg.norm(septum_filtered - apex_filtered, axis=-1)
        dist_2 = np.linalg.norm(midpoint - apex_filtered, axis=-1)

        diameter = np.linalg.norm(free_wall_filtered - septum_filtered, axis=-1)

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

            area_i = 0.5*np.abs(np.dot(poly[:,0], np.roll(poly[:,1], -1)) -
                                np.dot(poly[:,1], np.roll(poly[:,0], -1)))
            area_diastole.append(area_i)
        area_diastole = np.array(area_diastole)
        
        for fw, ap, sp in zip(free_wall, apex, septum): # loop for systole area calculation
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

def tapse_calculation(
    coordinates_septum_filtered: np.ndarray,
    coordinates_fw_filtered: np.ndarray,
    coordinates_septum: np.ndarray,
    coordinates_fw: np.ndarray,
    direction: np.ndarray,
    tapse_calc = 'distance',
    filter = False,
    best_combination = False,
):
    """
    Calculate TAPSE (Tricuspid Annular Plane Systolic Excursion) from the coordinates of the septum and free wall.
    The coordinates should be in the format (x, y) and the direction should be a unit vector.
    The pixelsize is a list containing the pixel size in mm for each dimension.

    based on the parameter "tapse_calc", the function will calculate the tapse in two different ways:
    - 'distance': just calculates the maximum distance between the points in the septum and free wall for the time you provide
    - 'projection': projects the points in the direction of the vector and calculates the distance between the maximum and minimum projection for both septum and free wall, then averages them
    """
    if not best_combination:
        if not filter:
            if tapse_calc == 'projection' and direction is None:
                raise ValueError("If tapse_calc is 'projection', direction must be provided")
            
            elif tapse_calc == 'distance':
                diff_septum = coordinates_septum[:, np.newaxis, :] - coordinates_septum[np.newaxis, :, :]  # forma (n, n, 2)
                dist_septum = np.linalg.norm(diff_septum, axis=-1)  # forma (n, n)
                tapse_septum = dist_septum.max()

                diff_fw = coordinates_fw[:, np.newaxis, :] - coordinates_fw[np.newaxis, :, :]  # forma (n, n, 2)
                dist_fw = np.linalg.norm(diff_fw, axis=-1)  # forma (n, n)
                tapse_fw = dist_fw.max()
                tapse = (tapse_septum + tapse_fw) / 2

            elif tapse_calc == 'projection':
                projection_septum = coordinates_septum @ direction
                projection_fw = coordinates_fw @ direction
                tapse_septum = projection_septum.max() - projection_septum.min()
                tapse_fw = projection_fw.max() - projection_fw.min()
                tapse = (tapse_septum + tapse_fw) / 2

        else: # apply filter
            if tapse_calc == 'projection' and direction is None:
                raise ValueError("If tapse_calc is 'projection', direction must be provided")
            
            elif tapse_calc == 'distance':
                diff_septum = coordinates_septum_filtered[:, np.newaxis, :] - coordinates_septum_filtered[np.newaxis, :, :]  # forma (n, n, 2)
                dist_septum = np.linalg.norm(diff_septum, axis=-1)  # forma (n, n)
                tapse_septum = dist_septum.max()

                diff_fw = coordinates_fw_filtered[:, np.newaxis, :] - coordinates_fw_filtered[np.newaxis, :, :]  # forma (n, n, 2)
                dist_fw = np.linalg.norm(diff_fw, axis=-1)  # forma (n, n)
                tapse_fw = dist_fw.max()
                tapse = (tapse_septum + tapse_fw) / 2

            elif tapse_calc == 'projection':
                projection_septum = coordinates_septum_filtered @ direction
                projection_fw = coordinates_fw_filtered @ direction
                tapse_septum = projection_septum.max() - projection_septum.min()
                tapse_fw = projection_fw.max() - projection_fw.min()
                tapse = (tapse_septum + tapse_fw) / 2        

    else: # best combination from experiments

        diff_septum = coordinates_septum_filtered[:, np.newaxis, :] - coordinates_septum_filtered[np.newaxis, :, :]  # forma (n, n, 2)
        dist_septum = np.linalg.norm(diff_septum, axis=-1)  # forma (n, n)
        tapse_septum = dist_septum.max()

        diff_fw = coordinates_fw_filtered[:, np.newaxis, :] - coordinates_fw_filtered[np.newaxis, :, :]  # forma (n, n, 2)
        dist_fw = np.linalg.norm(diff_fw, axis=-1)  # forma (n, n)
        tapse_fw = dist_fw.max()
        tapse = (tapse_septum + tapse_fw) / 2

    return tapse_septum, tapse_fw, tapse

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