import cupy as cp
import numpy as np
import torch
from numpy.polynomial import polynomial as poly
import numpy as np
from scipy import interpolate

def tric_apex_distance_calculation(free_wall, septum, apex, method="triangle"):
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
    elif method == "spline": # calculate the RV area as the area inside the spline through the three points
        area = []
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
            area.append(area_i)
        area = np.array(area)
    else:
        raise ValueError("method must be 'triangle' or 'spline'")

    diast_area = area.max()
    syst_area = area.min()
    rvfac = (diast_area - syst_area) / diast_area * 100

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

    tadd = diameter.max()
    tasd = diameter.min()

    return (rvfac, diast_area, syst_area, rvldfw, rvldsep,
            rvlsfw, rvlssep, rvldmid, rvlsmid, tadd, tasd,
            rvlsffw, rvlsfsep, rvlsfmid, rvlsfglobal)

def tapse_calculation(
    coordinates_septum: np.ndarray,
    coordinates_fw: np.ndarray,
    direction: np.ndarray,
    tapse_calc = 'distance'
):
    """
    Calculate TAPSE (Tricuspid Annular Plane Systolic Excursion) from the coordinates of the septum and free wall.
    The coordinates should be in the format (x, y) and the direction should be a unit vector.
    The pixelsize is a list containing the pixel size in mm for each dimension.

    based on the parameter "tapse_calc", the function will calculate the tapse in two different ways:
    - 'distance': just calculates the maximum distance between the points in the septum and free wall for the time you provide
    - 'projection': projects the points in the direction of the vector and calculates the distance between the maximum and minimum projection for both septum and free wall, then averages them
    """

    if tapse_calc not in ['distance', 'projection']:
        raise ValueError("tapse_calc must be either 'distance' or 'projection'")
    elif tapse_calc == 'projection' and direction is None:
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