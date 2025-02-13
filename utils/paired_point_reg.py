import numpy as np

def paired_point_registration(source, target):
    """
    Perform paired point registration between two coordinate systems using a similarity transform.
    This function computes the optimal rotation, uniform scale, and translation that align the source
    points to the target points.
    
    Args:
        source: (N, d) numpy array of source points.
        target: (N, d) numpy array of target points.
        
    Returns:
        transformation: (d+1, d+1) homogeneous transformation matrix, where the upper-left d x d block 
                        is the rotation (with scale) and the last column is the translation.
    """
    # Convert inputs to numpy arrays (in case they aren't)
    source = np.asarray(source)
    target = np.asarray(target)
    
    # For 2D points, try to use OpenCV's optimized function if available
    if source.shape[1] == 2:
        try:
            import cv2
            # estimateAffinePartial2D returns a 2x3 matrix (rotation, scale, translation)
            matrix, _ = cv2.estimateAffinePartial2D(source, target)
            # Convert to a 3x3 homogeneous transformation matrix
            transformation = np.eye(3)
            transformation[:2, :] = matrix
            return transformation
        except ImportError:
            # If OpenCV is not available, fall back to the custom implementation
            pass

    # -----------------------------
    # Custom implementation using the Umeyama method:
    # -----------------------------
    # 1. Compute centroids of both point sets
    src_mean = np.mean(source, axis=0)
    tgt_mean = np.mean(target, axis=0)

    # 2. Subtract centroids (demean the points)
    src_demean = source - src_mean
    tgt_demean = target - tgt_mean

    # 3. Compute the variance of the source points
    var_src = np.mean(np.sum(src_demean**2, axis=1))

    # 4. Compute the covariance matrix between target and source
    cov_matrix = np.dot(tgt_demean.T, src_demean) / source.shape[0]

    # 5. Perform Singular Value Decomposition (SVD)
    U, D, Vt = np.linalg.svd(cov_matrix)
    V = Vt.T

    # 6. Detect and correct for reflection (ensure a proper rotation)
    S = np.eye(source.shape[1])
    if np.linalg.det(np.dot(U, V.T)) < 0:
        S[-1, -1] = -1

    # 7. Compute the rotation matrix
    R = np.dot(np.dot(U, S), V.T)

    # 8. Compute the uniform scale factor using the trace formula
    scale = np.trace(np.dot(np.diag(D), S)) / var_src

    # 9. Compute the translation vector
    t = tgt_mean - scale * np.dot(R, src_mean)

    # 10. Assemble the homogeneous transformation matrix
    d = source.shape[1]
    transformation = np.eye(d + 1)
    transformation[:d, :d] = scale * R
    transformation[:d, d] = t

    return transformation

# Example usage:
if __name__ == "__main__":
    # Define two sets of 2D points (source and target)
    source_points = np.array([[0, 0],
                              [1, 0],
                              [0, 1]])
    # Apply a rotation of 30 degrees, scale by 2, and translate by (5, 3)
    theta = np.deg2rad(30)
    R_true = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta),  np.cos(theta)]])
    scale_true = 2.0
    t_true = np.array([5, 3])
    target_points = (scale_true * np.dot(source_points, R_true.T)) + t_true

    # Compute the transformation
    T = paired_point_registration(source_points, target_points)
    print("Estimated Transformation Matrix:\n", T)
