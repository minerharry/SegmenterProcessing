import numpy as np
from scipy.linalg import eigh


def fit_ellipse(indices, scale=None, xsize=None, ysize=None):
    # The following method determines the "mass density" of the ROI and fits
    # an ellipse to it. This is used to calculate the major and minor axes of
    # the ellipse, as well as its orientation. The orientation is calculated in
    # degrees counter-clockwise from the X axis.
    if xsize is None:
        xsize = np.shape(indices)[0]
    if ysize is None:
        ysize = np.shape(indices)[1]
    if scale is None:
        scale = [1.0, 1.0]
    if isinstance(scale, float):
        scale = [scale, scale]

    # Fake indices for testing purposes.
    if indices is None:
        xs = xsize // 4
        xf = xsize // 4 * 2
        ys = ysize // 4
        yf = ysize // 4 * 2
        array = np.zeros((xsize, ysize), dtype=np.uint8)
        array[xs:xf, ys:yf] = 255
        indices = np.transpose(np.where(array == 255))

    # Convert the indices to COL/ROW coordinates. Find min and max values
    cols, rows = np.transpose(indices)
    minX, maxX = np.min(cols), np.max(cols)
    minY, maxY = np.min(rows), np.max(rows)
    cols -= minX
    rows -= minY

    # Make an array large enough to hold the blob.
    arrayXSize = maxX - minX + 1
    arrayYSize = maxY - minY + 1
    array = np.zeros((arrayXSize, arrayYSize), dtype=np.uint8)
    array[cols, rows] = 255
    totalMass = np.sum(array)
    xcm = np.sum(np.sum(array, axis=1) * np.arange(arrayXSize) * scale[0]) / totalMass
    ycm = np.sum(np.sum(array, axis=0) * np.arange(arrayYSize) * scale[1]) / totalMass
    if center is not None:
        center[0] = xcm
        center[1] = ycm
    else:
        center = [xcm, ycm]

    # Obtain the position of every pixel in the image, with the origin
    # at the center of mass of the ROI.
    x = np.arange(arrayXSize) * scale[0]
    y = np.arange(arrayYSize) * scale[1]
    xx = np.outer(x, np.ones_like(y)) - xcm
    yy = np.outer(np.ones_like(x), y) - ycm
    npts = np.shape(indices)[0]

    # Calculate the mass distribution tensor.
    i11 = np.sum(yy[cols, rows] ** 2) / npts
    i22 = np.sum(xx[cols, rows] ** 2) / npts
    i12 = -np.sum(xx[cols, rows] * yy[cols, rows]) / npts
    tensor = np.array([[i11, i12], [i12, i22]])

    # Find the eigenvalues and eigenvectors of the tensor.
    evals,evecs = eigh(tensor)

    # The semi-major and semi-minor axes of the ellipse are obtained from the eigenvalues.
    semimajor = np.sqrt(evals[0]) * 2.0
    semiminor = np.sqrt(evals[1]) * 2.0

    # We want the actual axes lengths.
    major = semimajor * 2.0
    minor = semiminor * 2.0
    semiAxes = [semimajor, semiminor]
    axes = [major, minor]

    # The orientation of the ellipse is obtained from the first eigenvector.
    evec = evecs[:,0]

    # Degrees counter-clockwise from the X axis.
    orientation = np.arctan2(evec[1], evec[0])

    return ()

    