import numpy as np
from scipy.ndimage import median_filter
from scipy.ndimage import binary_dilation
from scipy.ndimage import uniform_filter1d
from scipy import interpolate


def remove_stripe_based_sorting(matindex, sinogram, size):
    """Algorithm 3 in the paper. To remove partial and full stripes.
    """
    sinogram = np.transpose(sinogram)
    matcomb = np.asarray(np.dstack((matindex, sinogram)))
    matsort = np.asarray(
        [row[row[:, 1].argsort()] for row in matcomb])
    matsort[:, :, 1] = median_filter(matsort[:, :, 1], (size, 1))
    matsortback = np.asarray(
        [row[row[:, 0].argsort()] for row in matsort])
    sino_corrected = matsortback[:, :, 1]
    return np.transpose(sino_corrected)

def detect_stripe(listdata, snr):
    """Algorithm 4 in the paper. To locate stripe positions.

    Parameters
    ----------
    listdata : 1D normalized array.
    snr : Ratio (>1.0) used to detect stripe locations.

    Returns
    -------
    listmask : 1D binary mask.
    """
    numdata = len(listdata)
    listsorted = np.sort(listdata)[::-1]
    xlist = np.arange(0, numdata, 1.0)
    ndrop = np.int16(0.25 * numdata)
    (_slope, _intercept) = np.polyfit(
        xlist[ndrop:-ndrop - 1], listsorted[ndrop:-ndrop - 1], 1)
    numt1 = _intercept + _slope * xlist[-1]
    noiselevel = np.abs(numt1 - _intercept)
    if noiselevel == 0.0:
        raise ValueError(
            "The method doesn't work on noise-free data. If you " \
            "apply the method on simulated data, please add" \
            " noise!")
    val1 = np.abs(listsorted[0] - _intercept) / noiselevel
    val2 = np.abs(listsorted[-1] - numt1) / noiselevel
    listmask = np.zeros_like(listdata)
    if val1 >= snr:
        upper_thresh = _intercept + noiselevel * snr * 0.5
        listmask[listdata > upper_thresh] = 1.0
    if val2 >= snr:
        lower_thresh = numt1 - noiselevel * snr * 0.5
        listmask[listdata <= lower_thresh] = 1.0
    return listmask

def remove_large_stripe(matindex, sinogram, snr, size):
    """Algorithm 5 in the paper. To remove large stripes.

    Parameters
    -----------
    sinogram : 2D array.
    snr : Ratio (>1.0) used to detect stripe locations.
    size : Window size of the median filter.

    Returns
    -------
    sinogram : stripe-removed sinogram.
    """
    badpixelratio = 0.05
    (nrow, ncol) = sinogram.shape
    ndrop = np.int16(badpixelratio * nrow)
    sinosorted = np.sort(sinogram, axis=0)
    sinosmoothed = median_filter(sinosorted, (1, size))
    list1 = np.mean(sinosorted[ndrop:nrow - ndrop], axis=0)
    list2 = np.mean(sinosmoothed[ndrop:nrow - ndrop], axis=0)
    listfact = np.divide(list1, list2,
                            out=np.ones_like(list1), where=list2 != 0)
    listmask = detect_stripe(listfact, snr)
    listmask = binary_dilation(listmask, iterations=1).astype(
        listmask.dtype)
    matfact = np.tile(listfact, (nrow, 1))
    sinogram = sinogram / matfact
    sinogram1 = np.transpose(sinogram)
    matcombine = np.asarray(np.dstack((matindex, sinogram1)))
    matsort = np.asarray(
        [row[row[:, 1].argsort()] for row in matcombine])
    matsort[:, :, 1] = np.transpose(sinosmoothed)
    matsortback = np.asarray(
        [row[row[:, 0].argsort()] for row in matsort])
    sino_corrected = np.transpose(matsortback[:, :, 1])
    listxmiss = np.where(listmask > 0.0)[0]
    sinogram[:, listxmiss] = sino_corrected[:, listxmiss]
    return sinogram

def remove_unresponsive_and_fluctuating_stripe(sinogram, snr, size):
    """Algorithm 6 in the paper. To remove unresponsive and fluctuating
    stripes.

    Parameters
    ----------
    sinogram : 2D array.
    snr : Ratio (>1.0) used to detect stripe locations.
    size : Window size of the median filter.

    Returns
    -------
    sinogram : stripe-removed sinogram.
    """
    (nrow, _) = sinogram.shape
    sinosmoothed = np.apply_along_axis(uniform_filter1d, 0, sinogram, 10)
    listdiff = np.sum(np.abs(sinogram - sinosmoothed), axis=0)
    nmean = np.mean(listdiff)
    listdiffbck = median_filter(listdiff, size)
    listdiffbck[listdiffbck == 0.0] = nmean
    listfact = listdiff / listdiffbck
    listmask = detect_stripe(listfact, snr)
    listmask = binary_dilation(listmask, iterations=1).astype(
        listmask.dtype)
    listmask[0:2] = 0.0
    listmask[-2:] = 0.0
    listx = np.where(listmask < 1.0)[0]
    listy = np.arange(nrow)
    matz = sinogram[:, listx]
    finter = interpolate.interp2d(listx, listy, matz, kind='linear')
    listxmiss = np.where(listmask > 0.0)[0]
    if len(listxmiss) > 0:
        matzmiss = finter(listxmiss, listy)
        sinogram[:, listxmiss] = matzmiss
    return sinogram


def remove_all_rings(projection_data, snr=3.0, sm_size=31, la_size=71):
    """
    assuming projection data to be [angles_total, detector_y, detector_x] 
    """    
    detector_y_size = len(projection_data[1])
    width1 = projection_data[2]
    height1 = projection_data[1]
    listindex = np.arange(0.0, height1, 1.0)
    matindex = np.tile(listindex, (width1, 1))
    la_size = np.clip(np.int16(la_size), 1, width1 - 1)
    sm_size = np.clip(np.int16(sm_size), 1, width1 - 1)
    snr = np.clip(np.float32(snr), 1.0, None)
    
    for i in range(0, detector_y_size):
        sinogram = np.copy(projection_data[:,i,:])    
        sinogram = remove_unresponsive_and_fluctuating_stripe(sinogram, snr, la_size)
        sinogram = remove_large_stripe(matindex, sinogram, snr, la_size)
        sinogram = remove_stripe_based_sorting(matindex, sinogram, sm_size)
    return sinogram