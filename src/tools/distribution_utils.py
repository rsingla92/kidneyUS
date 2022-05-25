import numpy as np
import math
from scipy.stats import nakagami, entropy

def estimate_nakagami(arr):

    #making arrays to compute expectations as per nakagami estimates
    #careful for overflows, need to declare python int in case
    
    arr = arr.astype(np.int64)

    N = arr.size

    arr2 = np.square(arr)
    arr4 = np.square(arr2)

    e_x2 = np.sum(arr2) / N
    e_x4 = np.sum(arr4) / N

    nak_scale = e_x2

    #using inverse normalized variance esimator for Nakagami
    if(( e_x4 - (e_x2**2)) == 0):
        nak_shape = 0
    else:
        nak_shape = e_x2**2 / ( e_x4 - (e_x2**2))
    
    return np.nan_to_num([nak_shape, nak_scale])

def compute_nak_for_mask(img, mask, num_classes):
    all_nak_params = []
    for i in range(1, num_classes + 1):
        pixels = img[np.where(mask==i)]
        nak_params = estimate_nakagami(pixels)
        all_nak_params.append(nak_params)
    return all_nak_params


def compute_snr_for_mask(img, mask, num_classes):
    all_snr = []
    for i in range(1, num_classes + 1):
        pixels = img[np.where(mask==i)]
        if pixels.size > 0:
            mean = np.mean(pixels)
            std = np.std(pixels)
            snr = np.log10(mean / std)
        else:
            snr = 0
        all_snr.append(snr)
    return all_snr

def kl_divergence(p, q):
    """
    Taken from https://towardsdatascience.com/kl-divergence-python-example-b87069e4b810
    """
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def compute_nakagami_kl_divergence(params1, params2):
    lim = max(params1[1], params2[1]) * 4
    x = np.arange(0.01, lim, 0.01)
    p = nakagami.pdf(x, params1[0], loc=0, scale=params2[1])
    q = nakagami.pdf(x, params2[0], loc=0, scale=params2[1])
    
    if params1[0]==0 and params1[1]==0 and params2[0]==0 and params2[1]==0:
        return 0

    # return kl_divergence(p, q)
    kl = entropy(p, qk = q) 

    if(math.isnan(kl)):
        kl = -1
    return kl

