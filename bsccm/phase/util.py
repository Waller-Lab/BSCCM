import numpy as np

### Operators
F = lambda x: np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(x)))
iF = lambda x: np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(x)))


def zernike_polynomials(na, wavelength,  sensor_shape, pixel_size_obj, max_order=5):

    def cart2pol(x, y):
        rho = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)
        return theta, rho

    def inv_factorial(n):
        if n == 0:
            return 1
        elif n > 0:
            return 1 / np.math.factorial(n)
        else:
            return 0


    fx, fy = spatial_freq_grid(sensor_shape, pixel_size_obj)

    theta, rho = cart2pol(fx, fy)
    rho = rho / (na / wavelength)
    Z = []
    pupil = fx ** 2 + fy ** 2 <= (na / wavelength) ** 2

    for n in range(1, max_order+1): #no piston
        for m in np.arange(start=-n, stop=n+1, step=2):
            # print('n: {}, m: {}'.format(n, m))
            R = 0
            for k in range((n - m) // 2 + 1):
                nmk_terms = (-1) ** k * np.math.factorial(n - k) * inv_factorial(k) * inv_factorial((n + m) // 2 - k) * \
                    inv_factorial((n - m) // 2 - k)
                if nmk_terms != 0:
                    R += nmk_terms * rho ** (n - 2 * k)
            Z.append(pupil * R * ((m < 0) * np.sin(np.abs(m)*theta) + (m >= 0) * np.cos(np.abs(m)*theta)))


    return np.array(Z)

def spatial_freq_grid(sensor_shape, pixel_size_object_um):
    """
    Create meshgrid of spatial frequency coordinates
    :param sensor_shape:
    :param pixel_size_object_um:
    :return:
    """
    FoV = np.array(sensor_shape) * pixel_size_object_um  # FoV in the object space

    # Sample size at Fourier plane (set by the image size (FoV))
    dfx_dfy = 1 / FoV

    if np.mod(sensor_shape[0], 2) == 1 or np.mod(sensor_shape[1], 2) == 1:
        raise Exception('Odd number of sensor pixels not supported yet')
        # if np.mod(sensor_shape[0], 2) == 1:
    #     # Odd numer of pixels on sensor -- TODO is this matlba translation correct?
    #     du_dv[0] = 1 / pixel_size_object_um / (sensor_shape[0] - 1)
    # if np.mod(sensor_shape[1], 2) == 1:  # Col
    #     # Odd numer of pixels on sensor -- TODO is this matlba translation correct?
    #     du_dv[1] = 1 / pixel_size_object_um / (sensor_shape[1] - 1)


    fx = dfx_dfy[0] * (np.arange(sensor_shape[0]) - np.round((sensor_shape[0] + 1) / 2))
    # horizontal, col index (object_x)
    fy = dfx_dfy[1] * (np.arange(sensor_shape[1]) - np.round((sensor_shape[1] + 1) / 2))
    fy_grid, fx_grid = np.meshgrid(fy, fx)  # Create grid
    return fx_grid, fy_grid

def compute_pupil_support(wavelength_um, NA, pixel_size_object_um, sensor_shape, inner_na=None):
    """
    Function for computing pupil support for DPC/FPM. Can also be used for computing source
    :param wavelength_um:
    :param NA:
    :param pixel_size_object_um:
    :param sensor_shape:
    :param inner_na: if not none, return annular pupil
    :return:
    """
    fx_grid, fy_grid = spatial_freq_grid(sensor_shape, pixel_size_object_um)

    max_optical_spatial_freq = NA / wavelength_um  # Maximum spatial frequency set by NA (1/um)

    # assume a circular pupil function, long pass filter due to finite NA
    # Find locations of acceptable circle in k-space
    # pupil support is indexed by number of pixels on sensor, but meaningless outside
    # of support (i.e. optical bandwidth)
    if inner_na is None:
        pupil_support = np.sqrt((fx_grid / max_optical_spatial_freq) ** 2 + (fy_grid / max_optical_spatial_freq) ** 2) < 1
    else:
        vals = np.sqrt((fx_grid / max_optical_spatial_freq) ** 2 + (fy_grid / max_optical_spatial_freq) ** 2)
        pupil_support = np.logical_and(vals < 1, vals > inner_na / wavelength_um)
    return pupil_support

def compute_synthetic_image_size(wavelength_um, NA, mag, pixel_size_camera_um, sensor_shape, illumination_na):
    """
    Compute information about the sampling and dimesnions of object, pupil
    :param wavelength_um: wavelength of light
    :param NA:
    :param mag:  magnification
    :param pixel_size_camera_um:  size of camera pixels on sensro
    :param sensor_shape:  size of the image region to be processed in [row, col] (that is, (object_y,object_x))
    :param illumNA:
    :param n_r:
    :return:
    """

    pupil_support = compute_pupil_support(wavelength_um, NA, mag, pixel_size_camera_um, sensor_shape)


    ##### Things related to object. Now NA includes illumination NA also ###########
    # maxium spatial frequency achievable based on the maximum illumination
    # angle from the LED array and NA of the objective
    max_spatial_freq_synthetic = np.max(illumination_na) / wavelength_um + max_optical_spatial_freq
    NA_synthetic = max_spatial_freq_synthetic * wavelength_um
    print('synthetic NA is {}'.format(NA_synthetic))

    # assume the max spatial freq of the original object is greater than the max synthetic spatial freq
    # First factor of 2: Doing intensity instead of amplitude -- autocorrelation of spectrum increases freq support
    # Second factor of 2: Nyquist sampling
    # Solving for the number of samples in freq space
    n_samples_synth = np.round(max_spatial_freq_synthetic / du_dv) * 2 * 2

    # need to enforce N_obj_pixels/NumPixels = integer to ensure no FT artifacts
    multiplier = np.max(np.ceil(n_samples_synth / sensor_shape))
    n_samples_synth = multiplier * sensor_shape

    return n_samples_synth, pupil_support