import numpy as np
import numpy as onp
from bsccm.phase.util import compute_pupil_support,  spatial_freq_grid

F = lambda x: np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(x)))
iF = lambda x: np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(x)))

def generate_source(wavelength_um, pixel_size_object_um, source_specs, image_shape,
                    coherent_source='gaussian'):
    """
    :param wavelength_um:
    :param pixel_size_object_um:
    :param source_specs:
    :param image_shape:
    :param coherent_source: make the coherent source a delta or a gaussian
    :return:
    """
    sources = []
    fx, fy = spatial_freq_grid(image_shape, pixel_size_object_um)

    for spec in source_specs:
        na = spec['NA']
        if 'LED NA' in spec:
            #compute a single LED source as a tiny gaussian spot
            led_f = spec['LED NA'] / wavelength_um
            if coherent_source == 'gaussian':
                sigma = 0.001 / wavelength_um
                source = np.exp(-(np.sum((np.stack([fy, fx], axis=2) - led_f) ** 2, axis=2) / sigma **2))
                pupil_support = compute_pupil_support(wavelength_um, na, pixel_size_object_um, image_shape)
                sources.append(source*pupil_support)
            elif coherent_source == 'delta':
                source = onp.zeros(image_shape)
                distances = np.sum((np.stack([fy, fx], axis=2) - led_f) ** 2, axis=2)
                min_index = onp.argmin(distances)
                min_indices = onp.unravel_index(min_index, distances.shape) #not yet implemented in jax
                source[min_indices[0], min_indices[1]] = 1
                sources.append(source)

        else:
            na_inner = 0 if 'NA inner' not in spec else spec['NA inner']
            rotation = spec['rotation']
            annular_mask = compute_pupil_support(wavelength_um, na, pixel_size_object_um, image_shape, inner_na=na_inner)
            sources.append(np.array(annular_mask * (np.cos(np.deg2rad(rotation)) * fx + 1e-15 >=
                            np.sin(np.deg2rad(rotation)) * fy), dtype=np.float32))
    return np.array(sources)

def generate_pupil(wavelength_um, pixel_size_object_um, na, image_shape):
    pupil_support = compute_pupil_support(wavelength_um, na, pixel_size_object_um, image_shape)
    pupil = pupil_support.astype(np.complex64)
    return pupil

def generate_wotf(sources, pupil):
    ######### Generate WOTF ################
    Hu = []  # amplitude transfer
    Hp = []  # phase transfer
    for i in range(sources.shape[0]):
        I0 = (sources[i] * pupil * pupil.conj()).sum()  # total intensity passing through system
        # # michael chens code: for a purely real pupil
        # cFP_FSP = F(pupil).conj() * F(sources[i] * pupil)
        # hu = 2.0 * iF(cFP_FSP.real)
        # hp = 2.0j * iF(1j * cFP_FSP.imag)

        term1 = F(pupil).conj() * F(sources[i] * pupil)
        term2 = F(sources[i] * pupil.conj()).conj() * F(pupil.conj())
        hu = iF(term1 + term2)
        hp = 1j * iF(term1 - term2)

        Hu.append(hu / I0)
        Hp.append(hp / I0)
        # napari_show({'Hu': hu.real, 'Hp': hp.imag})

    Hu = np.asarray(Hu)
    Hp = np.asarray(Hp)
    # napari_show({'Hu': hu.real,  'Hp': hp.imag })
    return Hu, Hp


def dpc_loss(object_real, object_imag, dpc_images, sources, pupil, reg_u=1e-1, reg_p=5e-3):
    Hu, Hp = generate_wotf(sources, pupil)

    f_object_real = F(object_real)
    f_object_imag = F(object_imag)

    predicted_intensity = iF(Hu * f_object_real + Hp * f_object_imag).real
    loss = (dpc_images - predicted_intensity) ** 2
    reg = iF(reg_u * f_object_real + reg_p * f_object_imag).real ** 2
    # reg = 0
    return np.sum(loss + reg)


def tikhinov_solver(dpc_images, sources, pupil, Hu=None, Hp=None, reg_u=1e-1, reg_p=5e-3):
    if Hu is None or Hp is None:
        Hu, Hp = generate_wotf(sources, pupil)

    ###### Solve with L2 regularization ############
    AHA = [(Hu.conj() * Hu).sum(axis=0) + reg_u, (Hu.conj() * Hp).sum(axis=0),
               (Hp.conj() * Hu).sum(axis=0), (Hp.conj() * Hp).sum(axis=0) + reg_p]
    determinant = AHA[0] * AHA[3] - AHA[1] * AHA[2]
    dpc_result = []
    for frame_index in range(dpc_images.shape[0] // len(sources)):
        fIntensity = np.asarray(
            [F(dpc_images[frame_index * len(sources) + image_index]) for image_index in range(len(sources))])
        AHy = np.asarray([(Hu.conj() * fIntensity).sum(axis=0), (Hp.conj() * fIntensity).sum(axis=0)])
        absorption = iF((AHA[3] * AHy[0] - AHA[1] * AHy[1]) / determinant).real
        phase = iF((AHA[0]*AHy[1]-AHA[2]*AHy[0])/determinant).real
        dpc_result.append(absorption + 1.0j * phase)


    return np.asarray(dpc_result)

def annular_sources(NA, NA_inner, rotation, image_shape, wavelength, pixel_size_obj):
    source_specs = [{'NAinner': NA_inner, 'NA': NA, 'rotation': rot} for rot in rotation]
    sources = generate_source(wavelength, pixel_size_obj, source_specs, image_shape)
    return sources
