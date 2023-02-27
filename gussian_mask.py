import matplotlib.pylab as plt
import numpy as np

def gaussian1d(pattern_shape, factor,direction = "column",center=None, cov=None):
    """
    Description: creates a 1D gaussian sampling pattern either in the row or column direction
    of a 2D image
    :param factor: sampling factor in the desired direction
    :param direction: sampling direction, 'row' or 'column'
    :param pattern_shape: shape of the desired sampling pattern.
    :param center: coordinates of the center of the Gaussian distribution
    :param cov: covariance matrix of the distribution
    :return: sampling pattern image. It is a boolean image
    """

    if direction != "column":
        pattern_shape = (pattern_shape[1],pattern_shape[0])

    if center is None:
        center = np.array([1.0 * pattern_shape[1] / 2 - 0.5])

    if cov is None:
        cov = np.array([[(1.0 * pattern_shape[1] / 4) ** 2]])


    factor = int(factor * pattern_shape[1])

    samples = np.array([0])

    m = 1  # Multiplier. We have to increase this value
    # until the number of points (disregarding repeated points)
    # is equal to factor

    while (samples.shape[0] < factor):

        samples = np.random.multivariate_normal(center, cov, m * factor)
        samples = np.rint(samples).astype(int)
        indexes = np.logical_and(samples >= 0, samples < pattern_shape[1])
        samples = samples[indexes]
        samples = np.unique(samples)
        if samples.shape[0] < factor:
            m *= 2
            continue

    indexes = np.arange(samples.shape[0], dtype=int)
    np.random.shuffle(indexes)
    samples = samples[indexes][:factor]

    under_pattern = np.zeros(pattern_shape, dtype=bool)
    under_pattern[:, samples] = True

    if direction != "column":
        under_pattern = under_pattern.T

    return under_pattern


def gaussian2d(pattern_shape, factor, center=None, cov=None):
    """
    Description: creates a 2D gaussian sampling pattern of a 2D image
    :param factor: sampling factor in the desired direction
    :param pattern_shape: shape of the desired sampling pattern.
    :param center: coordinates of the center of the Gaussian distribution
    :param cov: covariance matrix of the distribution
    :return: sampling pattern image. It is a boolean image
    """
    N = pattern_shape[0] * pattern_shape[1]  # Image length

    factor = int(N * factor)

    if center is None:
        center = np.array([1.0 * pattern_shape[0] / 2 - 0.5, 1.0 * pattern_shape[1] / 2 - 0.5])

    if cov is None:
        cov = np.array([[(1.0 * pattern_shape[0] / 4) ** 2, 0], [0, (1.0 * pattern_shape[1] / 4) ** 2]])

    samples = np.array([0])

    m = 1  # Multiplier. We have to increase this value
    # until the number of points (disregarding repeated points)
    # is equal to factor

    while (samples.shape[0] < factor):
        samples = np.random.multivariate_normal(center, cov, m * factor)
        samples = np.rint(samples).astype(int)
        indexesx = np.logical_and(samples[:, 0] >= 0, samples[:, 0] < pattern_shape[0])
        indexesy = np.logical_and(samples[:, 1] >= 0, samples[:, 1] < pattern_shape[1])
        indexes = np.logical_and(indexesx, indexesy)
        samples = samples[indexes]
        # samples[:,0] = np.clip(samples[:,0],0,input_shape[0]-1)
        # samples[:,1] = np.clip(samples[:,1],0,input_shape[1]-1)
        samples = np.unique(samples[:, 0] + 1j * samples[:, 1])
        samples = np.column_stack((samples.real, samples.imag)).astype(int)
        if samples.shape[0] < factor:
            m *= 2
            continue

    indexes = np.arange(samples.shape[0], dtype=int)
    np.random.shuffle(indexes)
    samples = samples[indexes][:factor]

    under_pattern = np.zeros(pattern_shape, dtype=bool)
    under_pattern[samples[:, 0], samples[:, 1]] = True
    return under_pattern


pattern_shape = (256,256)
factor= 0.90
af= 5
gussian_mask_uniform = gaussian1d(pattern_shape, factor, direction = "row")
gussian_mask_random = gaussian2d(pattern_shape,factor= factor)
#np.save('../mask/gaussian_mask_AF-{50}_2D_256X256.npy',gussian_mask)

gussian_mask = gussian_mask_uniform
uncentered_mask= np.fft.fftshift(gussian_mask)

plt.figure()
plt.subplot(121)
plt.imshow(gussian_mask,cmap='BuGn' )
plt.axis("off")
plt.title("center mask of AF={}".format(af))

plt.subplot(122)
plt.imshow(uncentered_mask, cmap='Blues')
plt.axis("off")
plt.title("".format(af))
plt.show()
#np.save('../mask/gussian_mask_2D_50_random.npy',uncentered_mask)

# ValueError: 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r',
# 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r',
# 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2',
# 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples',
# 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds',
# 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn',
# 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r',
# 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r',
# 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth',
# 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow',
# 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r',
# 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r',
# 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r',
# 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20',
# 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r',
# 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'