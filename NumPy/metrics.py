import numpy as np
import scipy.ndimage as ft
from skimage.transform.integral import integral_image as integral
from math import ceil, floor, log2
from torch import nn
import torch


def local_cross_correlation(img_1, img_2, half_width):
    """
        Cross-Correlation Field computation.

        Parameters
        ----------
        img_1 : Numpy Array
            First image on which calculate the cross-correlation. Dimensions: H, W
        img_2 : Numpy Array
            Second image on which calculate the cross-correlation. Dimensions: H, W
        half_width : int
            The semi-size of the window on which calculate the cross-correlation


        Return
        ------
        L : Numpy array
            The cross-correlation map between img_1 and img_2

    """

    w = int(half_width)
    ep = 1e-20

    if (len(img_1.shape)) != 3:
        img_1 = np.expand_dims(img_1, axis=-1)
    if (len(img_2.shape)) != 3:
        img_2 = np.expand_dims(img_2, axis=-1)

    img_1 = np.pad(img_1.astype(np.float64), ((w, w), (w, w), (0, 0)))
    img_2 = np.pad(img_2.astype(np.float64), ((w, w), (w, w), (0, 0)))

    img_1_cum = np.zeros(img_1.shape)
    img_2_cum = np.zeros(img_2.shape)
    for i in range(img_1.shape[-1]):
        img_1_cum[:, :, i] = integral(img_1[:, :, i]).astype(np.float64)
    for i in range(img_2.shape[-1]):
        img_2_cum[:, :, i] = integral(img_2[:, :, i]).astype(np.float64)

    img_1_mu = (img_1_cum[2 * w:, 2 * w:, :] - img_1_cum[:-2 * w, 2 * w:, :] - img_1_cum[2 * w:, :-2 * w, :]
                + img_1_cum[:-2 * w, :-2 * w, :]) / (4 * w ** 2)
    img_2_mu = (img_2_cum[2 * w:, 2 * w:, :] - img_2_cum[:-2 * w, 2 * w:, :] - img_2_cum[2 * w:, :-2 * w, :]
                + img_2_cum[:-2 * w, :-2 * w, :]) / (4 * w ** 2)

    img_1 = img_1[w:-w, w:-w, :] - img_1_mu
    img_2 = img_2[w:-w, w:-w, :] - img_2_mu

    img_1 = np.pad(img_1.astype(np.float64), ((w, w), (w, w), (0, 0)))
    img_2 = np.pad(img_2.astype(np.float64), ((w, w), (w, w), (0, 0)))

    i2 = img_1 ** 2
    j2 = img_2 ** 2
    ij = img_1 * img_2

    i2_cum = np.zeros(i2.shape)
    j2_cum = np.zeros(j2.shape)
    ij_cum = np.zeros(ij.shape)

    for i in range(i2_cum.shape[-1]):
        i2_cum[:, :, i] = integral(i2[:, :, i]).astype(np.float64)
    for i in range(j2_cum.shape[-1]):
        j2_cum[:, :, i] = integral(j2[:, :, i]).astype(np.float64)
    for i in range(ij_cum.shape[-1]):
        ij_cum[:, :, i] = integral(ij[:, :, i]).astype(np.float64)

    sig2_ij_tot = (ij_cum[2 * w:, 2 * w:, :] - ij_cum[:-2 * w, 2 * w:, :] - ij_cum[2 * w:, :-2 * w, :]
                   + ij_cum[:-2 * w, :-2 * w, :])
    sig2_ii_tot = (i2_cum[2 * w:, 2 * w:, :] - i2_cum[:-2 * w, 2 * w:, :] - i2_cum[2 * w:, :-2 * w, :]
                   + i2_cum[:-2 * w, :-2 * w, :])
    sig2_jj_tot = (j2_cum[2 * w:, 2 * w:, :] - j2_cum[:-2 * w, 2 * w:, :] - j2_cum[2 * w:, :-2 * w, :]
                   + j2_cum[:-2 * w, :-2 * w, :])

    sig2_ii_tot = np.clip(sig2_ii_tot, ep, sig2_ii_tot.max())
    sig2_jj_tot = np.clip(sig2_jj_tot, ep, sig2_jj_tot.max())

    xcorr = sig2_ij_tot / ((sig2_ii_tot * sig2_jj_tot) ** 0.5 + ep)

    return xcorr


def normalize_block(im):
    """
        Auxiliary Function for Q2n computation.

        Parameters
        ----------
        im : Numpy Array
            Image on which calculate the statistics. Dimensions: H, W

        Return
        ------
        y : Numpy array
            The normalized version of im
        m : float
            The mean of im
        s : float
            The standard deviation of im

    """

    m = np.mean(im)
    s = np.std(im, ddof=1)

    if s == 0:
        s = 1e-10

    y = ((im - m) / s) + 1

    return y, m, s


def cayley_dickson_property_1d(onion1, onion2):
    """
        Cayley-Dickson construction for 1-D arrays.
        Auxiliary function for Q2n calculation.

        Parameters
        ----------
        onion1 : Numpy Array
            First 1-D array
        onion2 : Numpy Array
            Second 1-D array

        Return
        ------
        ris : Numpy array
            The result of Cayley-Dickson construction on the two arrays.
    """

    n = onion1.__len__()

    if n > 1:
        half_pos = int(n / 2)
        a = onion1[:half_pos]
        b = onion1[half_pos:]

        neg = np.ones(b.shape)
        neg[1:] = -1

        b = b * neg
        c = onion2[:half_pos]
        d = onion2[half_pos:]
        d = d * neg

        if n == 2:
            ris = np.concatenate([(a * c) - (d * b), (a * d) + (c * b)])
        else:
            ris1 = cayley_dickson_property_1d(a, c)

            ris2 = cayley_dickson_property_1d(d, b * neg)
            ris3 = cayley_dickson_property_1d(a * neg, d)
            ris4 = cayley_dickson_property_1d(c, b)

            aux1 = ris1 - ris2
            aux2 = ris3 + ris4
            ris = np.concatenate([aux1, aux2])
    else:
        ris = onion1 * onion2

    return ris


def cayley_dickson_property_2d(onion1, onion2):
    """
        Cayley-Dickson construction for 2-D arrays.
        Auxiliary function for Q2n calculation.

        Parameters
        ----------
        onion1 : Numpy Array
            First MultiSpectral img. Dimensions: H, W, Bands
        onion2 : Numpy Array
            Second MultiSpectral img. Dimensions: H, W, Bands

        Return
        ------
        ris : Numpy array
            The result of Cayley-Dickson construction on the two arrays.
    """

    dim3 = onion1.shape[-1]
    if dim3 > 1:
        half_pos = int(dim3 / 2)

        a = onion1[:, :, :half_pos]
        b = onion1[:, :, half_pos:]
        b = np.concatenate([np.expand_dims(b[:, :, 0], -1), -b[:, :, 1:]], axis=-1)

        c = onion2[:, :, :half_pos]
        d = onion2[:, :, half_pos:]
        d = np.concatenate([np.expand_dims(d[:, :, 0], -1), -d[:, :, 1:]], axis=-1)

        if dim3 == 2:
            ris = np.concatenate([(a * c) - (d * b), (a * d) + (c * b)], axis=-1)
        else:
            ris1 = cayley_dickson_property_2d(a, c)
            ris2 = cayley_dickson_property_2d(d,
                                              np.concatenate([np.expand_dims(b[:, :, 0], -1), -b[:, :, 1:]], axis=-1))
            ris3 = cayley_dickson_property_2d(np.concatenate([np.expand_dims(a[:, :, 0], -1), -a[:, :, 1:]], axis=-1),
                                              d)
            ris4 = cayley_dickson_property_2d(c, b)

            aux1 = ris1 - ris2
            aux2 = ris3 + ris4

            ris = np.concatenate([aux1, aux2], axis=-1)
    else:
        ris = onion1 * onion2

    return ris


def q_index_metric(im1, im2, size):
    """
        Q2n calculation on a window of dimension (size, size).
        Auxiliary function for Q2n calculation.

        Parameters
        ----------
        im1 : Numpy Array
            First MultiSpectral img. Dimensions: H, W, Bands
        im2 : Numpy Array
            Second MultiSpectral img. Dimensions: H, W, Bands
        size : int
            The size of the squared windows on which calculate the UQI index


        Return
        ------
        q : Numpy array
            The Q2n calculated on a window of dimension (size,size).
    """

    im1 = im1.astype(np.double)
    im2 = im2.astype(np.double)
    im2 = np.concatenate([np.expand_dims(im2[:, :, 0], -1), -im2[:, :, 1:]], axis=-1)

    depth = im1.shape[-1]
    for i in range(depth):
        im1[:, :, i], m, s = normalize_block(im1[:, :, i])
        if m == 0:
            if i == 0:
                im2[:, :, i] = im2[:, :, i] - m + 1
            else:
                im2[:, :, i] = -(-im2[:, :, i] - m + 1)
        else:
            if i == 0:
                im2[:, :, i] = ((im2[:, :, i] - m) / s) + 1
            else:
                im2[:, :, i] = -(((-im2[:, :, i] - m) / s) + 1)

    m1 = np.mean(im1, axis=(0, 1))
    m2 = np.mean(im2, axis=(0, 1))

    mod_q1m = np.sqrt(np.sum(m1 ** 2))
    mod_q2m = np.sqrt(np.sum(m2 ** 2))

    mod_q1 = np.sqrt(np.sum(im1 ** 2, axis=-1))
    mod_q2 = np.sqrt(np.sum(im2 ** 2, axis=-1))

    term2 = mod_q1m * mod_q2m
    term4 = mod_q1m ** 2 + mod_q2m ** 2
    temp = (size ** 2) / (size ** 2 - 1)
    int1 = temp * np.mean(mod_q1 ** 2)
    int2 = temp * np.mean(mod_q2 ** 2)
    int3 = temp * (mod_q1m ** 2 + mod_q2m ** 2)
    term3 = int1 + int2 - int3

    mean_bias = 2 * term2 / term4

    if term3 == 0:
        q = np.zeros((1, 1, depth), dtype='float64')
        q[:, :, -1] = mean_bias
    else:
        cbm = 2 / term3
        qu = cayley_dickson_property_2d(im1, im2)
        qm = cayley_dickson_property_1d(m1, m2)

        qv = temp * np.mean(qu, axis=(0, 1))
        q = qv - temp * qm
        q = q * mean_bias * cbm

    return q


def Q2n(outputs, labels, q_block_size=32, q_shift=32):
    """
        Q2n calculation on a window of dimension (size, size).
        Auxiliary function for Q2n calculation.

        [Scarpa21]          Scarpa, Giuseppe, and Matteo Ciotola. "Full-resolution quality assessment for pansharpening.",
                            arXiv preprint arXiv:2108.06144
        [Garzelli09]        A. Garzelli and F. Nencini, "Hypercomplex quality assessment of multi/hyper-spectral images,"
                            IEEE Geoscience and Remote Sensing Letters, vol. 6, no. 4, pp. 662-665, October 2009.
        [Vivone20]          G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M.O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting pansharpening with classical and emerging pansharpening methods",
                            IEEE Geoscience and Remote Sensing Magazine, doi: 10.1109/MGRS.2020.3019315.

        Parameters
        ----------
        outputs : Numpy Array
            The Fused image. Dimensions: H, W, Bands
        labels : Numpy Array
            The reference image. Dimensions: H, W, Bands
        q_block_size : int
            The windows size on which calculate the Q2n index
        q_shift : int
            The stride for Q2n index calculation

        Return
        ------
        q2n_index : float
            The Q2n index.
        q2n_index_map : Numpy Array
            The Q2n map, on a support of (q_block_size, q_block_size)
    """

    height, width, depth = labels.shape
    stepx = ceil(height / q_shift)
    stepy = ceil(width / q_shift)

    if stepy <= 0:
        stepx = 1
        stepy = 1

    est1 = (stepx - 1) * q_shift + q_block_size - height
    est2 = (stepy - 1) * q_shift + q_block_size - width

    if (est1 != 0) and (est2 != 0):
        labels = np.pad(labels, ((0, est1), (0, est2), (0, 0)), mode='reflect')
        outputs = np.pad(outputs, ((0, est1), (0, est2), (0, 0)), mode='reflect')

        outputs = outputs.astype(np.int16)
        labels = labels.astype(np.int16)

    height, width, depth = labels.shape

    if ceil(log2(depth)) - log2(depth) != 0:
        exp_difference = 2 ** (ceil(log2(depth))) - depth
        diff_zeros = np.zeros((height, width, exp_difference), dtype="float64")
        labels = np.concatenate([labels, diff_zeros], axis=-1)
        outputs = np.concatenate([outputs, diff_zeros], axis=-1)

    height, width, depth = labels.shape

    values = np.zeros((stepx, stepy, depth))
    for j in range(stepx):
        for i in range(stepy):
            values[j, i, :] = q_index_metric(
                labels[j * q_shift:j * q_shift + q_block_size, i * q_shift: i * q_shift + q_block_size, :],
                outputs[j * q_shift:j * q_shift + q_block_size, i * q_shift: i * q_shift + q_block_size, :],
                q_block_size
            )

    q2n_index_map = np.sqrt(np.sum(values ** 2, axis=-1))
    q2n_index = np.mean(q2n_index_map)

    return q2n_index.item(), q2n_index_map


def ERGAS(outputs, labels, ratio):
    """
        Erreur Relative Globale Adimensionnelle de Synthèse (ERGAS).


        [Scarpa21]          Scarpa, Giuseppe, and Matteo Ciotola. "Full-resolution quality assessment for pansharpening.",
                            arXiv preprint arXiv:2108.06144
        [Ranchin00]         T. Ranchin and L. Wald, "Fusion of high spatial and spectral resolution images: the ARSIS concept and its implementation,"
                            Photogrammetric Engineering and Remote Sensing, vol. 66, no. 1, pp. 4961, January 2000.
        [Vivone20]          G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M.O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting pansharpening with classical and emerging pansharpening methods",
                            IEEE Geoscience and Remote Sensing Magazine, doi: 10.1109/MGRS.2020.3019315.

        Parameters
        ----------
        outputs : Numpy Array
            The Fused image. Dimensions: H, W, Bands
        labels : Numpy Array
            The reference image. Dimensions: H, W, Bands
        ratio : int
            PAN-MS resolution ratio

        Return
        ------
        ergas_index : float
            The ERGAS index.

    """

    mu = np.mean(labels, axis=(0, 1)) ** 2
    nbands = labels.shape[-1]
    error = np.mean((outputs - labels) ** 2, axis=(0, 1))
    ergas_index = 100 / ratio * np.sqrt(np.sum(error / mu) / nbands)

    return np.mean(ergas_index).item()


def SAM(outputs, labels):
    """
        Spectral Angle Mapper (SAM).


        [Scarpa21]          Scarpa, Giuseppe, and Matteo Ciotola. "Full-resolution quality assessment for pansharpening.",
                            arXiv preprint arXiv:2108.06144
        [Yuhas92]           R. H. Yuhas, A. F. H. Goetz, and J. W. Boardman, "Discrimination among semi-arid landscape endmembers using the Spectral Angle Mapper (SAM) algorithm,"
                            in Proceeding Summaries 3rd Annual JPL Airborne Geoscience Workshop, 1992, pp. 147-149.
        [Vivone20]          G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M.O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting pansharpening with classical and emerging pansharpening methods",
                            IEEE Geoscience and Remote Sensing Magazine, doi: 10.1109/MGRS.2020.3019315.

        Parameters
        ----------
        outputs : Numpy Array
            The Fused image. Dimensions: H, W, Bands
        labels : Numpy Array
            The reference image. Dimensions: H, W, Bands

        Return
        ------
        angle : float
            The SAM index in degree.
    """

    norm_outputs = np.sum(outputs ** 2, axis=-1)
    norm_labels = np.sum(labels ** 2, axis=-1)
    scalar_product = np.sum(outputs * labels, axis=-1)
    norm_product = np.sqrt(norm_outputs * norm_labels)
    scalar_product[norm_product == 0] = np.nan
    norm_product[norm_product == 0] = np.nan
    scalar_product = scalar_product.flatten()
    norm_product = norm_product.flatten()
    angle = np.nansum(np.arccos(np.clip(scalar_product / norm_product, a_min=-1, a_max=1)), axis=-1) / norm_product.shape[0]
    angle = angle * 180 / np.pi

    return angle


def Q(outputs, labels, block_size=32):
    """
        Universal Quality Index (UQI).


        [Scarpa21]          Scarpa, Giuseppe, and Matteo Ciotola. "Full-resolution quality assessment for pansharpening.",
                            arXiv preprint arXiv:2108.06144
        [Wang02]            Z. Wang and A. C. Bovik, "A universal image quality index,"
                            IEEE Signal Processing Letters, vol. 9, no. 3, pp. 81-84, March 2002.
        [Vivone20]          G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M.O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting pansharpening with classical and emerging pansharpening methods",
                            IEEE Geoscience and Remote Sensing Magazine, doi: 10.1109/MGRS.2020.3019315.

        Parameters
        ----------
        outputs : Numpy Array
            The Fused image. Dimensions: H, W, Bands
        labels : Numpy Array
            The reference image. Dimensions: H, W, Bands
        block_size : int
            The windows size on which calculate the Q2n index

        Return
        ------
        quality : float
            The UQI index.
    """

    N = block_size ** 2
    nbands = labels.shape[-1]
    kernel = np.ones((block_size, block_size))
    pad_size = floor((kernel.shape[0] - 1) / 2)
    outputs_sq = outputs ** 2
    labels_sq = labels ** 2
    outputs_labels = outputs * labels

    quality = np.zeros(nbands)
    for i in range(nbands):
        outputs_sum = ft.convolve(outputs[:, :, i], kernel)
        labels_sum = ft.convolve(labels[:, :, i], kernel)

        outputs_sq_sum = ft.convolve(outputs_sq[:, :, i], kernel)
        labels_sq_sum = ft.convolve(labels_sq[:, :, i], kernel)
        outputs_labels_sum = ft.convolve(outputs_labels[:, :, i], kernel)
        outputs_sum = outputs_sum[pad_size:-pad_size, pad_size:-pad_size]
        labels_sum = labels_sum[pad_size:-pad_size, pad_size:-pad_size]

        outputs_sq_sum = outputs_sq_sum[pad_size:-pad_size, pad_size:-pad_size]
        labels_sq_sum = labels_sq_sum[pad_size:-pad_size, pad_size:-pad_size]
        outputs_labels_sum = outputs_labels_sum[pad_size:-pad_size, pad_size:-pad_size]

        outputs_labels_sum_mul = outputs_sum * labels_sum
        outputs_labels_sum_mul_sq = outputs_sum ** 2 + labels_sum ** 2

        numerator = 4 * (N * outputs_labels_sum - outputs_labels_sum_mul) * outputs_labels_sum_mul
        denominator_temp = N * (outputs_sq_sum + labels_sq_sum) - outputs_labels_sum_mul_sq
        denominator = denominator_temp * outputs_labels_sum_mul_sq

        index = (denominator_temp == 0) & (outputs_labels_sum_mul_sq != 0)

        quality_map = np.ones(denominator.shape)
        quality_map[index] = 2 * outputs_labels_sum_mul[index] / outputs_labels_sum_mul_sq[index]
        index = denominator != 0
        quality_map[index] = numerator[index] / denominator[index]
        quality[i] = np.mean(quality_map)

    return np.mean(quality).item()


def coregistration(ms, pan, kernel, ratio=4, search_win=4):
    """
        Coregitration function for MS-PAN pair.

        [Scarpa21]          Scarpa, Giuseppe, and Matteo Ciotola. "Full-resolution quality assessment for pansharpening.",
                            arXiv preprint arXiv:2108.06144

        Parameters
        ----------
        ms : Numpy Array
            The Multi-Spectral image. Dimensions: H, W, Bands
        pan : Numpy Array
            The PAN image. Dimensions: H, W
        kernel : Numpy Array
            The filter array.
        ratio : int
            PAN-MS resolution ratio
        search_win : int
            The windows in which search the optimal value for the coregistration step

        Return
        ------
        r : Numpy Array
            The optimal raw values.
        c : Numpy Array
            The optimal column values.
    """

    nbands = ms.shape[-1]
    p = ft.convolve(pan, kernel, mode='nearest')
    rho = np.zeros((search_win, search_win, nbands))
    r = np.zeros(nbands)
    c = np.copy(r)

    for i in range(search_win):
        for j in range(search_win):
            rho[i, j, :] = np.mean(
                local_cross_correlation(ms, np.expand_dims(p[i::ratio, j::ratio], -1), floor(ratio / 2)), axis=(0, 1))

    max_value = np.amax(rho, axis=(0, 1))

    for b in range(nbands):
        x = rho[:, :, b]
        max_value = x.max()
        pos = np.where(x == max_value)
        if len(pos[0]) != 1:
            pos = (pos[0][0], pos[1][0])
        pos = tuple(map(int, pos))
        r[b] = pos[0]
        c[b] = pos[1]
        r = np.squeeze(r).astype(np.uint8)
        c = np.squeeze(c).astype(np.uint8)

    return r, c


def resize_with_mtf(outputs, ms, pan, sensor, ratio=4, dim_cut=21):
    """
        Resize of Fused Image to MS scale, in according to the coregistration with the PAN.
        If dim_cut is different by zero a cut is made on both outputs and ms, to discard possibly values affected by paddings.

        [Scarpa21]          Scarpa, Giuseppe, and Matteo Ciotola. "Full-resolution quality assessment for pansharpening.",
                            arXiv preprint arXiv:2108.06144

        Parameters
        ----------
        outputs : Numpy Array
            Fused MultiSpectral img. Dimensions: H, W, Bands
        ms : Numpy Array
            MultiSpectral img. Dimensions: H, W, Bands
        pan : Numpy Array
            Panchromatic img. Dimensions: H, W, Bands
        sensor : str
            The name of the satellites which has provided the images.
        ratio : int
            Resolution scale which elapses between MS and PAN.
        dim_cut : int
            Cutting dimension for obtaining "valid" image to which apply the metrics

        Return
        ------
        x : NumPy array
            Fused MultiSpectral image, coregistered with the PAN, low-pass filtered and decimated. If dim_cut is different
            by zero it is also cut
        ms : NumPy array
            MultiSpectral img. If dim_cut is different by zero it is cut.
    """

    from spectral_tools import gen_mtf

    kernel = gen_mtf(ratio, sensor)
    kernel = kernel.astype(np.float32)
    nbands = kernel.shape[-1]
    pad_size = floor((kernel.shape[0] - 1) / 2)

    r, c = coregistration(ms, pan, kernel[:, :, 0], ratio)

    kernel = np.moveaxis(kernel, -1, 0)
    kernel = np.expand_dims(kernel, axis=1)

    kernel = torch.from_numpy(kernel).type(torch.float32)

    depthconv = nn.Conv2d(in_channels=nbands,
                          out_channels=nbands,
                          groups=nbands,
                          kernel_size=kernel.shape,
                          bias=False)
    depthconv.weight.data = kernel
    depthconv.weight.requires_grad = False
    pad = nn.ReplicationPad2d(pad_size)

    x = np.zeros(ms.shape, dtype=np.float32)

    outputs = np.expand_dims(np.moveaxis(outputs, -1, 0), 0)
    outputs = torch.from_numpy(outputs)

    outputs = pad(outputs)
    outputs = depthconv(outputs)

    outputs = outputs.detach().cpu().numpy()
    outputs = np.moveaxis(np.squeeze(outputs, 0), 0, -1)

    for b in range(nbands):
        x[:, :, b] = outputs[r[b]::ratio, c[b]::ratio, b]

    if dim_cut != 0:
        x = x[dim_cut:-dim_cut, dim_cut:-dim_cut, :]
        ms = ms[dim_cut:-dim_cut, dim_cut:-dim_cut, :]

    return x, ms


def ReproERGAS(outputs, ms, pan, sensor, ratio=4, dim_cut=0):
    """
        Reprojected Erreur Relative Globale Adimensionnelle de Synthèse (ERGAS).

        [Scarpa21]          Scarpa, Giuseppe, and Matteo Ciotola. "Full-resolution quality assessment for pansharpening.",
                            arXiv preprint arXiv:2108.06144

        Parameters
        ----------
        outputs : Numpy Array
            Fused MultiSpectral img. Dimensions: H, W, Bands
        ms : Numpy Array
            MultiSpectral img. Dimensions: H, W, Bands
        pan : Numpy Array
            Panchromatic img. Dimensions: H, W, Bands
        sensor : str
            The name of the satellites which has provided the images.
        ratio : int
            Resolution scale which elapses between MS and PAN.
        dim_cut : int
            Cutting dimension for obtaining "valid" image to which apply the metrics

        Return
        ------
        R-ERGAS : float
            The R-ERGAS index

    """

    outputs, ms = resize_with_mtf(outputs, ms, pan, sensor, ratio, dim_cut)
    return ERGAS(outputs, ms, ratio)


def ReproSAM(outputs, ms, pan, sensor, ratio=4, dim_cut=0):
    """
        Reprojected Spectral Angle Mapper (SAM).

        [Scarpa21]          Scarpa, Giuseppe, and Matteo Ciotola. "Full-resolution quality assessment for pansharpening.",
                            arXiv preprint arXiv:2108.06144

        Parameters
        ----------
        outputs : Numpy Array
            Fused MultiSpectral img. Dimensions: H, W, Bands
        ms : Numpy Array
            MultiSpectral img. Dimensions: H, W, Bands
        pan : Numpy Array
            Panchromatic img. Dimensions: H, W, Bands
        sensor : str
            The name of the satellites which has provided the images.
        ratio : int
            Resolution scale which elapses between MS and PAN.
        dim_cut : int
            Cutting dimension for obtaining "valid" image to which apply the metrics

        Return
        ------
        R-SAM : float
            The R-SAM index

    """
    outputs, ms = resize_with_mtf(outputs, ms, pan, sensor, ratio, dim_cut)
    return SAM(outputs, ms)


def ReproQ2n(outputs, ms, pan, sensor, ratio=4, q_block_size=32, q_shift=32, dim_cut=0):
    """
        Reprojected Q2n.

        [Scarpa21]          Scarpa, Giuseppe, and Matteo Ciotola. "Full-resolution quality assessment for pansharpening.",
                            arXiv preprint arXiv:2108.06144

        Parameters
        ----------
        outputs : Numpy Array
            Fused MultiSpectral img. Dimensions: H, W, Bands
        ms : Numpy Array
            MultiSpectral img. Dimensions: H, W, Bands
        pan : Numpy Array
            Panchromatic img. Dimensions: H, W, Bands
        sensor : str
            The name of the satellites which has provided the images.
        ratio : int
            Resolution scale which elapses between MS and PAN.
        q_block_size : int
            The windows size on which calculate the Q2n index
        q_shift : int
            The stride for Q2n index calculation
        dim_cut : int
            Cutting dimension for obtaining "valid" image to which apply the metrics

        Return
        ------
        r_q2n : float
            The R-Q2n index

    """
    outputs, ms = resize_with_mtf(outputs, ms, pan, sensor, ratio, dim_cut)
    r_q2n, _ = Q2n(outputs, ms, q_block_size, q_shift)
    return r_q2n


def ReproQ(outputs, ms, pan, sensor, ratio=4, q_block_size=32, dim_cut=0):
    """
        Reprojected Q.

        [Scarpa21]          Scarpa, Giuseppe, and Matteo Ciotola. "Full-resolution quality assessment for pansharpening.",
                            arXiv preprint arXiv:2108.06144

        Parameters
        ----------
        outputs : Numpy Array
            Fused MultiSpectral img. Dimensions: H, W, Bands
        ms : Numpy Array
            MultiSpectral img. Dimensions: H, W, Bands
        pan : Numpy Array
            Panchromatic img. Dimensions: H, W, Bands
        sensor : str
            The name of the satellites which has provided the images.
        ratio : int
            Resolution scale which elapses between MS and PAN.
        q_block_size : int
            The windows size on which calculate the Q index
        dim_cut : int
            Cutting dimension for obtaining "valid" image to which apply the metrics

        Return
        ------
        r_q : float
            The R-Q index

    """
    outputs, ms = resize_with_mtf(outputs, ms, pan, sensor, ratio, dim_cut)
    r_q = Q(outputs, ms, q_block_size)
    return r_q


def ReproMetrics(outputs, ms, pan, sensor, ratio=4, q_block_size=32, q_shift=32, dim_cut=0):
    """
        Computation of all reprojected metrics.

        [Scarpa21]          Scarpa, Giuseppe, and Matteo Ciotola. "Full-resolution quality assessment for pansharpening.",
                            arXiv preprint arXiv:2108.06144

        Parameters
        ----------
        outputs : Numpy Array
            Fused MultiSpectral img. Dimensions: H, W, Bands
        ms : Numpy Array
            MultiSpectral img. Dimensions: H, W, Bands
        pan : Numpy Array
            Panchromatic img. Dimensions: H, W, Bands
        sensor : str
            The name of the satellites which has provided the images.
        ratio : int
            Resolution scale which elapses between MS and PAN.
        q_block_size : int
            The windows size on which calculate the Q2n and Q index
        q_shift : int
            The stride for Q2n index calculation
        dim_cut : int
            Cutting dimension for obtaining "valid" image to which apply the metrics

        Return
        ------
        r_q2n : float
            The R-Q2n index
        r_q : float
            The R-Q index
        R-SAM : float
            The R-SAM index
        R-ERGAS : float
            The R-ERGAS index

    """
    outputs, ms = resize_with_mtf(outputs, ms, pan, sensor, ratio, dim_cut)
    q2n, _ = Q2n(outputs, ms, q_block_size, q_shift)
    q = Q(outputs, ms, q_block_size)
    sam = SAM(outputs, ms)
    ergas = ERGAS(outputs, ms, ratio)
    return q2n, q, sam, ergas


def DRho(outputs, pan, sigma=4):
    """
        Spatial Quality Index based on local cross-correlation.

        [Scarpa21]          Scarpa, Giuseppe, and Matteo Ciotola. "Full-resolution quality assessment for pansharpening.",
                            arXiv preprint arXiv:2108.06144

        Parameters
        ----------
        outputs : Numpy Array
            Fused MultiSpectral img. Dimensions: H, W, Bands
        pan : Numpy Array
            Panchromatic img. Dimensions: H, W, Bands
        sigma : int
            The windows size on which calculate the Drho index; Accordingly with the paper it should be the
            resolution scale which elapses between MS and PAN.

        Return
        ------
        d_rho : float
            The d_rho index

    """
    half_width = ceil(sigma / 2)
    rho = np.clip(local_cross_correlation(outputs, pan, half_width), a_min=-1.0, a_max=1.0)
    d_rho = 1.0 - rho
    return np.mean(d_rho).item()
