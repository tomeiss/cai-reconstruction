#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By:       Tobias Meissner
# Created Date:     20.08.2024
# Date Modified:    04.09.2024
# Python Version:   3.8.18

# Dependencies:     NumPy (1.24.3), Tensorflow (2.10.1), SciPy (1.10.1), matplotlib (3.7.2), json (2.0.9),
#                   tifffile (2023.4.12), PIL (10.0.1)

# License:          GNU GENERAL PUBLIC LICENSE Version 3
#
# These scripts belongs to the publications about Coded Aperture from Tobias Meissner, Laura Antonia Cerbone,
# Paolo Russo, Werner Nahm, and Juergen Hesser. More details can
# be found there.
# ----------------------------------------------------------------------------


import os
import time
import argparse

import json
import tifffile
import numpy as np
from glob import glob
import tensorflow as tf
from argparse import Namespace
from PIL import Image
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
from scipy.ndimage import gaussian_filter, median_filter

from MaskObject import MuraMaskObject
from DetectorObject import DetectorObject
from MLEM_reconstruction import MLEM_3D_reconstruction


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Defines my parser and turns the arguments into the global global_args
def defineParser():
    parser = argparse.ArgumentParser(description='3D CAI Reconstruction')
    parser.add_argument('--files', type=str, required=False,
                        default="",
                        help='Files to reconstruct, either a REGEX string or a single file. Files must be in either '
                             'npy or tiff format.')
    parser.add_argument('--preprocess', type=str2bool, required=False,
                        default=False,
                        help='If detector images should be preprocessed before reconstruction or not. Preprocessing '
                             'contains outlier rejection based on the 1st and 99th percentile and '
                             'replacement with the 3x3 median value, plus Gaussian smoothing with sigma=1px.')
    parser.add_argument('--mlem3d', type=str2bool, nargs='?', default=False, required=False,
                        help='Reconstruct with 3D-MLEM')
    parser.add_argument('--mlem_iters', type=int, default=1, required=False,
                        help='How many iterations of MLEM should be carried out.')
    parser.add_argument('--mura_decoding', type=str2bool, nargs='?', default=False, required=False,
                        help='Reconstruct with MURA Decoding')
    parser.add_argument('--accorsi_decoding', type=str2bool, nargs='?', default=False, required=False,
                        help='Reconstruct with Accorsi Decoding where only the central pattern is used')
    parser.add_argument('--tobi_decoding', type=str2bool, nargs='?', default=False, required=False,
                        help='Reconstruct with Tobi Decoding: Only use central pattern, but does uses NN upsampling'
                             'instead of sparse upsampling of the decoding pattern, as in Accorsi Decoding.')
    parser.add_argument('--dist', "--list", type=json.loads, required=False, default=[5, 10, 15, 20, 25], nargs='+',
                        help='An array of floats which denote the distances of the in-focus planes measured '
                             'from coded aperture mask.')
    parser.add_argument('--rot180', type=str2bool, nargs='?', default=False,
                        help='Rotate output by 180 degree or not.')
    # Detector:
    parser.add_argument('--hd', type=float, default=14.08, help='Side length of detector in mm.')
    parser.add_argument('--pixels', type=int, default=256, help='Amount of pixels of the detector.')
    # Mask:
    parser.add_argument('--hm', type=float, default=9.92, help='Side length of mask in mm.')
    parser.add_argument('--r', type=float, default=0.08 / 2.0, help='Pinhole RADIUS in mm.')
    parser.add_argument('--t', type=float, default=0.11, help='Mask thickness in mm.')
    parser.add_argument('--mask_file', type=str, required=False,
                        default="./mask_pattern.tif",
                        help='Path and TIFF file that contains the entire mask patter, i.e. the entire 2x2 arrangement if used. '
                             'Must be binary, where 1 represents an opening and 0 opaque material.')
    # Geometry:
    parser.add_argument('--b', type=float, default=20, help='Detector-to-mask-distance in mm.')
    global_args = parser.parse_args()
    return global_args


def central_crop(img, cp):
    """
    Crops an image to the central portion.
    cp: crop portion. Can be either a scalar or a 2D value when columns and rows should be cropped in not the same ratio
    """
    cp = np.array(cp)
    img = np.array(img)
    img_shape = img.shape
    img = img.squeeze()

    if np.prod(cp.shape) == 1:
        cp = np.array([cp, cp])
    # Trim
    if np.greater(cp, 1.0).any():
        cp = np.clip(cp, 0.0, 1.0)

    center = [k / 2 for k in img.shape]
    new_size = [this_shape * this_cp for this_shape, this_cp in zip(img.shape, cp)]

    four_corners1 = int(center[0] - new_size[0] / 2)
    four_corners2 = int(center[0] + new_size[0] / 2)
    four_corners3 = int(center[1] - new_size[1] / 2)
    four_corners4 = int(center[1] + new_size[1] / 2)
    new_img = img[four_corners1:four_corners2, four_corners3:four_corners4]

    if len(img_shape) == 4:
        new_img = new_img[None, ..., None]
    return new_img


def plot(img, title="", ticks=True, colorbar=True, cmap=None, clim=None, filename="", dpi=None, cb_format=None,
         fontsize=None, aspect_ratio=None):
    """ Plot a sinlge image with several options for customization. This function might impede other plotting processed
     when called. """
    plt.clf()
    plt.cla()
    plt.imshow(np.squeeze(img), cmap=cmap)
    if aspect_ratio == "equal":
        plt.axis("equal")
    if ticks == False:
        plt.xticks([])
        plt.yticks([])
    if colorbar:
        cb = plt.colorbar(format=cb_format)
    if clim:
        plt.clim(clim)
    plt.title(str(title), fontsize=fontsize)

    if fontsize:
        cb.ax.tick_params(labelsize=fontsize)
    if dpi:
        plt.gcf().set_dpi(dpi)
    plt.tight_layout()
    fig = plt.gcf()
    if filename == "":
        plt.show()
    else:
        plt.savefig(filename)
    return fig


def plot_all_slices(f_array, titles=None, suptitle="", imgs_axis=2, cmap=None):
    """ Plots a 3D array as slices in a multi-image window. """
    plt.clf()
    plt.cla()
    plt.close()
    if imgs_axis == 2:
        axis_exc = np.delete(np.arange(len(np.shape(f_array))), 2).tolist()
        f_array = np.transpose(f_array, [imgs_axis, ] + axis_exc)

    n = len(f_array)
    mn = np.nanmin([np.nanmin(k) for k in f_array])
    mx = np.nanmax([np.nanmax(k) for k in f_array])

    if titles is None:
        titles = ["" for _ in range(n)]

    # Plots the stack of reconstructed slices as a quadratic plot
    plot_c = np.ceil(np.sqrt(n)).astype(int)
    plot_r = np.ceil(n / plot_c).astype(int)
    # plot_r = np.floor(n / (plot_c - 1)).astype(int)
    fig, axs = plt.subplots(plot_r, plot_c, figsize=(2.0 * plot_c, 2.0 * plot_r))
    _axs = np.ravel(axs)

    for i in range(len(_axs)):
        if i < n:
            im = _axs[i].imshow(f_array[i], vmin=mn, vmax=mx, cmap=cmap)
            _axs[i].set_title(titles[i])
        _axs[i].set_xticks([])
        _axs[i].set_yticks([])
        _axs[i].axis("off")

    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.colorbar(im, ax=_axs.tolist())
    return fig


def convert_mask_to_decoding_pattern(mask_pattern):
    """
     WARNING:
     This is only inteded for the Russo2023 mask pattern and does not necessarily work for other patterns.
     """
    central_decod = central_crop(mask_pattern, 0.5)
    # Shift so that empty beginning rows or columns will be on the TOP and LEFT side:
    # (Because then, the 31st pixel is the central pixel.)
    if central_decod[-1, :].sum() == 0:
        central_decod = np.roll(central_decod, 1, axis=0)
    if central_decod[:, -1].sum() == 0:
        central_decod = np.roll(central_decod, 1, axis=1)
    # Delete NTHT rows and cols:
    tht_cp = central_decod.copy()[1::2, 1::2]
    # Change central pixel from +1 to 0:
    tht_cp[tht_cp.shape[0] // 2, tht_cp.shape[1] // 2] = 0.0
    # Switch non-pinholes s to -1, by finding the original zero filling rows and cols:
    tht_cp[tht_cp == 0] = -1.0
    # Fill it back
    central_decod[1::2, 1::2] = tht_cp
    # Shift central pattern, so that it is a classical MURA pattern:
    central_decod = np.roll(central_decod, [central_decod.shape[0] // 2, central_decod.shape[1] // 2],
                            axis=[0, 1])
    return central_decod


def main(p):
    # Convert distance list to numpy array
    p.dist = np.array(p.dist).astype(np.float32)
    print("Parameters given to main: ", p)

    # Critical z, where less than one central mask pattern is projected onto the detector:
    z_krit = (p.b * p.hm / 2) / (p.hd - p.hm / 2)
    print("In this setting the closest we can unambiguously image is z_krit: %.2f" % z_krit)

    # Search for file names:
    fs = glob(p.files)
    fs = [f.replace("\\", "/") for f in fs]

    # Load files:
    imgs_in, img = [], None
    for f in fs:
        if f.endswith(".npy"):
            img = np.load(f)
        elif f.endswith(".tif") or f.endswith(".tiff"):
            img = tifffile.imread(f)
        img = np.squeeze(img).astype(np.float32)
        img = img[None, ...]
        imgs_in.append(img)
    imgs_in = np.concatenate(imgs_in)
    print("Loaded %i file(s)." % len(imgs_in))

    # Apply simple outlier detection with 1%, 99% percentile:
    if p.preprocess is True:
        print("Simple 1st and 99th percentile outlier removal is performed.")
        for k in range(len(imgs_in)):
            p1, p99 = np.percentile(imgs_in[k], [1, 99])
            mask = np.zeros_like(imgs_in[k], np.float32)
            mask[imgs_in[k] > p99] = 1.0
            mask[imgs_in[k] < p1] = 1.0
            med_filt = median_filter(imgs_in[k], size=3)
            imgs_in[k] = mask * med_filt + (1 - mask) * imgs_in[k]
        # Maybe someone want to apply a little Gaussian Blur?
        print("Do some Gaussian blurring (sigma=1.0).")
        imgs_in = np.array([gaussian_filter(img, 1.0) for img in imgs_in])

    Detector = DetectorObject(size_px=p.pixels, size_mm=p.hd)
    dummy_rank = 5
    Mask = MuraMaskObject(dummy_rank, size_mm=p.hm, thickness=p.t, aperture_shape="circle",
                          aperture_radius_mm=p.r)
    # Load given mask pattern:
    mask_pattern = tifffile.imread(p.mask_file)
    Mask.mask_array = mask_pattern
    plot(Mask.mask_array, filename="./mask_pattern_used.png").clf()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 3D MLEM Reconstruction ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if p.mlem3d:
        print("You chose 3D MLEM reconstruction")
        for img_i in range(len(imgs_in)):
            this_img = imgs_in[img_i]
            this_f = fs[img_i].replace("\\", "/")

            print("Processing %s" % this_f)
            print("We use the THT pattern to avoid periodic reconstruction noise.")
            mlem_reco = MLEM_3D_reconstruction(det_mask_dist=p.b,
                                               distance_array=p.dist, MLEM_iters=p.mlem_iters, Mask=Mask,
                                               Detector=Detector,
                                               img_size_px=p.pixels, PSF_style="THT",
                                               transmission_prob=p.transmission_prob)

            # Reconstruct image:
            mlem_reco.reconstruct_new(this_img)

            # Because MLEM_3D_reconstruction does it for us, we need to reverse it if 180° is NOT desired:
            if not p.rot180:
                mlem_reco.reconstructed_slices = np.rot90(mlem_reco.reconstructed_slices, 2, axes=(0, 1))

            # Save summed forward projections:
            base_name = this_f.split("/")[-1]
            base_name = "./reconstruction/" + base_name[0:base_name.rfind(".")]
            new_name = base_name + "_3D-MLEM_forward_i%02i_Sl%02i.png" % (p.mlem_iters, len(p.dist))
            plot(mlem_reco.forward_pros.sum(2), new_name.split("/")[-1], filename=new_name, dpi=500).clf()
            tifffile.imwrite(new_name.replace("png", "tif"), mlem_reco.forward_pros.sum(2))
            # Save forward projections:
            new_name = base_name + "_3D-MLEM_all_forward_i%02i_Sl%02i.png" % (p.mlem_iters, len(p.dist))
            plot_all_slices(mlem_reco.forward_pros, mlem_reco.distance_array.astype(str),
                            suptitle=new_name.split("/")[-1]).savefig(new_name, dpi=500)
            plt.close()
            # Save all reconstructed slices in a single plot:
            new_name = base_name + "_3D-MLEM_all_recos_i%02i_Sl%02i.png" % (p.mlem_iters, len(p.dist))
            plot_all_slices(mlem_reco.reconstructed_slices, mlem_reco.distance_array.astype(str),
                            suptitle=new_name.split("/")[-1]).savefig(new_name, dpi=500)
            plt.close()
            # Save the MSE plot over iterations
            fig, ax = plt.subplots(1, 1)
            ax.semilogy(np.arange(len(mlem_reco.mses)), mlem_reco.mses)
            ax.set_xlabel("Iterations")
            ax.set_ylabel("MSE")
            ax.grid()
            fig.tight_layout()
            fig.savefig(new_name.replace("all_recos", "mse_over_itits"), dpi=500)
            plt.close(fig)

            if p.save_each_img:
                # Save the files as tiff and as png plot:
                for k in range(len(p.dist)):
                    d = p.dist[k]
                    r = mlem_reco.reconstructed_slices[..., k]

                    # Write out tif file:
                    new_name = base_name + "_3D-MLEM_i%02i_Sl%02i_d%06.2f.tif" % (p.mlem_iters, len(p.dist), d)
                    tifffile.imwrite(new_name, r)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tobi Decoding ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if p.mura_decoding:
        print("You chose MURA Decoding")

        # Convert central mask pattern to decoding patter:
        central_decod = convert_mask_to_decoding_pattern(Mask.mask_array)

        tifffile.imwrite("./reconstruction/mura_decoding_pattern_used.tif", central_decod.astype(np.float32))
        for img_i in range(len(imgs_in)):
            this_img = imgs_in[img_i]
            this_f = fs[img_i].replace("\\", "/")

            all_recos = []
            start_time = time.process_time()

            for d in p.dist:
                # Magnification factor:
                m = (d + p.b) / d
                # Half of it because we care only about central part:
                proj_mask_px_hlf = int(np.round((m * p.hm) / (p.hd / p.pixels) / 2))

                # Enlarge decoding pattern to that size:
                pil_decod = Image.fromarray(central_decod)
                rs_decod = pil_decod.resize((proj_mask_px_hlf, proj_mask_px_hlf), Image.Resampling.NEAREST)
                rs_decod = np.array(rs_decod)

                if proj_mask_px_hlf > p.pixels:
                    print("WARNING: Projected CENTRAL mask pattern is larger than detector: %i and %i at distance %.2f"
                          % (proj_mask_px_hlf, p.pixels, d))
                    img_pad = np.NAN * np.ones_like(rs_decod)
                else:
                    img_pad = this_img.copy()
                # Crop central part of detector and of rs_deco
                img_crop = central_crop(img_pad, proj_mask_px_hlf / img_pad.shape[0])
                # And then use circular convolution: flipping is not really necessary
                r_tobi = np.fft.ifft2(np.fft.fft2(img_crop) * np.fft.fft2(rs_decod))
                r_tobi = np.real(r_tobi).astype(np.float32)

                # Rotate by 180° if desired:
                if p.rot180:
                    r_tobi = np.rot90(r_tobi, 2)
                all_recos.append(r_tobi)

                if p.save_each_img:
                    # Write out tif file:
                    base_name = this_f.split("/")[-1]
                    base_name = "./reconstruction/" + base_name[0:base_name.rfind(".")]
                    new_name = base_name + "_MURADecoded_d%06.2f.tif" % d
                    tifffile.imwrite(new_name, r_tobi)

            print("Reconstruction of %i slices completed in %.3fs" % (len(p.dist), time.process_time() - start_time))

            # Save all reconstructed slices in a single plot:
            base_name = this_f.split("/")[-1]
            base_name = "./reconstruction/" + base_name[0:base_name.rfind(".")]
            new_name = base_name + "_MURADecoded_all_recos_Sl%02i.png" % len(p.dist)
            plot_all_slices(all_recos, p.dist.astype(str), imgs_axis=0,
                            suptitle=new_name.split("/")[-1]).savefig(new_name, dpi=500)
            plt.close()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Accorsi Decoding ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if p.accorsi_decoding:
        # Planar CAI reconstruction from Accorsi's dissertation from 2001 p.246
        # Main difference: it works with a rounded alpha and does a sparse upsampling of the decoding pattern.
        # Especially the last might be known as "delta decoding".
        print("You chose Accorsi Decoding")

        central_decod = convert_mask_to_decoding_pattern(Mask.mask_array)
        tifffile.imwrite("./reconstruction/accorsi_decoding_pattern_used.tif", central_decod.astype(np.float32))

        for img_i in range(len(imgs_in)):
            this_img = imgs_in[img_i]
            this_f = fs[img_i].replace("\\", "/")

            print("Processing %s" % this_f)
            all_recos = []
            for d in p.dist:
                dect_img = this_img.copy()
                m = (d + p.b) / d
                rm, cm = central_decod.shape
                # DIFFERENT NOW: CONVERT EACH DISTANCE TO ALPHA FIRST:
                alpha = m * (Mask.aperture_radius_mm * 2) / Detector.resolution_mm
                alphai = int(np.round(alpha))
                proj_mask_px_hlf = int(np.round((m * p.hm) / (p.hd / p.pixels) / 2))
                # print(alpha, alphai)
                # "Upsample" decoding pattern to a very sparse enlarged version of itself.
                rs_decod = np.zeros((alphai * central_decod.shape[0], alphai * central_decod.shape[0]), np.float32)
                rs_decod[::alphai, ::alphai] = central_decod

                if proj_mask_px_hlf > p.pixels:
                    print("WARNING: Projected CENTRAL mask pattern is larger than detector: %i and %i at distance %.2f"
                          % (proj_mask_px_hlf, p.pixels, d))
                    dect_img = np.NAN * np.ones_like(rs_decod)
                else:
                    dect_img = this_img.copy()

                rd, cd = dect_img.shape
                # From Accorsi:
                # hs, vs = 0, 0
                # xi, yi = np.mgrid[(cd - alpha * cm) // 2: (cd + alpha * cm) / 2:alpha / alphai,
                #          (rd - alpha * rm) // 2: (rd + alpha * rm) / 2: alpha / alphai]
                # xi = hs + xi[0:cm * alphai, 0:cm * alphai]
                # yi = vs + yi[0:rm * alphai, 0: rm * alphai]
                # My interpretation of it: central crop of projected central pattern size,
                # up-sampled to same resolution:
                xq = np.linspace(- alpha * cm / 2, alpha * cm / 2, rs_decod.shape[0])
                yq = np.linspace(- alpha * rm / 2, alpha * rm / 2, rs_decod.shape[1])
                X, Y = np.meshgrid(xq, yq)
                pq = np.array([X.flatten(), Y.flatten()]).transpose()
                # Grid of entire detector image. It includes -/+.1 because xq and yq can be upto 1 pixel,
                # larger since it is based on the unrounded alpha
                xg, yg = np.linspace(-rd / 2 - 1, rd / 2 + 1, rd), np.linspace(-cd / 2 - 1, cd / 2 + 1, cd)
                dect_croped_rs = interpn((xg, yg), dect_img, xi=pq, method="linear", bounds_error=False)
                # Reshape and transpose due to (x,y) <-> (row, col)
                dect_croped_rs = dect_croped_rs.reshape(rs_decod.shape).transpose()

                # And then use circular convolution: the flipping of the decoding is not necessary:
                r_acc = np.fft.ifft2(np.fft.fft2(dect_croped_rs) * np.fft.fft2(rs_decod))
                r_acc = np.real(r_acc).astype(np.float32)

                # Rotate by 180° if desired:
                if p.rot180:
                    r_acc = np.rot90(r_acc, 2)
                all_recos.append(r_acc)

                if p.save_each_img:
                    # Write out tif file:
                    base_name = this_f.split("/")[-1]
                    base_name = "./reconstruction/" + base_name[0:base_name.rfind(".")]
                    new_name = base_name + "_AccorsiDecoded_d%06.2f.tif" % d
                    tifffile.imwrite(new_name, r_acc)

            # Save all reconstructed slices in a single plot:
            base_name = this_f.split("/")[-1]
            base_name = "./reconstruction/" + base_name[0:base_name.rfind(".")]
            new_name = base_name + "_AccorsiDecoded_all_recos_Sl%02i.png" % len(p.dist)
            plot_all_slices(all_recos, p.dist.astype(str), imgs_axis=0,
                            suptitle=new_name.split("/")[-1]).savefig(new_name, dpi=500)
            plt.close()


if __name__ == "__main__":
    # GPU stuff:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = tf.config.experimental.list_physical_devices('GPU')
    [tf.config.experimental.set_memory_growth(g, True) for g in gpus]

    p = defineParser()

    # If you want to use this script via a shell, don't let p be overwritten here:
    p = dict()
    # Chose input files and if preprocessing should be carried out or not
    p["files"] = "./x00y00z50_Minipix_MC_Am241_1mm_MM_1B_001.tif"
    p["preprocess"] = True

    # Reconstruction methods etc.:
    p["accorsi_decoding"]   = True
    p["mura_decoding"]      = True
    p["mlem3d"]             = True
    p["dist"] = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    p["mlem_iters"] = 40

    # Process reconstructions
    p["save_each_img"] = True
    p["rot180"] = False

    # Geometric parameters:
    p["b"] = 20.0
    p["hd"] = 14.08
    p["pixels"] = 256
    p["r"] = 0.08 / 2.0
    p["hm"] = 9.92
    p["t"] = 0.11
    p["mask_file"] = "./mask_pattern_Russo2023_NTHT_MURA_rank31.tif"
    p["transmission_prob"] = 0.46

    main(Namespace(**p))
