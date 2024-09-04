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
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



class MLEM_3D_reconstruction:
    def __init__(self,
                 det_mask_dist,
                 distance_array,
                 MLEM_iters,
                 Mask,
                 Detector,
                 img_size_px,
                 PSF_style="NTHT",
                 transmission_prob=0.46
                 ):
        """
        Uses the 3D MLEM image reconstruction algorithm given in Mu 2006
        to reconstruct slices of a 3D object from a single detector image.
        ----------------------------------------------------------------

        Parameters
        ----------
        det_mask_dist: scalar float32.
            Distance between detector and mask in mm.
        detector_image : 2d numpy array, float32
            the detector image from which to reconstruct slices
        distance_array : 1d numpy array
            Array of Object to Mask distances for each slice in mm. E.g. 180 will reconstruct a slice
            at 180 mm
        MLEM_iters: int
            how many MLEM iterations to perform for each slice.
        Mask: MuraMaskObject
            Contains information about mask pattern, dimension, aperture hole size, thickness, etc.
        Detector: DetectorObject
            Contains pixel size, detector dimension, and its resolution.
        img_size_px: int
            how large each slice array will be, eg 256 -> 256x256 image
        transmission_prob: float
            Probability that a photon penetrates the mask without any interaction. In the Russo2023 setting it is 0.46.

       (optional) epsilon: float
            small value for replacing negative or 0 values. default = 10e-8
        (optional) PSF_style: either "NTHT" or "THT"
            whether the not-two-holes-touching or the two-holes-touching pattern should be used for reconstruction.
        """

        # Here, the latest detector image will be stored.
        self.detector_image = None

        # Distance between centers of mask and object and between detector and mask in mm:
        self.det_mask_dist = det_mask_dist

        # Those objects store all information about pixel size, dimension, mask pattern etc..
        self.Mask = Mask
        self.Detector = Detector

        # Reconstruction parameters:
        # At which distance do you want to reconstruct each slice?
        self.distance_array = np.array(distance_array).reshape(-1, )
        self.MLEM_iters = MLEM_iters  # How many iterations to use
        self.img_size_px = img_size_px
        self.epsilon = 10 ** -8
        self.PSF_style = PSF_style

        self.transmission_prob = transmission_prob
        # Assert that the given Detector object has the same amount of pixels:
        assert self.Detector.size_px == self.img_size_px, "Amount of pixels do not match!"

        # Empty arrays for storing the forward projections and the reconstructed slices:
        self.forward_pros = np.zeros((self.img_size_px, self.img_size_px, self.distance_array.shape[0]),
                                     np.float32)
        self.reconstructed_slices = np.zeros((self.img_size_px, self.img_size_px, self.distance_array.shape[0]),
                                             np.float32)
        self.mses = np.zeros((self.MLEM_iters,), np.float32)

        # New: Depending on if tensorflow is installed or not, the way the convolutions are calculated changes:
        # With TF and GPU: use tensorflow-based conv_in_fourier2tf as tf.function
        # Without TF: use the numpy-based conv_in_fouriernp2.
        self.conv2use = self.conv_in_fouriernp2

        # New: Depending on if the GPU is used or not, change the way the convolutions are calculated:
        # With GPU: use tensorflow-based conv_in_fourier2tf as tf.function
        # Without GPU: use the numpy-based conv_in_fourier2.  Should be much slower.
        if len(tf.config.list_physical_devices("GPU")) >= 1:
            print(type(self).__name__ + ": GPU found. Will use the tf.function conv_in_fourier2tf for reconstruction.")
            gpus = tf.config.experimental.list_physical_devices('GPU')
            [tf.config.experimental.set_memory_growth(g, True) for g in gpus]
            self.conv2use = self.conv_in_fourier2tf

    def PSF_rs_mask_resolution_no_limit(self, z):
        """
        This one assumes pinhole size is the same as the gap between two pinholes. Then simple bilinear resizing can be
        used.
        """
        # Find magnification ratio
        m = (z + self.det_mask_dist) / z

        # Calculate size of the shadow in mm, then convert to pixels using the detector resolution
        shadow_size_mm = self.Mask.size_mm * m
        shadow_size_px = int(np.round(shadow_size_mm / self.Detector.resolution_mm))

        mask_pattern = self.Mask.mask_array

        if self.PSF_style == "THT":
            # Shift so that empty rows and columns are top left:
            if mask_pattern[-1, :].sum() == 0:
                mask_pattern = np.roll(mask_pattern, 1, axis=0)
            if mask_pattern[:, -1].sum() == 0:
                mask_pattern = np.roll(mask_pattern, 1, axis=1)
            # Delete NTHT rows and cols:
            tht_cp = mask_pattern.copy()[1::2, 1::2]
            # DOne: MAKE THIS BACK AGAIN!
            # tht_cp = np.pad(tht_cp,[[0, 0,], [0, 1]])
            psf = tf.image.resize(tht_cp[None, ..., None], [shadow_size_px, shadow_size_px], "bilinear")

        elif self.PSF_style == "NTHT":
            psf = tf.image.resize(mask_pattern[None, ..., None], [shadow_size_px, shadow_size_px])

        psf = np.squeeze(psf)

        # IMPORTANT HERE: no transposition needed!
        return psf

    def crop_PSFs_to_max_needed_size(self, psfs):
        """
        crops the generated PSFs to the maximum needed size. That is twice the detector size minus 1.
        """
        CROP_SIZE = self.Detector.size_px * 2 - 1
        for i in range(len(psfs)):
            y, x = psfs[i].shape
            if x > CROP_SIZE or y > CROP_SIZE:
                startx = x // 2 - (CROP_SIZE // 2)
                starty = y // 2 - (CROP_SIZE // 2)
                psfs[i] = psfs[i][starty:starty + CROP_SIZE, startx:startx + CROP_SIZE]
            assert psfs[i].shape[0] <= CROP_SIZE, "Something went wrong in crop_PSFs_to_max_needed_size"
        return psfs

    def run_conv_exp(self):
        """ This function should show how the Convolution in space and fourier domain work and where differences are."""
        from reco import plot, plot_all_slices
        psf = self.Mask.mask_array.copy()
        p_shape = psf.shape[0]
        PSF = np.fft.rfft2(psf, (self.Detector.size_px + p_shape - 1, self.Detector.size_px + p_shape - 1))
        PSF = tf.cast(PSF, tf.complex64)

        test_img = np.zeros((self.Detector.size_px, self.Detector.size_px), np.float32)
        test_img[50, 200] = 100
        rtf = self.conv_in_fourier2tf(test_img, PSF)
        rnp = self.conv_in_fouriernp2(test_img, PSF)
        rspace = self.conv_in_space(test_img, psf)

        plot(rtf, "Conv in Fourier domain (tf)")
        plot(rnp, "Conv in Fourier domain (numpy)")
        plot(rspace, "Conv in space domain")
        plot(rtf * [rtf < self.epsilon], "Fourier (tf) where values are below " + str(self.epsilon))

        plot(rtf - rspace, "Difference Fourier (tf) - Space\n=> A change in the corresponding pattern is visible")
        plot(rtf - rnp, "Difference Fourier (tf) - Fourier (np)\n=> Virtually no difference between np and tf")

        # rtf_mirroredPSF = self.conv_in_fourier2tf(test_img, np.flip(PSF, (0, 1)))
        # => flipping in Fourier domain does not make sense:
        psf_mirr = np.flip(psf, (0, 1))
        PSF_mirr = np.fft.rfft2(psf_mirr, (self.Detector.size_px + p_shape - 1, self.Detector.size_px + p_shape - 1))
        PSF_mirr = tf.cast(PSF_mirr, tf.complex64)

        rtf_mirroredPSF = self.conv_in_fourier2tf(test_img, PSF_mirr)

        plot(rtf_mirroredPSF, "Conv in Fourier domain (tf) with mirrored(psf)")
        plot(rtf_mirroredPSF - rspace,
             "Difference Fourier (tf) with mirrored(psf) - Space\n=> Structurally much better")

        diff = np.array(rtf_mirroredPSF - rspace)
        plt.boxplot(np.reshape(diff, -1))
        plt.title("L2 mean: %.2f, L1 mean: %.2f" % (np.mean(diff ** 2), np.mean(np.abs(diff))))
        plt.show()

        f = plot_all_slices([rtf, rnp, rspace, rtf_mirroredPSF],
                            ["Fourier (ft)", "Fourier (numpy)", "space domain", "ft with mirrored psf"], imgs_axis=0)
        f.suptitle(
            "Solution: mirror the PSF before Transformation into FOURIER domain!\nSmall artefacts will still remain",
            fontsize="small")
        f.set_dpi(300)
        f.show()

    def reconstruct_new(self, detector_image, f0=None):
        """
        This is the main reconstruction function. 
        Reference Mu and Liu 2006 for the 3D MLEM algorithm
        Returns 3D array of reconstructed slices, e.g. 256x256xN

        detector_image: 2d numpy array, float32.
            Reconstruction is based on this input image. Image is supposed to be UNFLIPPED and
            no final flipping takes place!!
        f0: initial guess, if None it is set to a small constant value based on the detector_image.
        """

        # Some aliases to make coding easier:
        distance_array = self.distance_array
        num_iters = self.MLEM_iters
        img_size_px = self.img_size_px
        epsilon = self.epsilon
        detector_image = np.squeeze(detector_image)
        t = self.transmission_prob
        self.detector_image = detector_image
        self.mses = np.zeros((self.MLEM_iters,), np.float32)
        assert self.detector_image.shape[0] == self.detector_image.shape[1]

        start_time = time.process_time()

        # Initialize initial guess
        N = distance_array.shape[0]
        if f0 is None:
            f_array = 1.0 * np.ones((img_size_px, img_size_px, N), np.float32)
        else:
            f_array = f0.copy()

        # Precalculate array of PSFs for each desired reconstruction slice:
        # Here, reshaping of the entire mask pattern is used. This only works if the pinholes are as large as
        # the gap between to pinholes.
        psf_array = []
        for i in range(N):
            temp_PSF = self.PSF_rs_mask_resolution_no_limit(distance_array[i])
            psf_array.append(temp_PSF)

        # Crop PSFs to maximally necessary size
        psf_array = self.crop_PSFs_to_max_needed_size(psf_array)

        # Zero-pad and transform PSFs into Fourier domain:
        # IMPORTANT CHANGE: The PSF must be mirrored before being transformed into Fourier domain!!
        for i in range(N):
            p_shape = psf_array[i].shape[0]
            psf_array[i] = np.flip(psf_array[i], (0, 1))
            psf_array[i] = np.fft.rfft2(psf_array[i],
                                        (self.Detector.size_px + p_shape - 1, self.Detector.size_px + p_shape - 1))
            # Complex128 and complex64 is very similar in speed. With complex128 I need to convert
            # _input to float64 as well, which I don't like.
            psf_array[i] = tf.cast(psf_array[i], tf.complex64)

        print("Calculated, cropped and stored all PSFs. Took %.2fs" % (time.process_time() - start_time))

        # Normalization images obtained by convolution the PSF with all 1s:
        all_ns = np.zeros((img_size_px, img_size_px, N), np.float32)
        for i, p in enumerate(psf_array):
            this_w = self.conv2use(np.ones_like(detector_image), p)
            all_ns[..., i] = this_w

        # Calculate forward projections. They are accessible even after the reconstruction:
        # Initialize here for all slices and then only the lastly updated slice:
        for j in range(N):
            temp_forward = self.conv2use(f_array[:, :, j], psf_array[j])
            temp_forward = np.flip(temp_forward, axis=(0, 1))
            temp_forward = (1 - t) * temp_forward + t * np.sum(f_array[:, :, j])
            self.forward_pros[..., j] = temp_forward

        print("Calculated all initial forward projections. Start main loop now.")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Main loop starts here ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for k in range(num_iters):
            print("Iteration %i" % k)
            for i in range(N):
                # Sum all forwards projections except the current one:
                all_except_i = np.arange(N) != i
                bracket_sum = np.sum(self.forward_pros[:, :, all_except_i], 2)

                # Subtract out of focus contribution to p:
                numerator = detector_image - bracket_sum

                # Replace all negative or 0 values with epsilon to avoid negative pixels
                numerator[numerator < epsilon] = epsilon

                # This is the Forward Projection
                denominator = self.conv2use(f_array[:, :, i], psf_array[i]) + epsilon
                denominator = (1 - t) * denominator + t * np.sum(f_array[:, :, i])
                denominator = np.flip(denominator, axis=(0, 1))

                # Calculate the right hand side of the value in brackets
                bracket_val_right = numerator / denominator

                # Calculate the entire value in the brackets and rotate by 180° to represent the correlation:
                bracket_val_right = np.flip(bracket_val_right, axis=(0, 1))
                bracketval_total = self.conv2use(bracket_val_right, psf_array[i])

                # Perform element-wise division then element-wise multiplication
                f_array[:, :, i] = (f_array[:, :, i] / (all_ns[..., i])) * bracketval_total

                # Update forward projection:
                temp_forward = self.conv2use(f_array[..., i], psf_array[i])
                temp_forward = np.flip(temp_forward, axis=(0, 1))
                temp_forward = (1 - t) * temp_forward + t * np.sum(f_array[:, :, i])
                self.forward_pros[..., i] = temp_forward

            self.mses[k] = self.mse(detector_image, np.sum(self.forward_pros, 2))

        final_time = time.process_time() - start_time
        print("Reconstruction of %i slices with %i iterations completed in %.3fs with PSF_style=%s"
              % (N, num_iters, final_time, self.PSF_style))

        # Store the reconstructed slices for later use, but also return them to the user
        self.reconstructed_slices = f_array.copy()

        # Return reconstructed slices and the MSE over the iterations
        return self.reconstructed_slices, self.mses

    @staticmethod
    def mse(img1, img2):
        """ Return the mean squared error of two 2D images. """
        e = img1 - img2
        mse = np.mean(np.square(e))
        return mse

    def conv_in_fouriernp2(self, _input, _Filter):
        """ Difference to conv_in_fouriernp: The _Filter is already zero-padded and in Fourier domain!
        NEW: Fourier calculations take place in 64 bit quantification. Not anymore, it is useless, since numpy
        internally calculates in complex128 anyways.
        """
        size_input = np.shape(_input)[0]
        size_Filter = np.shape(_Filter)[0]

        # Transform into Fourier domain:
        S = np.fft.rfft2(_input, (size_Filter, size_Filter))
        # NEW: Downsizing to complex64 bit. No, don't. complex128 is faster.
        F = _Filter
        R = S * F
        # Transform back and crop the center:
        r = np.fft.irfft2(R)
        start_x = np.shape(r)[0] // 2 - (size_input // 2)
        start_y = np.shape(r)[1] // 2 - (size_input // 2)
        r_cropped = r[start_y:start_y + size_input, start_x:start_x + size_input]
        r_cropped = r_cropped.astype(np.float32)
        return r_cropped

    @tf.function(reduce_retracing=True)
    def conv_in_fourier2tf(self, _input, _Filter):
        """ Difference to conv_in_fourier: The _Filter is already zero-padded and in Fourier domain!
        NEW: Fourier calculations take place in 64 bit quantification. Not anymore, it is useless, since numpy
        internally calculates in complex128 anyway.
        """
        print("TF Tracing takes place with, ", _input.shape, _input.dtype, _Filter.shape, _Filter.dtype)
        size_input = tf.shape(_input)[0]
        size_Filter = tf.shape(_Filter)[0]

        # Transform into Fourier domain:
        S = tf.signal.rfft2d(_input, (size_Filter, size_Filter))
        F = _Filter
        R = S * F
        # Transform back and crop the center:
        r = tf.signal.irfft2d(R)

        # Debugging:
        """
        if start_x != tf.cast(tf.round(tf.shape(r)[0] / 2 - (size_input / 2)), tf.int32) or start_y != tf.cast(tf.round(tf.shape(r)[1] / 2 - (size_input / 2)), tf.int32):
            tf.print(start_x, start_y)
            tf.print("Rounded: True top-left corner: %.2f, %.2f: " % (tf.round(tf.shape(r)[0] / 2 - (size_input / 2)),
                                                                  tf.round(tf.shape(r)[1] / 2 - (size_input / 2))))
            import matplotlib.patches as patches
            fig, ax = plt.subplots()
            ax.imshow(r)
            ax.set_title(str(r.shape))
            rect = patches.Rectangle((tf.shape(r)[0] // 2 - (size_input // 2),
                                      tf.shape(r)[1] // 2 - (size_input // 2)),
                                     size_input, size_input, linewidth=1, edgecolor='r', facecolor='none')
            rect2 = patches.Rectangle((tf.cast(tf.round(tf.shape(r)[0] / 2 - (size_input / 2)), tf.int32),
                                       tf.cast(tf.round(tf.shape(r)[1] / 2 - (size_input / 2)), tf.int32)),
                                      size_input, size_input, linewidth=1, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            ax.add_patch(rect2)
            fig.tight_layout()
            fig.show()
        """

        # Way of rounding does not matter:
        start_x = tf.shape(r)[0] // 2 - (size_input // 2)
        start_y = tf.shape(r)[1] // 2 - (size_input // 2)

        r_cropped = r[start_y:start_y + size_input, start_x:start_x + size_input]
        r_cropped = tf.cast(r_cropped, tf.float32)
        return r_cropped

    @staticmethod
    def conv_in_fouriernp(_input, _filter):
        """ This function calculates the linear 2d convolution of the two inputs inside the Fourier domain.
        To yield the result of the LINEAR convolution of two images, the shape of the Fourier transformed is chosen.
        The result might be one or two pixels shifted due to odd image sizes. Due to numerical issues, small negative
        numbers can occur. They are dealt with somewhere else!
        ATTENTION: The filter must not be zero-padded!
        """
        size_input = np.shape(_input)[0]
        size_filter = np.shape(_filter)[0]

        # Transform into Fourier domain:
        S = np.fft.rfft2(_input, (size_input + size_filter - 1, size_input + size_filter - 1))
        F = np.fft.rfft2(_filter, (size_input + size_filter - 1, size_input + size_filter - 1))
        R = S * F
        # Transform back and crop the center:
        r = np.fft.irfft2(R).astype(np.float32)
        start_x = np.shape(r)[0] // 2 - (size_input // 2)
        start_y = np.shape(r)[1] // 2 - (size_input // 2)
        r_cropped = r[start_y:start_y + size_input, start_x:start_x + size_input]

        return r_cropped

