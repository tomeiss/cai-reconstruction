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

class DetectorObject:
    def __init__(self, size_px=256, size_mm=14.1):
        self.size_px = size_px  # N, pixel size of detector (NxN): always square
        self.size_mm = size_mm  # height of detector
        self.resolution_mm = self.size_mm / self.size_px  # pixel resolution
