# cai-reconstruction
This repository contains several commonly used reconstruction for Coded Aperture Imaging (CAI) and is capable of both planar reconstruction and 3D reconstruction.

If you decide to use this code, please cite the following publication:
*Meißner, T., Cerbone, L. A., Russo, P., Nahm, W., & Hesser, J. (2024). 3D-localization of single point-like gamma sources with a coded aperture camera. Physics in Medicine & Biology, 69(16), 165004, https://doi.org/10.1088/1361-6560/ad6370*


# Required Python packages:
    NumPy (1.24.3)
    Tensorflow (2.10.1)
    SciPy (1.10.1)
    matplotlib (3.7.2)
    json (2.0.9),
    tifffile (2023.4.12)
    PIL (10.0.1)

# How to use:
The main script is "reco.py" and can be either executed either directly from a shell with given arguments (in this case the section overwriting p must be commented), or within a Python IDE by changing the dictionary p.
If reco.py is executed as it is, the examplary detector image "x00y00z50_Minipix_MC_Am241_1mm_MM_1B_001.tif" will be reconstructed in 20 equidistant planes from 5mm to 100mm mask-to-source distance by all provided reconstruction methods. 
Accorsi Decoding corresponds tightly to the implementation from Roberto Accorsi's dissertation and is added here for the sake of completeness. Compared to MURA Decoding it can yield slightly better results but is generally less robust to noisy detector images.
