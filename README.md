# Reconstruction Methods for Coded Aperture Imaging
<p align="center">
  <img width="228" height="163" src="https://github.com/user-attachments/assets/246cfac4-4e01-44c5-b362-985a750dd923">
</p>

Coded Aperture Imaging (CAI) promises a better trade-off between sensitivity and spatial resolution in imaging of gamma sources. However, it requires image reconstruction to obtain an interpretable image.
This repository contains two commonly used reconstruction for CAI: MURA Decoding and convolutional 3D Maximum Likelihood Expectation Maximization (3D-MLEM) algorithm, an extended version based on the original algorithm from [Mu et al](https://ieeexplore.ieee.org/document/1637528). Both can be used for planar reconstruction and 3D reconstruction.
Additionally to MURA Decoding, a method called "Accorsi Decoding" is implemented as well. It corresponds tightly to the implementation from [Roberto Accorsi's dissertation](http://hdl.handle.net/1721.1/8684) and is added here for the sake of completeness. Compared to MURA Decoding, it somtimes yields slightly better results, but is generally less robust to noisy detector images.

If you decide to use these implementation, please cite the following publication:

*Meißner, T., Cerbone, L. A., Russo, P., Nahm, W., & Hesser, J. (2024). 3D-localization of single point-like gamma sources with a coded aperture camera. Physics in Medicine & Biology, 69(16), 165004, https://doi.org/10.1088/1361-6560/ad6370*


## Get Started
### Dependencies and Installation
1. Clone repo ```git clone https://github.com/tomeiss/cai-reconstruction```
2. Install Python 3.8.18
3. Install Dependencies (Conda/Miniconda is recommended)
    - NumPy (1.24.3)
    - Tensorflow (2.10.1)
    - SciPy (1.10.1)
    - matplotlib (3.7.2)
    - json (2.0.9)
    - tifffile (2023.4.12)
    - PIL (10.0.1) 
4. Run exemplary reconstruction with:
```
python reco.py
```
The exemplary detector image "x00y00z50_Minipix_MC_Am241_1mm_MM_1B_001.tif" will be reconstructed in 20 equidistant planes from 5mm to 100mm mask-to-source distance by all provided reconstruction methods. The camera setup, the geometrical properties, the source etc. is explained in detail in the publication mentioned above.
<p align="center">
  <img src="https://github.com/user-attachments/assets/1f6a1cdf-3a70-4a58-a1a6-c562680ed851">
</p>

Reconstruction results from 3D-MLEM (source hardly visible inside the red circle), Accorsi Decoding, and MURA Decoding:
<p align="center">
  <img src="https://github.com/user-attachments/assets/292ed313-c21e-4f38-ba97-53c37f8c2b8d">
</p>



## Details
The main script is "reco.py" and can be either executed either directly from a shell with given arguments (in this case the section overwriting ```p``` must be commented), or within a Python IDE by changing the dictionary ```p```. The parameter describing the camera setup (detector size, pixelation, mask size, mask-to-detector distance, mask pattern, pinhole size, ...) can be set via the arguments and the mask pattern must be given via tiff file that describes the binary mask pattern. Currently, deriving the MURA Decoding pattern from the given mask pattern works only for the pattern used on the publication mentioned above. More about deriving the decoding pattern can be found in [Cieslak et al.](https://doi.org/10.1016/j.radmeas.2016.08.002).

## Datasets
As part of other publications, we have made two datasets available to the research community to ease the access to CAI. One dataset can be found [here](https://github.com/tomeiss/assessment_of_axial_resolution_in_CAI) and the other one [here](https://github.com/tomeiss/3d_localization_with_cai).
Synthetic coded aperture images can be simulated via the convolutional model, which we implemented with [*ConvSim*](https://github.com/tomeiss/convsim) where a describtion can be found under [Meißner et al.](https://doi.org/10.1117/12.2670883).
Other than that, it is common to simulate coded aperture cameras in Monte Carlo simulation frameworks like TOPAS MC or Geant4. 
