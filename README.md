# magaox_poppy: Fresnel propagation simulation of MagAO-X system
This directory contains various notebooks which examine the Fresnel propagation effects in the Magellan Adaptive Optics eXtreme (MagAO-X) system, which is to be placed in the Magellan telescopes in Chile. The primary objective for this project is to compare the science PSF in various situations (with and without aberration; with and without vAPP coronagraph mask). By implementing the various optical elements' surface values and calculating the propagation with diffraction, this project aims to provide simulated support for the optical elements' required quality and monetary costs for the MagAO-X Preliminary Design Review.

## Requirements and Installation
### <i>Prerequisites</i>:
- Python 3 (all code is documented using Jupyter Notebook)
- numpy, scipy, matplotlib, astropy, etc 
- POPPY (Download and install from here: https://github.com/mperrin/poppy)

For installing Python3, Jupyter Notebook, and the required libraries, I strongly recommend using Anaconda3: https://www.continuum.io/downloads

## Getting Started
After installing the prerequisites, you can download and run any of the notebooks as you need. The notebooks are independent of each other but all reference to the same data folder, which contains the various PSD and mask FITS files used, including the vAPP coronagraph zero-padded to work in the notebooks. Please note that different notebooks will have different setups.

The notebooks to focus on are:
- <b>magaox_pdr</b>: Contains the full MagAO-X design with surface PSDs and masks incorporated using FITS files
- <b>magaox_pdr_noAberration</b>: Contains full MagAO-X design but no aberrated surfaces. Will use the same masks.

All the other notebooks are present as references and are not necessary for understanding the two major notebooks mentioned above.

## Warnings, Disclaimers
A lot of the code will have walls of warnings. They are not detrimental to the operation.

The bare minimum files are posted. These are the PSD surfaces of all the optical elements and the vAPP coronagraph. Although the notebooks will call for additional files, these files will be generated internally in the notebook. If there is a file missing in the repository, please contact me and I will upload it.

This is still in an organizing phase, so some code may be moved around occasionally. When this occurs, there will be a note posted somewhere here.

This code is only a Fresnel simulation for the MagAO-X system. It is not an adaptive optics simulator nor a high fidelity optical design software (Zemax, Code V, etc)
