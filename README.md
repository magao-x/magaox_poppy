# magaox_poppy: Fresnel propagation simulation of MagAO-X system
This directory contains various notebooks which examine the Fresnel propagation effects in the Magellan Adaptive Optics eXtreme (MagAO-X) system, which is to be placed in the Magellan telescopes in Chile. The primary objective for this project is to compare the science PSF in various situations (with/without tweeter DM surface, various PSD surface sets). By implementing the various optical elements' surface values and calculating the propagation with Fresnel diffraction, this project aims to provide simulated support for the optical elements' required quality and monetary costs.

Code written by Jennifer Lumbres, with generous assistance and bug-catching help from Ewan Douglas. The PI for MagAO-X is Jared Males.

## Requirements and Installation
### <i>Prerequisites</i>:
- Python 3 (all code is documented using Jupyter Notebook)
- numpy, scipy, matplotlib, astropy, etc 
- POPPY (Download and install from here: https://github.com/mperrin/poppy)

For installing Python3, Jupyter Notebook, and the required libraries, I strongly recommend using Anaconda3: https://www.continuum.io/downloads

## Getting Started
After installing the prerequisites, you can download and run any of the notebooks as you need. The notebooks are independent of each other but all reference to the same data folder, which contains the various PSD and mask FITS files used. 

Currently, the simulation sampling and oversampling has changed to 256 pix sampling and 8x oversampling from the PDR-level 512 pix sampling and 3x oversampling. This is likely to stay as the standard.

There is an ongoing effort to reorganize this repository (it's more difficult than coding). Older data files may be moved to other directories. Any mislabeled surface PSD FITS files in old notebooks (PDR-timeframe) can be found in the data/PSDset1/ folder. 

The notebooks will call for additional FITS files that are not hosted online. These FITS files may be generated through running the notebook and changing some file names around. They are not posted due to github's space allocation limit and each output FITS file is 32 MB.

Please note that different notebooks will have different setups.

The notebooks to focus on are:
- <b>magaox_surfaceCheck</b>: Investigates the surface quality of each optical element as it contributes to the vAPP PSF dark hole contrast level
- <b>magaoxFunctions.py</b>: Contains full MagAO-X design but no aberrated surfaces. Will use the same masks.

All the other notebooks are present as references and are not necessary for understanding the two major notebooks mentioned above.

## Warnings, Disclaimers
A lot of the code will have walls of warnings. They are not detrimental to the operation.

The bare minimum files are posted. If there is a file missing in the repository that is mandatory for calculating the science PSFs, please contact me and I will upload it.

This is still in an organizing phase, so some code may be moved around occasionally. When this occurs, there will be a note posted somewhere here.

This code is only a Fresnel simulation for the MagAO-X system. It is not comparable to a model produced by a high fidelity optical design software (Zemax, Code V, etc).
