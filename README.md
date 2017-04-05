# magaox_poppy: Fresnel propagation simulation of MagAO-X system
This directory contains various notebooks which examine the Fresnel propagation effects in the MagAO-X system, which is to be placed in the Magellan telescopes in Chile. The motivation for this project is to compare the science PSF in various situations (with and without aberration; ideal and vAPP coronagraph mask) as support for the Preliminary Design Review.

## Requirements and Installation
### <i>Prerequisites</i>:
- Python 3 (all code is documented using Jupyter Notebook)
- numpy, scipy, matplotlib, astropy, etc (I strongly recommend Anaconda: https://www.continuum.io/downloads)
- POPPY (Download and install from here: https://github.com/mperrin/poppy)

## Getting Started
You can download and run any of the notebooks as you need. The notebooks are independent of each other but all reference to the same data folder, which contains the various PSD and mask FITS files used. Please note that different notebooks will have different setups.

The notebooks to focus on are:
- <b>magaox_pdr</b>: Contains the full MagAO-X design with surface PSDs and masks incorporated using FITS files
- <b>magaox_pdr_noAberration</b>: Contains full MagAO-X design but no aberrated surfaces. Will use the same masks.

All the other notebooks are present as references and are not necessary for understanding the two major notebooks mentioned above.

## Warnings, Disclaimers
A lot of the code will have walls of warnings. They are not detrimental to the operation.

This is still in an organizing phase, so some code may be moved around occasionally. When this occurs, there will be a note posted somewhere here.

This code is only a Fresnel simulation for the MagAO-X system. It is not an adaptive optics simulator nor a high fidelity optical design software (Zemax, Code V, etc)
