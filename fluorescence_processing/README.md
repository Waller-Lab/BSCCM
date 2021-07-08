# Fluorescence processing, shading correction, demixing

This folder contains everything needed to go from raw fluorescence images to demixed measurements of surface marker expression. The steps in this process are:

1. Compute raw fluorescence measurements from images
     - **compute_fluorescence.ipynb** takes in raw fluorescence images of cells, computes mean fluorescence and mean background, and resaves into dataframe
2. Compute and perform a spatial shading correction to compensate for inhomogenous illumination across the field of view
     - **shading_correction.ipynb** uses the raw fluorescence and background measurements estimate the fixed background and shading correction across the field of view, and correct the raw measurements, taking this into account
3. Visualize this corrected data, and measure the spectrum of each antibody
     - **fluor_gating.ipynb** provides a GUI for visualizing fluoresence data and selecting cells that corrspond to positive examples of stainging with a specific antibody
4. Demix the corrected fluorescence to get measurements of individual surface markers
     - **do_demixing.ipynb** performs nonnegative matrix factorization using the measured fluorophore spectra to estimate the underlying surface marker expression that gave rise to them
     - **demixing_tuning.ipynb** contains code for performing experiments to determine the optimal parameters for doing this demixing
   
     