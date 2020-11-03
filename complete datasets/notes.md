This folder contains 2 annotated datasets of marine vessels captured University of Victoria's Coastal and Ocean Resource Analysis Laboratory [CORAL](www.coral.geog.uvic.ca) using a static camera focused off-shore to the south and west of southern Vancouver Island, BC, Canada, during the years of 2019 and 2020. 

A pan-tilt-zoom (PTZ) camera was installed at two fixed positions on a headland overlooking a major vessel traffic thoroughfare and configured to continuously capture three 1920 x1080 pixels photos in the first 15 seconds of each minute. 

We manually annotate and make publicly available two datasets used to evaluate the proposed hybrid detector under two conditions: 
* **D1**: 633 images containing boats of various sizes (mean vessel area of 953 pixels)
* **D2**: 138 images presenting only small boats with a mean area of 79 pixels. D2 highlights the capabilities of the DSMV, as most of its marine vessels are missed by the state-of-the-art object detectors. 

While creating both D1 and D2 we selected images under different weather conditions and vessel layouts, so that all monitoring scenarios are well represented. 

**Note**: the annotations were created using MATLAB's *imagelabeler*. Make sure that you change their source path (i.e., gTruth.DataSource.Source) when using these datasets. 

<p align="center">
  <img src="https://i.imgur.com/fwiNzRm.jpg" width="600">
</p>

