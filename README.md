[IMPLEMENTATION OF MODEL-BLIND VIDEO DENOISING VIA FRAME-TO-FRAME TRAINING](http://openaccess.thecvf.com/content_CVPR_2019/papers/Ehret_Model-Blind_Video_Denoising_via_Frame-To-Frame_Training_CVPR_2019_paper.pdf)
=========================================================================

* Author    : EHRET Thibaud <ehret.thibaud@gmail.com>
* Licence   : AGPL v3+, see gpl.txt

OVERVIEW
--------

This code is provided to reproduce the results from
 "Model-blind Video Denoising Via Frame-to-frame Training
, T. Ehret, A. Davy, J.M. Morel, G. Facciolo, P. Arias, CVPR 2019".
Please cite it if you use this code as part of your research.

USAGE
-----

List all available options:</br>
```python blind_denoising.py --help```

There are 4 mandatory input arguments:
* `--input` the path to input frames (C type)
* `--flow` the path to optical flow (C type), must be readable by readFlowFile.py
* `--output` the path to output frames (C type)
* `--ref` the path to reference frames (C type), it can be set to the input frames 
            if the true reference is not available (only used to compute the PSNR)

There are 5 optional input arguments:
* `--first` the index of the first frame to be processed, default is 1
* `--last` the index of the last frame to be processed, default is 300
* `--output_psnr` the path to file where the PSNRs are going to be written, default is `plot_psnr.txt`
* `--output_network` the path to file where the fine-tuned network will be saved, default is `final.pth`
* `--iter` the number of backpropagation to done, default is 20
* `--network` the path to the network (only change if you know what you are doing), default is set to DnCNN 25


The input sequence provided should already be a degraded grayscale sequence (the code can read grayscale png, jpeg and tiff files). The optical flow should be computed before running the denoising code (and preferably on the degraded sequence). 

OPTICAL FLOW
------------

The code used to compute the optical flow for the CVPR paper is provided in the tvl1flow folder. It's a modified version of "Javier Sánchez Pérez, Enric Meinhardt-Llopis, and Gabriele Facciolo, TV-L1 Optical Flow Estimation, Image Processing On Line, 3 (2013), pp. 137–150."

The code is compilable on Unix/Linux and hopefully on Mac OS (not tested!). 

**Compilation:** requires the cmake and make programs.

Compile the source code using make.

UNIX/LINUX/MAC:
```
$ mkdir build; cd build
$ cmake ..
$ make
```

Binaries will be created in the `build/ folder`.

NOTE: By default, the code is compiled with OpenMP multithreaded
parallelization enabled (if your system supports it). 


A script `tvl1flow.sh` is also provided.
The command to run this script is (assuming it is in the same folder as the tvl1flow binary):
```
$ ./tvl1flow.sh inputPath first last outputPath
```
where the mandatory inputs are:
* `inputPath` the path to the input frames (using the C standard), for example 'frame%03d.png'
* `first` the index of the first frame, for example '1'
* `last` the index of the last frame, for example '100'
* `outputPath` the path to the output flow, for example 'tvl1_%03d.flo'
