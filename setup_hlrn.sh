#!/bin/bash

module add openblas/gcc.9/0.3.7 impi/2019.5 intel/19.0.5 fftw3/impi/intel/3.3.8
module save ed
conda create --name p3 --file conda_env.list -y
conda activate p3
