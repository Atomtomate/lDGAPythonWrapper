#!/bin/bash

module add openblas intel module add fftw/3.3.9
module save ed
conda create --name p3 --file conda_env.list -y
conda activate p3
