#!/bin/bash

set -e

eval "$(conda shell.bash hook)"
conda activate piperaio
exec roslaunch realsense2_camera multi_camera.launch
