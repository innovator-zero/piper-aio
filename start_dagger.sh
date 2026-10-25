#!/bin/bash

set -e

eval "$(conda shell.bash hook)"
conda activate piperaio
exec roslaunch piper start_leader_follower_dagger.launch
