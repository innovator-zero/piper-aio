#!/bin/bash

set -e

eval "$(conda shell.bash hook)"
conda activate piperaio
exec roslaunch piper start_leader_follower.launch mode:=1 auto_enable:=true
