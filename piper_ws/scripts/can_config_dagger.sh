#!/bin/bash
set -e

# 4CAN DAgger CAN activation. Override these bus-info values with env vars or edit this file
# after checking the adapters with ./find_all_can_port.sh.
: "${CAN_LEFT_FOLLOWER_USB:=3-2:1.0}"
: "${CAN_RIGHT_FOLLOWER_USB:=3-1:1.0}"
: "${CAN_LEFT_LEADER_USB:=1-5:1.0}"
: "${CAN_RIGHT_LEADER_USB:=1-11:1.0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/can_activate.sh" can_left_fol 1000000 "$CAN_LEFT_FOLLOWER_USB"
bash "$SCRIPT_DIR/can_activate.sh" can_right_fol 1000000 "$CAN_RIGHT_FOLLOWER_USB"
bash "$SCRIPT_DIR/can_activate.sh" can_left_lea 1000000 "$CAN_LEFT_LEADER_USB"
bash "$SCRIPT_DIR/can_activate.sh" can_right_lea 1000000 "$CAN_RIGHT_LEADER_USB"
