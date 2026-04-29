#!/bin/bash

# Usage:

# 1. Prerequisites
#     Install the ip and ethtool tools on the system.
#     sudo apt install ethtool can-utils
#     Make sure the gs_usb driver is installed correctly.

# 2. Background
#  This script automatically manages, renames, and activates CAN (Controller Area Network) interfaces.
#  It checks the number of CAN modules currently present in the system, then renames and activates
#  CAN interfaces according to predefined USB ports.
#  This is useful for systems with multiple CAN modules, especially when different modules require specific names.

# 3. Main features
#  Check the CAN module count: ensure the detected CAN module count matches the configured count.
#  Get USB port information: use ethtool to get USB port information for each CAN module.
#  Validate USB ports: check whether each CAN module is connected to a predefined USB port.
#  Rename CAN interfaces: rename each CAN interface to the target name according to its predefined USB port.

# 4. Script configuration
#   Key configuration items include the expected CAN module count, default CAN interface name, and bitrate settings:
#   1. Expected CAN module count:
#     EXPECTED_CAN_COUNT=1
#     This value determines how many CAN modules should be detected in the system.
#   2. Default CAN interface name when using a single CAN module:
#     DEFAULT_CAN_NAME="${1:-can0}"
#     The default CAN interface name can be specified through a command-line argument. If omitted, it defaults to can0.
#   3. Default bitrate when using a single CAN module:
#     DEFAULT_BITRATE="${2:-500000}"
#     The bitrate for a single CAN module can be specified through a command-line argument. If omitted, it defaults to 500000.
#   4. Configuration for multiple CAN modules:
#     declare -A USB_PORTS
#     USB_PORTS["1-2:1.0"]="can_device_1:500000"
#     USB_PORTS["1-3:1.0"]="can_device_2:250000"
#     Each key is a USB port. Each value is an interface name and bitrate separated by a colon.

# 5. Usage steps
#  1. Edit the script:
#   1. Modify the predefined values:
#      - Predefined CAN module count: EXPECTED_CAN_COUNT=2. Set this to the number of CAN modules connected to the industrial PC.
#      - If there is only one CAN module, set the parameter above and skip the multi-module USB port configuration.
#      - Predefined USB ports and target interface names for multiple CAN modules:
#          First, plug one CAN module into the expected USB port. During initial setup, plug in only one CAN module at a time.
#          Then run sudo ethtool -i can0 | grep bus, and record the value after bus-info:.
#          Next, plug in another CAN module. Use a different USB port from the previous module, then repeat the previous step.
#          You can also use one CAN module to test different USB ports, because modules are identified by USB address.
#          After all modules have been assigned to their expected USB ports and all records are complete,
#          update the USB ports (bus-info) and target interface names according to the actual setup.
#          In can_device_1:500000, the first part is the configured CAN name and the second part is the configured bitrate.
#            declare -A USB_PORTS
#            USB_PORTS["1-2:1.0"]="can_device_1:500000"
#            USB_PORTS["1-3:1.0"]="can_device_2:250000"
#          Modify the quoted value in USB_PORTS["1-3:1.0"] to match the value recorded after bus-info:.
#   2. Grant execute permission:
#       Open a terminal, navigate to the script directory, and run the following command:
#       chmod +x can_config.sh
#   3. Run the script:
#     Run the script with sudo because it needs administrator privileges to modify network interfaces:
#       1. Single CAN module
#         1. Specify the default CAN interface name and bitrate through command-line arguments. Defaults are can0 and 500000:
#           sudo bash ./can_config.sh [CAN interface name] [bitrate]
#           For example, set the interface name to my_can_interface and the bitrate to 1000000:
#           sudo bash ./can_config.sh my_can_interface 1000000
#         2. Specify the CAN name by providing the USB hardware address:
#           sudo bash ./can_config.sh [CAN interface name] [bitrate] [USB hardware address]
#           For example, set the interface name to my_can_interface, the bitrate to 1000000, and the USB hardware address to 1-3:1.0:
#           sudo bash ./can_config.sh my_can_interface 1000000 1-3:1.0
#           This assigns the CAN device at USB address 1-3:1.0 to my_can_interface with a bitrate of 1000000.
#       2. Multiple CAN modules
#         For multiple CAN modules, set the USB_PORTS array in the script to specify each module's interface name and bitrate.
#         No extra parameters are required. Run the script directly:
#         sudo ./can_config.sh

# Notes

#     Permission requirements:
#         The script must run with sudo because renaming and configuring network interfaces requires administrator privileges.
#         Make sure you have sufficient permissions to run this script.

#     Script environment:
#         This script assumes a bash environment. Make sure your system uses bash instead of another shell such as sh.
#         Check the shebang line (#!/bin/bash) to confirm that bash is used.

#     USB port information:
#         Make sure the predefined USB port information (bus-info) matches the ethtool output on the actual system.
#         Use commands such as sudo ethtool -i can0 and sudo ethtool -i can1 to check each CAN interface's bus-info.

#     Interface conflicts:
#         Make sure target interface names such as can_device_1 and can_device_2 are unique and do not conflict with existing interface names.
#         If you need to change the USB port-to-interface mapping, update the USB_PORTS array according to the actual setup.
#-------------------------------------------------------------------------------------------------#

# Predefined CAN module count
EXPECTED_CAN_COUNT=2

if [ "$EXPECTED_CAN_COUNT" -eq 1 ]; then
    # Default CAN name, configurable through a command-line argument
    DEFAULT_CAN_NAME="${1:-can0}"

    # Default bitrate for a single CAN module, configurable through a command-line argument
    DEFAULT_BITRATE="${2:-1000000}"

    # USB hardware address (optional argument)
    USB_ADDRESS="${3}"
fi

# Predefined USB ports, target interface names, and bitrates (used for multiple CAN modules)
if [ "$EXPECTED_CAN_COUNT" -ne 1 ]; then
    declare -A USB_PORTS 
    USB_PORTS["3-2:1.0"]="can_left:1000000"
    USB_PORTS["3-1:1.0"]="can_right:1000000"
fi

# Get the current CAN module count in the system
CURRENT_CAN_COUNT=$(ip link show type can | grep -c "link/can")

# Check whether the current CAN module count matches the expected count
if [ "$CURRENT_CAN_COUNT" -ne "$EXPECTED_CAN_COUNT" ]; then
    echo "Error: detected CAN module count ($CURRENT_CAN_COUNT) does not match the expected count ($EXPECTED_CAN_COUNT)."
    exit 1
fi

# Load the gs_usb module
sudo modprobe gs_usb
if [ $? -ne 0 ]; then
    echo "Error: failed to load the gs_usb module."
    exit 1
fi

# Check whether only one CAN module needs to be handled
if [ "$EXPECTED_CAN_COUNT" -eq 1 ]; then
    if [ -n "$USB_ADDRESS" ]; then
        echo "Detected USB hardware address argument: $USB_ADDRESS"
        
        # Use ethtool to find the CAN interface corresponding to the USB hardware address
        INTERFACE_NAME=""
        for iface in $(ip -br link show type can | awk '{print $1}'); do
            BUS_INFO=$(sudo ethtool -i "$iface" | grep "bus-info" | awk '{print $2}')
            if [ "$BUS_INFO" = "$USB_ADDRESS" ]; then
                INTERFACE_NAME="$iface"
                break
            fi
        done
        
        if [ -z "$INTERFACE_NAME" ]; then
            echo "Error: failed to find the CAN interface corresponding to USB hardware address $USB_ADDRESS."
            exit 1
        else
            echo "Found interface $INTERFACE_NAME corresponding to USB hardware address $USB_ADDRESS."
        fi
    else
        # Get the only CAN interface
        INTERFACE_NAME=$(ip -br link show type can | awk '{print $1}')
        
        # Check whether the interface name was found
        if [ -z "$INTERFACE_NAME" ]; then
            echo "Error: failed to detect a CAN interface."
            exit 1
        fi

        echo "Expected one CAN module and detected interface $INTERFACE_NAME."
    fi

    # Check whether the current interface is already up
    IS_LINK_UP=$(ip link show "$INTERFACE_NAME" | grep -q "UP" && echo "yes" || echo "no")

    # Get the current interface bitrate
    CURRENT_BITRATE=$(ip -details link show "$INTERFACE_NAME" | grep -oP 'bitrate \K\d+')

    if [ "$IS_LINK_UP" = "yes" ] && [ "$CURRENT_BITRATE" -eq "$DEFAULT_BITRATE" ]; then
        echo "Interface $INTERFACE_NAME is already up with bitrate $DEFAULT_BITRATE."
        
        # Check whether the interface name matches the default name
        if [ "$INTERFACE_NAME" != "$DEFAULT_CAN_NAME" ]; then
            echo "Renaming interface $INTERFACE_NAME to $DEFAULT_CAN_NAME."
            sudo ip link set "$INTERFACE_NAME" down
            sudo ip link set "$INTERFACE_NAME" name "$DEFAULT_CAN_NAME"
            sudo ip link set "$DEFAULT_CAN_NAME" up
            echo "Interface has been renamed to $DEFAULT_CAN_NAME and brought back up."
        else
            echo "Interface name is already $DEFAULT_CAN_NAME."
        fi
    else
        # Configure the interface if it is down or uses a different bitrate
        if [ "$IS_LINK_UP" = "yes" ]; then
            echo "Interface $INTERFACE_NAME is already up, but its bitrate is $CURRENT_BITRATE instead of the configured value $DEFAULT_BITRATE."
        else
            echo "Interface $INTERFACE_NAME is down or has no bitrate configured."
        fi
        
        # Set the interface bitrate and bring it up
        sudo ip link set "$INTERFACE_NAME" down
        sudo ip link set "$INTERFACE_NAME" type can bitrate $DEFAULT_BITRATE
        sudo ip link set "$INTERFACE_NAME" up
        echo "Interface $INTERFACE_NAME has been configured with bitrate $DEFAULT_BITRATE and brought up."
        
        # Rename the interface to the default name
        if [ "$INTERFACE_NAME" != "$DEFAULT_CAN_NAME" ]; then
            echo "Renaming interface $INTERFACE_NAME to $DEFAULT_CAN_NAME."
            sudo ip link set "$INTERFACE_NAME" down
            sudo ip link set "$INTERFACE_NAME" name "$DEFAULT_CAN_NAME"
            sudo ip link set "$DEFAULT_CAN_NAME" up
            echo "Interface has been renamed to $DEFAULT_CAN_NAME and brought back up."
        fi
    fi
else
    # Handle multiple CAN modules

    # Check whether the number of USB ports and target interface names matches the expected CAN module count
    PREDEFINED_COUNT=${#USB_PORTS[@]}
    if [ "$EXPECTED_CAN_COUNT" -ne "$PREDEFINED_COUNT" ]; then
        echo "Error: configured CAN module count ($EXPECTED_CAN_COUNT) does not match the predefined USB port count ($PREDEFINED_COUNT)."
        exit 1
    fi

    # Iterate through all CAN interfaces
    for iface in $(ip -br link show type can | awk '{print $1}'); do
        # Use ethtool to get bus-info
        BUS_INFO=$(sudo ethtool -i "$iface" | grep "bus-info" | awk '{print $2}')
        
        if [ -z "$BUS_INFO" ];then
            echo "Error: failed to get bus-info for interface $iface."
            continue
        fi
        
        echo "Interface $iface is connected to USB port $BUS_INFO."

        # Check whether bus-info is in the predefined USB port list
        if [ -n "${USB_PORTS[$BUS_INFO]}" ];then
            IFS=':' read -r TARGET_NAME TARGET_BITRATE <<< "${USB_PORTS[$BUS_INFO]}"
            
            # Check whether the current interface is already up
            IS_LINK_UP=$(ip link show "$iface" | grep -q "UP" && echo "yes" || echo "no")

            # Get the current interface bitrate
            CURRENT_BITRATE=$(ip -details link show "$iface" | grep -oP 'bitrate \K\d+')

            if [ "$IS_LINK_UP" = "yes" ] && [ "$CURRENT_BITRATE" -eq "$TARGET_BITRATE" ]; then
                echo "Interface $iface is already up with bitrate $TARGET_BITRATE."
                
                # Check whether the interface name matches the target name
                if [ "$iface" != "$TARGET_NAME" ]; then
                    echo "Renaming interface $iface to $TARGET_NAME."
                    sudo ip link set "$iface" down
                    sudo ip link set "$iface" name "$TARGET_NAME"
                    sudo ip link set "$TARGET_NAME" up
                    echo "Interface has been renamed to $TARGET_NAME and brought back up."
                else
                    echo "Interface name is already $TARGET_NAME."
                fi
            else
                # Configure the interface if it is down or uses a different bitrate
                if [ "$IS_LINK_UP" = "yes" ]; then
                    echo "Interface $iface is already up, but its bitrate is $CURRENT_BITRATE instead of the configured value $TARGET_BITRATE."
                else
                    echo "Interface $iface is down or has no bitrate configured."
                fi
                
                # Set the interface bitrate and bring it up
                sudo ip link set "$iface" down
                sudo ip link set "$iface" type can bitrate $TARGET_BITRATE
                sudo ip link set "$iface" up
                echo "Interface $iface has been configured with bitrate $TARGET_BITRATE and brought up."
                
                # Rename the interface to the target name
                if [ "$iface" != "$TARGET_NAME" ]; then
                    echo "Renaming interface $iface to $TARGET_NAME."
                    sudo ip link set "$iface" down
                    sudo ip link set "$iface" name "$TARGET_NAME"
                    sudo ip link set "$TARGET_NAME" up
                    echo "Interface has been renamed to $TARGET_NAME and brought back up."
                fi
            fi
        else
            echo "Error: unknown USB port $BUS_INFO for interface $iface."
            exit 1
        fi
    done
fi

echo "All CAN interfaces have been renamed and brought up successfully."
