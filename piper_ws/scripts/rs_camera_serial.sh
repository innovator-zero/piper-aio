ws_path=$(pwd)
echo $ws_path

cd $ws_path/camera_ws

file="$ws_path/camera_ws/devel/lib/realsense2_camera/list_device_camera"
if [[ -f "$file" ]]; then
    echo "list_devices_camera exist."
else
    echo "list_devices_camera no exist!"
    exit 0
fi

echo ""
echo "camera serial number: "
$ws_path/camera_ws/devel/lib/realsense2_camera/list_device_camera

echo "-----------------"
echo ""