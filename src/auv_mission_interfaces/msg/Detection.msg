# One detected object, as seen from the vehicle camera.
string class_name            # gate, red_buoy, green_buoy, yellow_buoy, ...
float32 confidence           # 0..1
float32 range_m              # estimated distance camera -> object
float32 bearing_rad          # +left, body frame
float32 elevation_rad        # +up, body frame
geometry_msgs/Point position # estimated world-frame position (ENU, z up, z=0 is the surface)
int32 x1                     # pixel bbox (same convention as AUV CameraMsg)
int32 y1
int32 x2
int32 y2
string source                # ground_truth | camera
