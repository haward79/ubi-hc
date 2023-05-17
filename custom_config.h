
#ifndef CUSTOM_CONFIG_H

    #define CUSTOM_CONFIG_H

    #include <stdio.h>
    #include <stdlib.h>
    #include <json.h>

    #define JSON_CONFIG_MAX_LENGTH 1024

    struct json_object *read_configs();
    int get_config_vehicle_width();
    int get_config_vehicle_height();
    double get_config_vehicle_camHorizontalAngleDegree();
    int get_config_outputDisplay_enable();
    int get_config_soundAlert_enable();
    int get_config_HMW_enable();
    int get_config_HMW_recording_enable();
    const char *get_config_HMW_recording_path();
    int get_config_HMW_objectMark_enable();

#endif
