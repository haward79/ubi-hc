
#include "custom_config.h"


struct json_object *read_configs()
{
    FILE* fin = fopen("/usr/local/configs/user/nycu.json", "r");
    char buffer[JSON_CONFIG_MAX_LENGTH];
    struct json_object *parsed_json = NULL;

    if(fin == NULL)
    {
        printf("Error: Failed to open config file.\n");
        exit(1);

        return NULL;
    }
    else
    {
        fread(buffer, JSON_CONFIG_MAX_LENGTH, 1, fin);
        fclose(fin);

        parsed_json = json_tokener_parse(buffer);

        return parsed_json;
    }
}


int get_config_vehicle_width()
{
    struct json_object *vehicle = NULL, *vehicle_width = NULL;
    int vehicle_width_int = 0;

    json_object_object_get_ex(read_configs(), "vehicle", &vehicle);
    json_object_object_get_ex(vehicle, "width", &vehicle_width);
    vehicle_width_int = json_object_get_int(vehicle_width);

    return vehicle_width_int;
}


int get_config_vehicle_height()
{
    struct json_object *vehicle = NULL, *vehicle_height = NULL;
    int vehicle_height_int = 0;

    json_object_object_get_ex(read_configs(), "vehicle", &vehicle);
    json_object_object_get_ex(vehicle, "height", &vehicle_height);
    vehicle_height_int = json_object_get_int(vehicle_height);

    return vehicle_height_int;
}


double get_config_vehicle_camHorizontalAngleDegree()
{
    struct json_object *vehicle = NULL, *vehicle_camHorizontalAngleDegree = NULL;
    double vehicle_camHorizontalAngleDegree_double = 0;

    json_object_object_get_ex(read_configs(), "vehicle", &vehicle);
    json_object_object_get_ex(vehicle, "camHorizontalAngleDegree", &vehicle_camHorizontalAngleDegree);
    vehicle_camHorizontalAngleDegree_double = json_object_get_double(vehicle_camHorizontalAngleDegree);

    return vehicle_camHorizontalAngleDegree_double;
}


int get_config_outputDisplay_enable()
{
    struct json_object *outputDisplay = NULL, *outputDisplay_enable = NULL;
    int outputDisplay_enable_int = 0;

    json_object_object_get_ex(read_configs(), "outputDisplay", &outputDisplay);
    json_object_object_get_ex(outputDisplay, "enable", &outputDisplay_enable);
    outputDisplay_enable_int = json_object_get_int(outputDisplay_enable);

    return outputDisplay_enable_int;
}


int get_config_soundAlert_enable()
{
    struct json_object *soundAlert = NULL, *soundAlert_enable = NULL;
    int soundAlert_enable_int = 0;

    json_object_object_get_ex(read_configs(), "soundAlert", &soundAlert);
    json_object_object_get_ex(soundAlert, "enable", &soundAlert_enable);
    soundAlert_enable_int = json_object_get_int(soundAlert_enable);

    return soundAlert_enable_int;
}


int get_config_HMW_enable()
{
    struct json_object *HMW = NULL, *HMW_enable = NULL;
    int HMW_enable_int = 0;

    json_object_object_get_ex(read_configs(), "HMW", &HMW);
    json_object_object_get_ex(HMW, "enable", &HMW_enable);
    HMW_enable_int = json_object_get_int(HMW_enable);

    return HMW_enable_int;
}


int get_config_HMW_recording_enable()
{
    struct json_object *HMW = NULL, *HMW_recording = NULL, *HMW_recording_enable = NULL;
    int HMW_recording_enable_int = 0;

    json_object_object_get_ex(read_configs(), "HMW", &HMW);
    json_object_object_get_ex(HMW, "recording", &HMW_recording);
    json_object_object_get_ex(HMW_recording, "enable", &HMW_recording_enable);
    HMW_recording_enable_int = json_object_get_int(HMW_recording_enable);

    return HMW_recording_enable_int;
}


const char *get_config_HMW_recording_path()
{
    struct json_object *HMW = NULL, *HMW_recording = NULL, *HMW_recording_path = NULL;
    const char *HMW_recording_path_str = 0;

    json_object_object_get_ex(read_configs(), "HMW", &HMW);
    json_object_object_get_ex(HMW, "recording", &HMW_recording);
    json_object_object_get_ex(HMW_recording, "path", &HMW_recording_path);
    HMW_recording_path_str = json_object_get_string(HMW_recording_path);

    return HMW_recording_path_str;
}


int get_config_HMW_objectMark_enable()
{
    struct json_object *HMW = NULL, *HMW_objectMark = NULL, *HMW_objectMark_enable = NULL;
    int HMW_objectMark_enable_int = 0;

    json_object_object_get_ex(read_configs(), "HMW", &HMW);
    json_object_object_get_ex(HMW, "objectMark", &HMW_objectMark);
    json_object_object_get_ex(HMW_objectMark, "enable", &HMW_objectMark_enable);
    HMW_objectMark_enable_int = json_object_get_int(HMW_objectMark_enable);

    return HMW_objectMark_enable_int;
}
