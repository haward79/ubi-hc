
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


const char * get_config_nanomsg_address()
{
    struct json_object *nanomsg = NULL, *nanomsg_address = NULL;
    const char * nanomsg_address_str = 0;

    json_object_object_get_ex(read_configs(), "nanomsg", &nanomsg);
    json_object_object_get_ex(nanomsg, "address", &nanomsg_address);
    nanomsg_address_str = json_object_get_string(nanomsg_address);

    return nanomsg_address_str;
}


int get_config_nanomsg_ignoreError()
{
    struct json_object *nanomsg = NULL, *nanomsg_ignoreError = NULL;
    int nanomsg_ignoreError_int = 0;

    json_object_object_get_ex(read_configs(), "nanomsg", &nanomsg);
    json_object_object_get_ex(nanomsg, "ignoreError", &nanomsg_ignoreError);
    nanomsg_ignoreError_int = json_object_get_int(nanomsg_ignoreError);

    return nanomsg_ignoreError_int;
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


int get_config_MCD_enable()
{
    struct json_object *MCD = NULL, *MCD_enable = NULL;
    int MCD_enable_int = 0;

    json_object_object_get_ex(read_configs(), "MCD", &MCD);
    json_object_object_get_ex(MCD, "enable", &MCD_enable);
    MCD_enable_int = json_object_get_int(MCD_enable);

    return MCD_enable_int;
}


int get_config_MCD_recording_enable()
{
    struct json_object *MCD = NULL, *MCD_recording = NULL, *MCD_recording_enable = NULL;
    int MCD_recording_enable_int = 0;

    json_object_object_get_ex(read_configs(), "MCD", &MCD);
    json_object_object_get_ex(MCD, "recording", &MCD_recording);
    json_object_object_get_ex(MCD_recording, "enable", &MCD_recording_enable);
    MCD_recording_enable_int = json_object_get_int(MCD_recording_enable);

    return MCD_recording_enable_int;
}


const char *get_config_MCD_recording_path()
{
    struct json_object *MCD = NULL, *MCD_recording = NULL, *MCD_recording_path = NULL;
    const char *MCD_recording_path_str = NULL;

    json_object_object_get_ex(read_configs(), "MCD", &MCD);
    json_object_object_get_ex(MCD, "recording", &MCD_recording);
    json_object_object_get_ex(MCD_recording, "path", &MCD_recording_path);
    MCD_recording_path_str = json_object_get_string(MCD_recording_path);

    return MCD_recording_path_str;
}


int get_config_MCD_objectMark_enable()
{
    struct json_object *MCD = NULL, *MCD_objectMark = NULL, *MCD_objectMark_enable = NULL;
    int MCD_objectMark_enable_int = 0;

    json_object_object_get_ex(read_configs(), "MCD", &MCD);
    json_object_object_get_ex(MCD, "objectMark", &MCD_objectMark);
    json_object_object_get_ex(MCD_objectMark, "enable", &MCD_objectMark_enable);
    MCD_objectMark_enable_int = json_object_get_int(MCD_objectMark_enable);

    return MCD_objectMark_enable_int;
}


int get_config_ECD_enable()
{
    struct json_object *ECD = NULL, *ECD_enable = NULL;
    int ECD_enable_int = 0;

    json_object_object_get_ex(read_configs(), "ECD", &ECD);
    json_object_object_get_ex(ECD, "enable", &ECD_enable);
    ECD_enable_int = json_object_get_int(ECD_enable);

    return ECD_enable_int;
}


int get_config_ECD_recording_enable()
{
    struct json_object *ECD = NULL, *ECD_recording = NULL, *ECD_recording_enable = NULL;
    int ECD_recording_enable_int = 0;

    json_object_object_get_ex(read_configs(), "ECD", &ECD);
    json_object_object_get_ex(ECD, "recording", &ECD_recording);
    json_object_object_get_ex(ECD_recording, "enable", &ECD_recording_enable);
    ECD_recording_enable_int = json_object_get_int(ECD_recording_enable);

    return ECD_recording_enable_int;
}


const char *get_config_ECD_recording_path()
{
    struct json_object *ECD = NULL, *ECD_recording = NULL, *ECD_recording_path = NULL;
    const char *ECD_recording_path_str = NULL;

    json_object_object_get_ex(read_configs(), "ECD", &ECD);
    json_object_object_get_ex(ECD, "recording", &ECD_recording);
    json_object_object_get_ex(ECD_recording, "path", &ECD_recording_path);
    ECD_recording_path_str = json_object_get_string(ECD_recording_path);

    return ECD_recording_path_str;
}


int get_config_ECD_objectMark_enable()
{
    struct json_object *ECD = NULL, *ECD_objectMark = NULL, *ECD_objectMark_enable = NULL;
    int ECD_objectMark_enable_int = 0;

    json_object_object_get_ex(read_configs(), "ECD", &ECD);
    json_object_object_get_ex(ECD, "objectMark", &ECD_objectMark);
    json_object_object_get_ex(ECD_objectMark, "enable", &ECD_objectMark_enable);
    ECD_objectMark_enable_int = json_object_get_int(ECD_objectMark_enable);

    return ECD_objectMark_enable_int;
}
