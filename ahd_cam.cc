
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <signal.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/timerfd.h>
#include <assert.h>
#include <pthread.h>

#include <time.h>
#include <fcntl.h>
#include <unistd.h>

#include <fstream>

#include <nanomsg/nn.h>
#include <nanomsg/pipeline.h>
#include <nanomsg/pubsub.h>
#include <nanomsg/reqrep.h>

#include <userconfig.h>
#include "jmpp/jmpp.h"
#include "jmpp/jmpp_npu_api.h"
#include "jmpp/jmpp_vo_rgb_api.h"
#include "jmpp/jmpp_rga_api.h"
#include "jmpp/jmpp_raw_stream_api.h"
#include <jmpp/jmpp_api.h>

// hsunchi code.
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing/render_face_detections.h"
#include "dlib/image_processing.h"
#include "dlib/image_io.h"
#include "dlib/opencv.h"
#include "dlib/array.h"

// hsunchi code.
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#include "rknn_api.h"
#include "ssd.h"

#include "custom_config.h"
#include "digit.h"
#include "nycu_mhw_api.h"

// hsunchi code.
using namespace std;
using namespace std::chrono;
using namespace dlib;

// Set this to image size.
#define MODEL_INPUT_SIZE 300

// For min distance calculation.
#define VEHICLE_WIDTH_METER 1.75
const double TAN_OF_HHVIEW = tan(get_config_vehicle_camHorizontalAngleDegree() / 180.0 * M_PI * 0.5);

const int CAMERA_ID = 0;  // camera id 0
const int RGA_INPUT_WIDTH = 544;
const int RGA_INPUT_HEIGHT = 304;
const int RGA_OUTPUT_WIDTH = 532;
const int RGA_OUTPUT_HEIGHT = MODEL_INPUT_SIZE;
const int RGA_OUTPUT_CROP_OFFSET = 116;

const char* SSD_MODEL_PATH = "model/ssd_inception_v2_rv1109_rv1126.rknn";
//const char* SSD_MODEL_PATH = "model/MobilenetV2_SSD_Lite_bc.rknn";

// hsunchi model.
const char *HSUNCHI_SSD_MODEL_PATH = "model/vgg16_eyeclose.rknn";

const int RGA_BUFFER_INPUT_SIZE = RGA_INPUT_WIDTH * RGA_INPUT_HEIGHT * 3;
const int RGA_BUFFER_OUTPUT_SIZE = RGA_OUTPUT_WIDTH * RGA_OUTPUT_HEIGHT * 3;
const int RGA_BUFFER_SCALED_SIZE = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 3;
const int MAX_RKNN_LIST_NUM = 5;
const int MAX_IMAGE_LIST_NUM = 5;

const int MODEL_DETECT_BOUNDARY_TOP = 150;
const int MODEL_DETECT_BOUNDARY_BOTTOM = 299;
const int MODEL_DETECT_BOUNDARY_LEFT = 100;
const int MODEL_DETECT_BOUNDARY_RIGHT = 199;

/*! Depending on the video decoder pipeline, video input color space can either be NV12 or NV16 */

int mpower_npu_enable;          // mpower process enable or disable
int mpower_2D_graphic_enable;   // mpower helps do some 2D drawing on the frame
int npu_resolution_width;
int npu_resolution_height;

IMAGE_TYPE_E  npu_color_space;

JMPP_RAW_STREAM_S raw_stream;    // do not modify this data or read structure data directly.
                                 // always use API to access information.
                                 // this may take a large memory. declare as a global variable

int raw_stream_vi_chn;
int raw_stream_vi_pipe;

uint32_t raw_stream_data_size = 0;
uint8_t *raw_stream_data_addr = 0;
uint64_t raw_stream_data_timestamp = 0;
uint32_t      raw_stream_data_width = 0;
uint32_t      raw_stream_data_height = 0;
IMAGE_TYPE_E  raw_stream_data_color_space = IMAGE_TYPE_UNKNOW;

// hsunchi
uint32_t        image_size_h = 0;

// Howard
uint32_t        image_size_hi = 0;

uint8_t*        image_addr         = (uint8_t*)malloc(RGA_BUFFER_OUTPUT_SIZE);  // These memory will be released at the end of main function.
uint8_t*        image_drawing_addr = (uint8_t*)malloc(RGA_BUFFER_OUTPUT_SIZE);
uint8_t*        image_drawing_cropped_addr = (uint8_t*)malloc(RGA_BUFFER_SCALED_SIZE);

uint64_t        image_timestamp = 0;
uint32_t        image_width = 0;
uint32_t        image_height = 0;
IMAGE_TYPE_E    image_color_space = IMAGE_TYPE_UNKNOW;

int image_id = -1;
int image_id_prev = -1;

rknn_context rknn_ctx, hsunchi_rknn_ctx;
rknn_input_output_num rknn_io_num, hsunchi_rknn_io_num;


// rknn list structure.
typedef struct node
{
    long timeval;
    detect_result_group_t detect_result_group;
    struct node *next;
} Node;

typedef struct my_stack
{
    int size;
    Node *top;
} rknn_list;

rknn_list *rknn_list_all_;


// Color struct.
typedef struct
{
    int red = 0;
    int green = 0;
    int blue = 0;
} Color;

Color COLOR_RED;
Color COLOR_GREEN;
Color COLOR_BLUE;


// Function declaration.
static int npu_config_json_get();        // getting mpower configuration
void rga_chn12_frame_cb(MEDIA_BUFFER mb);
static void wait_jmpp_init_complete();
static unsigned char *load_model(const char *, int *);
void create_rknn_list(rknn_list **);
void destory_rknn_list(rknn_list **);
void rknn_list_push(rknn_list *, long, detect_result_group_t);
void rknn_list_pop(rknn_list *, long *, detect_result_group_t *);
void rknn_list_drop(rknn_list *);
int rknn_list_size(rknn_list *);
static void print_rknn_tensor(rknn_tensor_attr *);
int predict_one_pic(cv::Mat, rknn_context, rknn_input_output_num);
static cv::Rect dlibRectangleToOpenCV(dlib::rectangle);
static void *detect_thread_handler(void *);
void draw_border(char*, int, int, int, int, int, int, Color);
void put_text(char*, int, int, int, int, char*);
void crop_hFlip_image(uint8_t*, uint8_t*);
static long getCurrentTimeMsec();


// Main

int main()
{
    // Define color codes.
    COLOR_RED.red = 255;
    COLOR_GREEN.green = 255;
    COLOR_BLUE.blue = 255;

    #define NUM_NPU_FD 2
    int r = 0, hr = 0;
    int model_len = 0, hsunchi_model_len = 0;
    unsigned char *model = nullptr, *hsunchi_model = nullptr;


    create_rknn_list(&rknn_list_all_);
    wait_jmpp_init_complete();
    jmpp_api_init();  // before calling JMPP API. Need to initialize it
    npu_config_json_get();  // read mpower NPU configuration


    // open raw stream(224x224)
    // raw stream is in NV16 color space.
    // User has to call Rockchip native API to get frame data.
    r = jmpp_raw_stream_open(CAMERA_ID, &raw_stream);

    if(r != 0)
    {
        printf("Can NOT open RAW stream.\n");
    }


    // user can use vi_pipe and vi_chn to call rockchip API to get data.
    // It is possible to get VI data if vi_chn is known.
    // It is possible to bind VI data to one or mutiple RGA if vi_chn is known.
    // 
    // There are different ways to get frame data, either using RK_MPI_SYS_GetMediaBuffer()
    // or RK_MPI_SYS_RegisterOutCb(). But application may have to create thread when calling them.
    raw_stream_vi_pipe = jmpp_raw_stream_vi_pipe(CAMERA_ID, &raw_stream);
    raw_stream_vi_chn = jmpp_raw_stream_vi_chn(CAMERA_ID, &raw_stream);


    printf("RGA image input size is set to (%d, %d).\n", RGA_INPUT_WIDTH, RGA_INPUT_HEIGHT);
    printf("RGA image output size is set to (%d, %d).\n", RGA_OUTPUT_WIDTH, RGA_OUTPUT_HEIGHT);


    {
        // RGA[12] color space conversation
        // coverting color space
        
        RGA_ATTR_S stRgaAttr;
        stRgaAttr.bEnBufPool = RK_TRUE;
        stRgaAttr.u16BufPoolCnt = 2;  // Please do not set too many. 2 is default
        stRgaAttr.u16Rotaion = 0;
        stRgaAttr.stImgIn.u32X = 0;
        stRgaAttr.stImgIn.u32Y = 0;
        stRgaAttr.stImgIn.imgType = IMAGE_TYPE_NV16;  // MUST BE NV16.
        stRgaAttr.stImgIn.u32Width = RGA_INPUT_WIDTH;
        stRgaAttr.stImgIn.u32Height = RGA_INPUT_HEIGHT;
        stRgaAttr.stImgIn.u32HorStride = RGA_INPUT_WIDTH;
        stRgaAttr.stImgIn.u32VirStride = RGA_INPUT_HEIGHT;
        
        stRgaAttr.stImgOut.u32X = 0;
        stRgaAttr.stImgOut.u32Y = 0;
        stRgaAttr.stImgOut.imgType = IMAGE_TYPE_RGB888;
        stRgaAttr.stImgOut.u32Width = RGA_OUTPUT_WIDTH;
        stRgaAttr.stImgOut.u32Height = RGA_OUTPUT_HEIGHT;
        stRgaAttr.stImgOut.u32HorStride = RGA_OUTPUT_WIDTH;
        stRgaAttr.stImgOut.u32VirStride = RGA_OUTPUT_HEIGHT;

        // RGA channel 12 .. 15 can be used with rockchip native API
        r = RK_MPI_RGA_CreateChn(12, &stRgaAttr);
        
        if(r)
        {
            printf("ERROR: Create rga[12] falied! ret=%d\n", r);
        }

        // register a callback when there is a data on rga channel 12
        MPP_CHN_S stEncChn;

        stEncChn.enModId = RK_ID_RGA;
        stEncChn.s32ChnId = 12;
        
        r = RK_MPI_SYS_RegisterOutCb(&stEncChn, rga_chn12_frame_cb);

        if(r)
        {
            printf("ERROR: Register cb for RGA CB error! code:%d\n", r);
        }
    }

    // starting the VI -> RGA[12], VI -> RGA[13] pipeline data.
    // VI automatically starts capturing data after binding is completed.
    {
        MPP_CHN_S stSrcChn;
        stSrcChn.enModId = RK_ID_VI;
        stSrcChn.s32DevId = raw_stream_vi_pipe;
        stSrcChn.s32ChnId = raw_stream_vi_chn;
        MPP_CHN_S stDestChn;
        stDestChn.enModId = RK_ID_RGA;
        stDestChn.s32DevId = 0;
        stDestChn.s32ChnId = 12;
        
        r = RK_MPI_SYS_Bind(&stSrcChn, &stDestChn);
        
        if(r)
        {
            printf("ERROR: Bind vi[%d:%d] and rga[12] failed! ret=%d\n", raw_stream_vi_pipe, raw_stream_vi_chn, r);
        }
    }

    // ::sleep(1);
    // FILE* fout = fopen("image.raw", "wb");
    // fwrite(image_addr, 1, image_size_hi, fout);
    // fclose(fout);
    // printf("[!!!] Saved a raw image to disk.\n");

    // Preload rknn model.
    printf("Loading model ...\n");
    model = load_model(SSD_MODEL_PATH, &model_len);
    r = rknn_init(&rknn_ctx, model, model_len, 0);
    
    if(r < 0)
    {
        printf("rknn initialization failed! Returned %d.\n", r);
    }
    else
    {
        // Get Model Input Output Info
        r = rknn_query(rknn_ctx, RKNN_QUERY_IN_OUT_NUM, &rknn_io_num, sizeof(rknn_io_num));
        
        if(r != RKNN_SUCC)
        {
            printf("rknn query failed! Return %d.\n", r);
        }
        else
        {
            printf("Model loaded successfully.\n");
            printf("Model input num: %d, output num: %d\n", rknn_io_num.n_input, rknn_io_num.n_output);

            {
                printf("Model input tensors:\n");
                
                rknn_tensor_attr input_attrs[rknn_io_num.n_input];
                memset(input_attrs, 0, sizeof(input_attrs));
                
                for(unsigned int i = 0; i < rknn_io_num.n_input; i++)
                {
                    input_attrs[i].index = i;
                    r = rknn_query(rknn_ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
                    
                    if(r != RKNN_SUCC)
                    {
                        printf("Failed to query rknn input tensor.\n");
                    }
                    else
                    {
                        print_rknn_tensor(&(input_attrs[i]));
                    }
                }
            }

            {
                printf("Model output tensors:\n");
                
                rknn_tensor_attr output_attrs[rknn_io_num.n_output];
                memset(output_attrs, 0, sizeof(output_attrs));
                
                for(unsigned int i = 0; i < rknn_io_num.n_output; i++)
                {
                    output_attrs[i].index = i;
                    r = rknn_query(rknn_ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));

                    if(r != RKNN_SUCC)
                    {
                        printf("Failed to query rknn output tensor.\n");
                    }
                    else
                    {
                        print_rknn_tensor(&(output_attrs[i]));
                    }
                }
            }
        }
    }

    // Preload hsunchi rknn model.
    printf("Loading hsunchi model ...\n");
    hsunchi_model = load_model(HSUNCHI_SSD_MODEL_PATH, &hsunchi_model_len);
    hr = rknn_init(&hsunchi_rknn_ctx, hsunchi_model, hsunchi_model_len, 0);
    
    if(hr < 0)
    {
        printf("rknn initialization failed! Returned %d.\n", r);
    }
    else
    {
        // Get Model Input Output Info
        hr = rknn_query(hsunchi_rknn_ctx, RKNN_QUERY_IN_OUT_NUM, &hsunchi_rknn_io_num, sizeof(hsunchi_rknn_io_num));
        
        if(hr != RKNN_SUCC)
        {
            printf("rknn query failed! Return %d.\n", hr);
        }
        else
        {
            printf("Model loaded successfully.\n");
            printf("Model input num: %d, output num: %d\n", hsunchi_rknn_io_num.n_input, hsunchi_rknn_io_num.n_output);

            {
                printf("Model input tensors:\n");
                
                rknn_tensor_attr input_attrs[hsunchi_rknn_io_num.n_input];
                memset(input_attrs, 0, sizeof(input_attrs));
                
                for(unsigned int i = 0; i < hsunchi_rknn_io_num.n_input; i++)
                {
                    input_attrs[i].index = i;
                    hr = rknn_query(hsunchi_rknn_ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
                    
                    if(hr != RKNN_SUCC)
                    {
                        printf("Failed to query rknn input tensor.\n");
                    }
                    else
                    {
                        print_rknn_tensor(&(input_attrs[i]));
                    }
                }
            }

            {
                printf("Model output tensors:\n");
                
                rknn_tensor_attr output_attrs[hsunchi_rknn_io_num.n_output];
                memset(output_attrs, 0, sizeof(output_attrs));
                
                for(unsigned int i = 0; i < hsunchi_rknn_io_num.n_output; i++)
                {
                    output_attrs[i].index = i;
                    hr = rknn_query(hsunchi_rknn_ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));

                    if(hr != RKNN_SUCC)
                    {
                        printf("Failed to query rknn output tensor.\n");
                    }
                    else
                    {
                        print_rknn_tensor(&(output_attrs[i]));
                    }
                }
            }
        }
    }

    pthread_t detect_thread;
    pthread_create(&detect_thread, NULL, detect_thread_handler, NULL);
    pthread_detach(detect_thread);

    while(1)
    {
        ::sleep(1);
    }

    // Release resources.
    jmpp_raw_stream_stop(CAMERA_ID, &raw_stream);
    jmpp_raw_stream_close(CAMERA_ID, &raw_stream);
    printf("JMPP raw stream released.\n");

    free(image_addr);
    printf("Image buffer released.\n");

    if(rknn_ctx)
    {
        rknn_destroy(rknn_ctx);
        printf("rknn context released.\n");
    }
    
    if(model)
    {
        free(model);
        printf("rknn model released.\n");
    }

    return 0;
}

// Define function.

int npu_config_json_get()
{
    JsonNode *cfg;
    const char *str;

    mpower_npu_enable = 0;
    mpower_2D_graphic_enable = 0;
    npu_resolution_width = 0;
    npu_resolution_height = 0;
    npu_color_space = IMAGE_TYPE_UNKNOW;

    // get first video channel 0 NPU configuration
    cfg = user_config_query_to_json("{\"mild\":{\"videoInput\":{\"0\":{\"npuStream\":{}}}}}");
    
    if(NULL == cfg)
    {
        printf("cannot find mild video input\n");
        return 0;
    }

    // video input 0
    str = json_object_get_string_field(cfg, "mild.videoInput.0.npuStream.status", NULL);
    
    if((str != NULL) && (strcmp(str, "okay") == 0))
    {
        mpower_npu_enable = 1;
    }

    mpower_2D_graphic_enable = json_object_get_int_field(cfg, "mild.videoInput.0.npuStream.2D", 0);
    str = json_object_get_string_field(cfg, "mild.videoInput.0.npuStream.resolution", NULL);
    npu_resolution_width = json_object_get_int_field(cfg, "mild.videoInput.0.npuStream.width", 0);

    if(npu_resolution_width == 0)
    {
        printf("something is wrong\n");
    }
    
    npu_resolution_height = json_object_get_int_field(cfg, "mild.videoInput.0.npuStream.height", 0);

    if(npu_resolution_height == 0)
    {
        printf("something is wrong\n");
    }

    str = json_object_get_string_field(cfg, "mild.videoInput.0.npuStream.imageType", NULL);

    if((str != NULL) && (strcmp(str, "NV12")) == 0)
    {
        npu_color_space = IMAGE_TYPE_NV12;
    }
    else if((str != NULL) && (strcmp(str, "NV16")) == 0)
    {
        npu_color_space = IMAGE_TYPE_NV16;
    }
    else if((str != NULL) && (strcmp(str, "YUV420P") == 0))
    {
        npu_color_space = IMAGE_TYPE_YUV420P;
    }
    else
    {
        printf("unsupported color space %s\n", str);
    }

    printf("MPOWER NPU video stream: %d x %d, color space %s, 2D graphic %d\n", npu_resolution_width, npu_resolution_height, str, mpower_2D_graphic_enable );
    
    // free memory
    json_object_put(cfg);

    return 0;
}


// callback on RGA channel.
// DO NOT do time comsuming operations in the callback.
// RK_MPI_MB_ReleaseBuffer() must be CALLED!
// The purpose of the callback is to get (copy) the data and get out.
// Data rate is coming at 30 FPS. If you do not need every frame,
// pick the frame you want and drop others.
// 
// If this function is blocked for too long, frame lost may happen.

void rga_chn12_frame_cb(MEDIA_BUFFER mb)
{
    int chn;
    int r;
    MB_IMAGE_INFO_S stImageInfo;

    memset(&stImageInfo, 0, sizeof(stImageInfo));
    chn = RK_MPI_MB_GetChannelID(mb);

    if(chn != 12)   // depending on the setup
    {
        printf("RGA channel invalid\n");
    }

    r = RK_MPI_MB_GetImageInfo(mb, &stImageInfo);

    if(r)
    {
        printf("Warn: Get image info failed! r = %d\n", r);
    }
    
    raw_stream_data_addr = (uint8_t  *) RK_MPI_MB_GetPtr(mb);
    raw_stream_data_size = RK_MPI_MB_GetSize(mb);
    raw_stream_data_timestamp = RK_MPI_MB_GetTimestamp(mb);
    raw_stream_data_width = stImageInfo.u32Width;
    raw_stream_data_height = stImageInfo.u32Height;
    raw_stream_data_color_space = stImageInfo.enImgType;

    // Do image and image info copy.
    image_size_hi = raw_stream_data_size;
    image_timestamp = raw_stream_data_timestamp;
    image_width = raw_stream_data_width;
    image_height = raw_stream_data_height;
    image_color_space = raw_stream_data_color_space;
    memcpy(image_addr, raw_stream_data_addr, image_size_hi);

    // MUST release memory as soon as possible!
    RK_MPI_MB_ReleaseBuffer(mb);

    ++image_id;
}


static void wait_jmpp_init_complete()
{
    int jmpp_api_sock;
    int delay = 100;
    JMPP_API_MSG_REQREP_S jmpp_api_msg;
    int r;
    
    if((jmpp_api_sock = nn_socket(AF_SP, NN_REQ)) < 0)
    {
        printf("nn_socket\n");
    }

    // set a 100ms message timeout
    if(nn_setsockopt(jmpp_api_sock, NN_SOL_SOCKET, NN_RCVTIMEO, &delay, sizeof(int)) < 0)
    {
        printf("nn_setsockopt\n");
    }
    
    if(nn_connect(jmpp_api_sock, JMPP_API_INTF_REQREP_URL) < 0)
    {
        printf("nn_connect\n");
    }


    while(1)
    {
        jmpp_api_msg.msg_type = JMPP_API_MSG_INIT_CMPL_STATUS_GET;
        jmpp_api_msg.rc = -1;
        jmpp_api_msg.data.status = JMPP_API_MSG_STATUS_UNKNOWN;
        
        if(nn_send(jmpp_api_sock, &jmpp_api_msg, sizeof(jmpp_api_msg), 0) < 0)
        {
            printf("nn_send\n");
            continue;
        }
        
        r = nn_recv(jmpp_api_sock, &jmpp_api_msg, sizeof(jmpp_api_msg), 0);
        printf("JMPP returned code = %d\n", nn_errno());
        
        if (r >= 0)
        {
            if(jmpp_api_msg.rc != 0)
            {
                // something is wrong in the message format
                printf("jmpp api msg rc != 0\n");
            }

            if(jmpp_api_msg.data.status == JMPP_API_MSG_STATUS_INIT_CMPL)
            {
                printf("JMPP init completed\n");
                break;
            }
        }

        printf("JMPP INIT CMPL msg status error r %d, rc %d, status %d\n", r, jmpp_api_msg.rc, jmpp_api_msg.data.status);

        // sleep 1 second and send message again
        ::sleep(1);
    }

    nn_close(jmpp_api_sock);
}


static unsigned char *load_model(const char *filename, int *model_size)
{
    int model_len = 0;
    unsigned char *model = nullptr;
    FILE *fp = fopen(filename, "rb");

    if(fp == nullptr)
    {
        printf("fopen %s fail!\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    model_len = ftell(fp);
    model = (unsigned char*)malloc(model_len);

    fseek(fp, 0, SEEK_SET);
    if(model_len != fread(model, 1, model_len, fp))
    {
        printf("fread %s fail!\n", filename);
        free(model);
        return NULL;
    }

    *model_size = model_len;
    if(fp)
    {
        fclose(fp);
    }

    return model;
}


void create_rknn_list(rknn_list **s)
{
    if (*s != NULL)
        return;
    *s = (rknn_list *)malloc(sizeof(rknn_list));
    (*s)->top = NULL;
    (*s)->size = 0;
    printf("create rknn_list success\n");
}

void destory_rknn_list(rknn_list **s)
{
    Node *t = NULL;
    if (*s == NULL)
        return;
    while ((*s)->top) {
        t = (*s)->top;
        (*s)->top = t->next;
        free(t);
    }
    free(*s);
    *s = NULL;
}

void rknn_list_push(rknn_list *s, long timeval, detect_result_group_t detect_result_group)
{
    Node *t = NULL;
    t = (Node *)malloc(sizeof(Node));
    t->timeval = timeval;
    t->detect_result_group = detect_result_group;
    if (s->top == NULL) {
        s->top = t;
        t->next = NULL;
    } 
    else {
        t->next = s->top;
        s->top = t;
    }
    s->size++;
}

void rknn_list_pop(rknn_list *s, long *timeval, detect_result_group_t *detect_result_group)
{
    Node *t = NULL;
    if (s == NULL || s->top == NULL) return;
    t = s->top;
    *timeval = t->timeval;
    *detect_result_group = t->detect_result_group;
    s->top = t->next;
    free(t);
    s->size--;
}

void rknn_list_drop(rknn_list *s)
{
    Node *t = NULL;
    if (s == NULL || s->top == NULL) return;
    t = s->top;
    s->top = t->next;
    free(t);
    s->size--;
}

int rknn_list_size(rknn_list *s)
{
    if (s == NULL)
        return -1;
    return s->size;
}

static void print_rknn_tensor(rknn_tensor_attr *attr)
{
    printf("index=%d name=%s n_dims=%d dims=[%d %d %d %d] n_elems=%d size=%d fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n",
            attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2], attr->dims[1], attr->dims[0],
            attr->n_elems, attr->size, 0, attr->type, attr->qnt_type, attr->fl, attr->zp, attr->scale);
}


int predict_one_pic(cv::Mat img, rknn_context ctx, rknn_input_output_num io_num)
{
    int ret, predict;
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = img.cols * img.rows * img.channels();
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = img.data;
    ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
    if (ret < 0)
    {
        throw std::runtime_error("rknn_inputs_set fail!");
    }
    ret = rknn_run(ctx, nullptr);
    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;
    ret = rknn_outputs_get(ctx, 1, outputs, NULL);
    if (ret < 0)
    {
        throw std::runtime_error("rknn_outputs_get fail!");
    }

    // Post Predict
    for (int i = 0; i < io_num.n_output; i++)
    {
        float *buffer = (float *)outputs[i].buf;
        if (buffer[0] < 0.5)
        {
            predict = 0;
        }
        else
        {
            predict = 1;
        }
    }
    // Release rknn_outputs
    rknn_outputs_release(ctx, 1, outputs);
    return predict;
}


static cv::Rect dlibRectangleToOpenCV(dlib::rectangle r)
{
    return cv::Rect(cv::Point2i(r.left(), r.top()), cv::Point2i(r.right() + 1, r.bottom() + 1));
}


static void *detect_thread_handler(void *arg)
{
    while(1)
    {
        if(image_id > image_id_prev)
        {
            long timer_start = getCurrentTimeMsec();
            int bbox_center_x = 0, bbox_center_y = 0;
            int count_cars = 0;
            int r = 0;
            double e = 0;
            struct json_object *HMW = NULL, *objectMark = NULL, *objectMark_enable = NULL;
            
            image_id_prev = image_id;
            //printf("[%d] Detecting objects in image.\n", image_id_prev);

            memcpy(image_drawing_addr, image_addr, image_size_hi);
            crop_hFlip_image(image_drawing_addr, image_drawing_cropped_addr);

            // Set input image.
            rknn_input input[1];
            memset(input, 0, sizeof(input));
            input[0].index = 0;
            input[0].type = RKNN_TENSOR_UINT8;
            input[0].size = RGA_BUFFER_SCALED_SIZE;
            input[0].fmt = RKNN_TENSOR_NHWC;
            input[0].buf = image_drawing_cropped_addr;

            // Draw boundary of car detect zone.
            if(get_config_HMW_objectMark_enable())
            {
                draw_border((char *)image_drawing_cropped_addr,
                                MODEL_INPUT_SIZE,
                                MODEL_INPUT_SIZE,
                                MODEL_DETECT_BOUNDARY_LEFT,
                                MODEL_DETECT_BOUNDARY_TOP,
                                MODEL_DETECT_BOUNDARY_RIGHT - MODEL_DETECT_BOUNDARY_LEFT,
                                MODEL_DETECT_BOUNDARY_BOTTOM - MODEL_DETECT_BOUNDARY_TOP,
                                COLOR_BLUE);
            }

            r = rknn_inputs_set(rknn_ctx, rknn_io_num.n_input, input);
            
            if(r < 0)
            {
                printf("Failed to set rknn input! Returned %d.\n", r);
            }
            else
            {
                // Run rknn.
                r = rknn_run(rknn_ctx, NULL);

                if(r < 0)
                {
                    printf("Failed to run rknn! Returned %d.\n", r);
                }
                else
                {
                    // Get Output.
                    rknn_output outputs[2];
                    memset(outputs, 0, sizeof(outputs));
                    outputs[0].want_float = 1;
                    outputs[1].want_float = 1;
                    r = rknn_outputs_get(rknn_ctx, rknn_io_num.n_output, outputs, NULL);

                    if(r < 0)
                    {
                        printf("Failed to get rknn output! Returned %d.\n", r);
                    }
                    else
                    {
                        // Deal with output.
                        detect_result_group_t detect_result_group;
                        postProcessSSD((float*)(outputs[0].buf), (float*)(outputs[1].buf), MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, &detect_result_group);
                        
                        // Release rknn_outputs.
                        rknn_outputs_release(rknn_ctx, 2, outputs);

                        int pixel_distance = 0;
	                    int min_pixel_distance = MODEL_INPUT_SIZE;
                        double min_meter_distance = 0;
                        int min_distance_boundary_top = 0, min_distance_boundary_bottom = 0, min_distance_boundary_left = 0, min_distance_boundary_right = 0;
                        int sock = 0, sock_return = 0;
                        NYCU_MHW_ALARM_EVENT_s ae;

                        // Print detected objects.
                        for(int i = 0; i < detect_result_group.count; i++)
                        {
                            detect_result_t *det_result = &(detect_result_group.results[i]);

                            // Calculate center of the bbox.
                            bbox_center_x = (det_result->box.left + det_result->box.right) / 2;
                            bbox_center_y = (det_result->box.top + det_result->box.bottom) / 2;

                            // It is a car &&
                            // center of the bbox is located in detection zone &&
                            // bbox width >= 60% detection zone size.
                            if(strcmp("car", det_result->name) == 0 &&
                                bbox_center_x >= MODEL_DETECT_BOUNDARY_LEFT &&
                                bbox_center_x <= MODEL_DETECT_BOUNDARY_RIGHT &&
                                bbox_center_y >= MODEL_DETECT_BOUNDARY_TOP &&
                                bbox_center_y <= MODEL_DETECT_BOUNDARY_BOTTOM &&
                                (det_result->box.right - det_result->box.left) >= (MODEL_DETECT_BOUNDARY_RIGHT - MODEL_DETECT_BOUNDARY_LEFT) * 0.6)
                            {
                                ++count_cars;

                                printf("%s @ (top=%d, right=%d, bottom=%d, left=%d) confidence=%f\n",
                                    det_result->name,
                                    det_result->box.top,
                                    det_result->box.right,
                                    det_result->box.bottom,
                                    det_result->box.left,
                                    det_result->prop);

                                // Draw annotation on image.
                                if(get_config_HMW_objectMark_enable())
                                {
                                    int x = det_result->box.left;
                                    int y = det_result->box.top;
                                    int w = det_result->box.right - det_result->box.left;
                                    int h = det_result->box.bottom - det_result->box.top;

                                    draw_border((char *)image_drawing_cropped_addr, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, x, y, w, h, COLOR_RED);
                                }

                                // Get min distance.
                                pixel_distance = MODEL_INPUT_SIZE - det_result->box.bottom;

                                if(pixel_distance < min_pixel_distance)
                                {
                                    min_pixel_distance = pixel_distance;

                                    min_distance_boundary_top = det_result->box.top;
                                    min_distance_boundary_bottom = det_result->box.bottom;
                                    min_distance_boundary_left = det_result->box.left;
                                    min_distance_boundary_right = det_result->box.right;

                                    e = sqrt(pow(bbox_center_x - MODEL_INPUT_SIZE / 2.0, 2.0) + pow(bbox_center_y - MODEL_INPUT_SIZE / 2.0, 2.0));
                                }
                            }

                            // It is a car.
                            else if(strcmp("car", det_result->name) == 0)
                            {
                                // Draw annotation on image.
                                if(get_config_HMW_objectMark_enable())
                                {
                                    int x = det_result->box.left;
                                    int y = det_result->box.top;
                                    int w = det_result->box.right - det_result->box.left;
                                    int h = det_result->box.bottom - det_result->box.top;

                                    draw_border((char *)image_drawing_cropped_addr, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, x, y, w, h, COLOR_GREEN);
                                }
                            }
                        }

                        // Print object detect result on terminal and truncate rknn list.
                        if(count_cars)
                        {
                            printf("%d car(s) found in the boundary.\n", count_cars);

                            // Save this detection result to linked list.
                            rknn_list_push(rknn_list_all_, getCurrentTimeMsec(), detect_result_group);

                            if(rknn_list_size(rknn_list_all_) >= MAX_RKNN_LIST_NUM)
                            {
                                rknn_list_drop(rknn_list_all_);
                            }
                        }


                        // No min distance.
                        if(min_pixel_distance >= MODEL_INPUT_SIZE)
                        {
                            min_pixel_distance = -1;
                            min_meter_distance = -1;

                            min_distance_boundary_top = -1;
                            min_distance_boundary_bottom = -1;
                            min_distance_boundary_left = -1;
                            min_distance_boundary_right = -1;
                        }

                        // Calculate min distance from pixels to meters.
                        else
                        {
                            min_meter_distance = sqrt(pow(e*VEHICLE_WIDTH_METER/(min_distance_boundary_right-min_distance_boundary_left), 2.0) + pow(MODEL_INPUT_SIZE*VEHICLE_WIDTH_METER/(2.0*(min_distance_boundary_right-min_distance_boundary_left)*TAN_OF_HHVIEW), 2.0));

                            printf("Min Distance = %d pixels\n", min_pixel_distance);
                            printf("               %.2lf meters\n", min_meter_distance);
                            printf("               @ (top=%d, right=%d, bottom=%d, left=%d)\n", min_distance_boundary_top, min_distance_boundary_right, min_distance_boundary_bottom, min_distance_boundary_left);
                        }


                        // Put min distance value on image.
                        if(get_config_HMW_objectMark_enable())
                        {
                            char min_meter_distance_str[100];
                            sprintf(min_meter_distance_str, "%.2lf", min_meter_distance);

                            put_text((char *)image_drawing_cropped_addr, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 15, 15, min_meter_distance_str);
                        }

                        // Write annotated image to disk.
                        if(get_config_HMW_recording_enable())
                        {
                            FILE* fout = nullptr;
                            char filename[100] = "";
                            char output_path[100] = "";

                            strcat(output_path, get_config_HMW_recording_path());

                            // Remove trailing slash.
                            if(output_path[strlen(output_path) - 1] == '/')
                            {
                                output_path[strlen(output_path) - 1] = '\0';
                            }

                            sprintf(filename, "%s/%d.raw", output_path, image_id_prev);
                            fout = fopen(filename, "wb");
                            fwrite(image_drawing_cropped_addr, 1, RGA_BUFFER_SCALED_SIZE, fout);
                            fclose(fout);
                        }

                        long timer_interval = getCurrentTimeMsec() - timer_start;
                        printf("[%d] process takes %ld ms.\n", image_id_prev, timer_interval);
                        
                        
                        // Publish by nanomsg.
                        if((sock = nn_socket(AF_SP, NN_PUB)) < 0)
                        {
                            printf("Data is NOT published due to socket error chkpt 1.\n");
                        }
                        else
                        {
                            if(nn_bind(sock, "URL") < 0)
                            {
                                printf("Data is NOT published due to socket error chkpt 2.\n");
                            }
                            else
                            {
                                ae.frame_dimension.width = MODEL_INPUT_SIZE;
                                ae.frame_dimension.height = MODEL_INPUT_SIZE;
                                ae.front_vehicle_coord.top_x = min_distance_boundary_left;
                                ae.front_vehicle_coord.top_x = min_distance_boundary_top;
                                ae.front_vehicle_coord.bottom_x = min_distance_boundary_right;
                                ae.front_vehicle_coord.bottom_y = min_distance_boundary_bottom;
                                gettimeofday(&ae.timestamp, NULL);
                                ae.front_vehicle_distance = (float)min_meter_distance;
                                ae.car_speed = -1;
                                
                                sock_return = nn_send(sock, &ae, sizeof(ae), 0);
                                
                                if(sock_return < 0)
                                {
                                    printf("Data is NOT published due to socket error chkpt 3.\n");
                                }
                                else
                                {
                                    printf("Data is published successfully.\n");
                                }
                            }
                        }
                    }
                }
            }

            // hsunchi code.
            const int MODEL_IN_WIDTH = 96;
            const int MODEL_IN_HEIGHT = 96;
            const int MODEL_IN_CHANNELS = 3;
            
            int raw_image_width = 532;
            int raw_image_height = 300;

            unsigned char *src_data = image_drawing_addr;

            try
            {

                dlib::array2d<dlib::rgb_pixel> img(raw_image_height, raw_image_width);

                for (int r = 0; r < raw_image_height; r++)
                {
                    for (int c = 0; c < raw_image_width; c++)
                    {
                        img[r][c] = dlib::rgb_pixel(src_data[r * raw_image_width * 3 + c * 3 + 0],
                                                    src_data[r * raw_image_width * 3 + c * 3 + 1],
                                                    src_data[r * raw_image_width * 3 + c * 3 + 2]);
                    }
                }

                // save_png(img,"output.png");

                frontal_face_detector detector = get_frontal_face_detector();
                shape_predictor sp;
                deserialize("./model/shape_predictor_5_face_landmarks.dat") >> sp;
                printf("processing image \n");

                std::vector<rectangle> dets = detector(img);
                printf("Number of faces detected: %d\n", dets.size());

                std::vector<full_object_detection> shapes;
                for (unsigned long j = 0; j < dets.size(); ++j)
                {
                    full_object_detection shape = sp(img, dets[j]);
                    cout << "number of parts: " << shape.num_parts() << endl;
                    // rect (左, 上,右,下)
                    // 右眼邊界 0-右 1-左
                    int length = (shape.part(0).x() - shape.part(1).x()) / 2 + 5;
                    rectangle rightrec(shape.part(1).x(), shape.part(1).y() - length, shape.part(0).x(), shape.part(0).y() + length);
                    cv::Rect right_rec = dlibRectangleToOpenCV(rightrec);
                    // 左眼邊界  2-右 3-左
                    length = (shape.part(2).x() - shape.part(3).x()) / 2 + 5;
                    rectangle leftrec(shape.part(3).x(), shape.part(3).y() - length, shape.part(2).x(), shape.part(2).y() + length);
                    cv::Rect left_rec = dlibRectangleToOpenCV(leftrec);

                    // 畫在原本讀近來的那張圖上
                    // draw_rectangle(img, leftrec, dlib::rgb_pixel(255, 0, 0), 1);
                    // draw_rectangle(img, rightrec, dlib::rgb_pixel(255, 0, 0), 1);
                    cv::Mat org_img = dlib::toMat(img);
                    cv::Mat right_eye = org_img(right_rec);
                    cv::Mat left_eye = org_img(left_rec);
                    // to 96X96
                    cv::resize(right_eye, right_eye, cv::Size(MODEL_IN_WIDTH, MODEL_IN_HEIGHT), (0, 0), (0, 0), cv::INTER_LINEAR);
                    cv::resize(left_eye, left_eye, cv::Size(MODEL_IN_WIDTH, MODEL_IN_HEIGHT), (0, 0), (0, 0), cv::INTER_LINEAR);

                    if (!left_eye.data)
                    {
                        printf("No left eyes detect");
                    }
                    else
                    {
                        // rknn_input_output_num io_num = print_model_info(ctx);
                        int res = predict_one_pic(left_eye, hsunchi_rknn_ctx, hsunchi_rknn_io_num);
                        if (res == 1)
                        {
                            cout << "Left eyes open" << endl;
                        }
                        else
                        {
                            cout << "Left eyes close" << endl;
                        }
                    }

                    if (!right_eye.data)
                    {
                        printf("No right eyes detect");
                    }
                    else
                    {
                        // rknn_input_output_num io_num = print_model_info(ctx);
                        int res = predict_one_pic(right_eye, hsunchi_rknn_ctx, hsunchi_rknn_io_num);
                        if (res == 1)
                        {
                            cout << "Right eyes open" << endl;
                        }
                        else
                        {
                            cout << "Right eyes close" << endl;
                        }
                    }
                }
                
                // cout << "Hit enter to process the next image..." << endl;
                // cin.get();
            }
            catch (exception& e)
            {
                cout << "\nexception thrown!" << endl;
                cout << e.what() << endl;
            }
        }

        // Program is tend to go to death without this.
        usleep(100);
    }
}


void draw_border(char* image_addr, int image_width, int image_height, int x, int y, int w, int h, Color stroke_color)
{
    // Top.
    for(int i=x; i<x+w; ++i)
    {
        int j = y;

        // Set pixel (j, i) to red.
        image_addr[j*image_width*3 + i*3 + 0] = stroke_color.blue;
        image_addr[j*image_width*3 + i*3 + 1] = stroke_color.green;
        image_addr[j*image_width*3 + i*3 + 2] = stroke_color.red;
    }

    // Bottom.
    for(int i=x; i<x+w; ++i)
    {
        int j = y + h;

        // Set pixel (j, i) to red.
        image_addr[j*image_width*3 + i*3 + 0] = stroke_color.blue;
        image_addr[j*image_width*3 + i*3 + 1] = stroke_color.green;
        image_addr[j*image_width*3 + i*3 + 2] = stroke_color.red;
    }

    // Left.
    for(int j=y; j<y+h; ++j)
    {
        int i = x;

        // Set pixel (j, i) to red.
        image_addr[j*image_width*3 + i*3 + 0] = stroke_color.blue;
        image_addr[j*image_width*3 + i*3 + 1] = stroke_color.green;
        image_addr[j*image_width*3 + i*3 + 2] = stroke_color.red;
    }

    // Right.
    for(int j=y; j<y+h; ++j)
    {
        int i = x + w;

        // Set pixel (j, i) to red.
        image_addr[j*image_width*3 + i*3 + 0] = stroke_color.blue;
        image_addr[j*image_width*3 + i*3 + 1] = stroke_color.green;
        image_addr[j*image_width*3 + i*3 + 2] = stroke_color.red;
    }
}


void put_text(char* image_addr, int image_width, int image_height, int x, int y, char* text)
{
    int char_select = -1;
    int R = 255, G = 0, B = 0;

    for(int i=0; i<strlen(text); ++i)
    {
        if(text[i] >= '0' && text[i] <= '9')
        {
            char_select = text[i] - '0';
        }
        else if(text[i] == '.')
        {
            char_select = 10;
        }
        else if(text[i] == '-')
        {
            char_select = 11;
        }

        if(char_select >= 0)
        {
            // row.
            for(int j=0; j<DIGIT_SIZE; ++j)
            {
                // column.
                for(int k=0; k<DIGIT_SIZE; ++k)
                {
                    if(DIGIT_ARRAYS[char_select][j][k] == 0)
                    {
                        image_addr[(j+y)*image_width*3 + (k+x)*3 + 0] = B;
                        image_addr[(j+y)*image_width*3 + (k+x)*3 + 1] = G;
                        image_addr[(j+y)*image_width*3 + (k+x)*3 + 2] = R;
                    }
                }
            }
        }

        x += DIGIT_SIZE;
    }
}


void crop_hFlip_image(uint8_t* source_image, uint8_t* dest_image)
{
    // row.
    for(int i=0; i<RGA_OUTPUT_HEIGHT; ++i)
    {
        // column.
        for(int j=0; j<RGA_OUTPUT_HEIGHT; ++j)
        {
            // RGB.
            for(int k=0; k<3; ++k)
            {
                dest_image[i*MODEL_INPUT_SIZE*3 + (RGA_OUTPUT_HEIGHT-j-1)*3 + k] = source_image[i*RGA_OUTPUT_WIDTH*3 + (RGA_OUTPUT_CROP_OFFSET+j)*3 + k];
            }
        }
    }
}


static long getCurrentTimeMsec()
{
    long msec = 0;
    char str[20] = {0};
    struct timeval stuCurrentTime;

    gettimeofday(&stuCurrentTime, NULL);

    sprintf(str, "%ld%03ld", stuCurrentTime.tv_sec, (stuCurrentTime.tv_usec) / 1000);

    for(size_t i = 0; i < strlen(str); i++)
    {
        msec = msec * 10 + (str[i] - '0');
    }

    return msec;
}
