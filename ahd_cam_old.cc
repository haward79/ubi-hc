
#include <stdio.h>
#include <stdlib.h>
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

#include "rknn_api.h"
#include "ssd.h"

// Set this to image size.
#define MODEL_INPUT_SIZE 300

const int CAMERA_ID = 0;  // camera id 0
const char* SSD_MODEL_PATH = "model/ssd_inception_v2_rv1109_rv1126.rknn";
const int RGA_BUFFER_MODEL_INPUT_SIZE = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 3;  // MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 3 = 270000
const int MAX_RKNN_LIST_NUM = 5;
const int MAX_IMAGE_LIST_NUM = 5;


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

uint32_t        image_size = 0;
uint8_t*        image_addr         = (uint8_t*)malloc(RGA_BUFFER_MODEL_INPUT_SIZE);
uint8_t*        image_drawing_addr = (uint8_t*)malloc(RGA_BUFFER_MODEL_INPUT_SIZE);
                                                        // This memory will be released at the end of main function.
uint64_t        image_timestamp = 0;
uint32_t        image_width = 0;
uint32_t        image_height = 0;
IMAGE_TYPE_E    image_color_space = IMAGE_TYPE_UNKNOW;

int image_id = -1;
int image_id_prev = -1;

rknn_context rknn_ctx;
rknn_input_output_num rknn_io_num;


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

rknn_list *rknn_list_;


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
static void *detect_thread_handler(void *);
static void *distance_thread_handler(void *);
void draw_border(char*, int, int, int, int, int, int);
static long getCurrentTimeMsec();



// Main

int main()
{
    #define NUM_NPU_FD 2
    int r = 0;
    int model_len = 0;
    unsigned char *model = nullptr;

    create_rknn_list(&rknn_list_);
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
        stRgaAttr.stImgIn.u32Width = 416;
        stRgaAttr.stImgIn.u32Height = 416;
        stRgaAttr.stImgIn.u32HorStride = 416;
        stRgaAttr.stImgIn.u32VirStride = 416;
        
        stRgaAttr.stImgOut.u32X = 0;
        stRgaAttr.stImgOut.u32Y = 0;
        stRgaAttr.stImgOut.imgType = IMAGE_TYPE_RGB888;
        stRgaAttr.stImgOut.u32Width = MODEL_INPUT_SIZE;
        stRgaAttr.stImgOut.u32Height = MODEL_INPUT_SIZE;
        stRgaAttr.stImgOut.u32HorStride = MODEL_INPUT_SIZE;
        stRgaAttr.stImgOut.u32VirStride = MODEL_INPUT_SIZE;

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

    // sleep(1);
    // FILE* fout = fopen("image.raw", "wb");
    // fwrite(image_addr, 1, image_size, fout);
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

    pthread_t detect_thread;
      pthread_create(&detect_thread, NULL, detect_thread_handler, NULL);
    pthread_join(detect_thread, NULL);

    pthread_t distance_thread;
      pthread_create(&distance_thread, NULL, distance_thread_handler, NULL);
    pthread_join(distance_thread, NULL);

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
    image_size = raw_stream_data_size;
    image_timestamp = raw_stream_data_timestamp;
    image_width = raw_stream_data_width;
    image_height = raw_stream_data_height;
    image_color_space = raw_stream_data_color_space;
    memcpy(image_addr, raw_stream_data_addr, image_size);

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
        sleep(1);
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


static void *detect_thread_handler(void *arg)
{
    while(1)
    {
        if(image_id > image_id_prev)
        {
            long timer_start = getCurrentTimeMsec();
            bool has_car = false;
            int r = 0;
            
            image_id_prev = image_id;
            printf("[%d] Detecting objects in image.\n", image_id_prev);

            memcpy(image_drawing_addr, image_addr, image_size);

            // Set input image.
            rknn_input input[1];
            memset(input, 0, sizeof(input));
            input[0].index = 0;
            input[0].type = RKNN_TENSOR_UINT8;
            input[0].size = RGA_BUFFER_MODEL_INPUT_SIZE;
            input[0].fmt = RKNN_TENSOR_NHWC;
            input[0].buf = image_drawing_addr;

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
                        printf("Post Process of SSD passed.\n");

                        // Release rknn_outputs.
                        rknn_outputs_release(rknn_ctx, 2, outputs);

                        // Print detected objects.
                        for(int i = 0; i < detect_result_group.count; i++)
                        {
                            detect_result_t *det_result = &(detect_result_group.results[i]);

                            // if(strcmp("car", det_result->name) == 0)
                            if(1)
                            {
                                has_car = true;

                                printf("%s @ (%d %d %d %d) %f\n",
                                    det_result->name,
                                    det_result->box.top,
                                    det_result->box.right,
                                    det_result->box.bottom,
                                    det_result->box.left,
                                    det_result->prop);

                                // Draw annotation on image.
                                int x = det_result->box.left;
                                int y = det_result->box.top;
                                int w = det_result->box.right - det_result->box.left;
                                int h = det_result->box.bottom - det_result->box.top;

                                draw_border((char *)image_drawing_addr, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, x, y, w, h);
                            }
                        }

                        // Write annotated image to disk.
                        FILE* fout = nullptr;
                        char filename[100];
                        sprintf(filename, "/root/output/%d.raw", image_id_prev);
                        fout = fopen(filename, "wb");
                        fwrite(image_drawing_addr, 1, MODEL_INPUT_SIZE*MODEL_INPUT_SIZE*3, fout);
                        fclose(fout);

                        if(has_car)
                        {
                            // Save this detection result to linked list.
                            rknn_list_push(rknn_list_, getCurrentTimeMsec(), detect_result_group);

                            if(rknn_list_size(rknn_list_) >= MAX_RKNN_LIST_NUM)
                            {
                                rknn_list_drop(rknn_list_);
                            }
                        }

                        long timer_interval = getCurrentTimeMsec() - timer_start;
                        printf("[%d] process takes %ld ms.\n", image_id_prev, timer_interval);
                    }
                }
            }
        }

        // Program is tend to go to death without this.
        usleep(100);
    }
}


static void *distance_thread_handler(void *arg)
{
    long timeval = 0;
    detect_result_group_t* detections = nullptr;
    detect_result_t* detection = nullptr;
	
	const int MAX_CAR_POSITIONS = 10;
	int car_positions[MAX_CAR_POSITIONS];
	int car_positions_index = 0;

    if(rknn_list_size > 0)
    {
        rknn_list_pop(rknn_list_, &timeval, detections);

        for(int i = 0; i < detections->count; ++i)
        {
            detection = &(detections->results[i]);

            if(strcmp("car", detection->name) == 0)
            {
                printf("Car founed...............................................\n");

				if(car_positions_index < MAX_CAR_POSITIONS)
				{
					car_positions[car_positions_index++] = detection->box.top;
				}
            }

            /*
            printf("%s @ (%d %d %d %d) %f\n",
                detection->name,
                detection->box.top,
                detection->box.right,
                detection->box.bottom,
                detection->box.left,
                detection->prop);
            */
        }
    }
}


void draw_border(char* image_drawing_addr, int image_width, int image_height, int x, int y, int w, int h)
{
    int R = 255, G = 0, B = 0;

    // Top.
    for(int i=x; i<x+w; ++i)
    {
        int j = y;

        // Set pixel (j, i) to red.
        image_drawing_addr[j*image_width*3 + i*3 + 0] = B;
        image_drawing_addr[j*image_width*3 + i*3 + 1] = G;
        image_drawing_addr[j*image_width*3 + i*3 + 2] = R;
    }

    // Bottom.
    for(int i=x; i<x+w; ++i)
    {
        int j = y + h;

        // Set pixel (j, i) to red.
        image_drawing_addr[j*image_width*3 + i*3 + 0] = B;
        image_drawing_addr[j*image_width*3 + i*3 + 1] = G;
        image_drawing_addr[j*image_width*3 + i*3 + 2] = R;
    }

    // Left.
    for(int j=y; j<y+h; ++j)
    {
        int i = x;

        // Set pixel (j, i) to red.
        image_drawing_addr[j*image_width*3 + i*3 + 0] = B;
        image_drawing_addr[j*image_width*3 + i*3 + 1] = G;
        image_drawing_addr[j*image_width*3 + i*3 + 2] = R;
    }

    // Right.
    for(int j=y; j<y+h; ++j)
    {
        int i = x + w;

        // Set pixel (j, i) to red.
        image_drawing_addr[j*image_width*3 + i*3 + 0] = B;
        image_drawing_addr[j*image_width*3 + i*3 + 1] = G;
        image_drawing_addr[j*image_width*3 + i*3 + 2] = R;
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
