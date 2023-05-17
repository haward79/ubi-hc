/****************************************************************************
*
*    Copyright (c) 2017 - 2018 by Rockchip Corp.  All rights reserved.
*
*    The material in this file is confidential and contains trade secrets
*    of Rockchip Corporation. This is proprietary information owned by
*    Rockchip Corporation. No part of this work may be disclosed,
*    reproduced, copied, transmitted, or used in any way for any purpose,
*    without the express written permission of Rockchip Corporation.
*
*****************************************************************************/

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <string.h>
#include <signal.h>
#include <pthread.h>

#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/timerfd.h>
#include <sys/time.h>
#include <assert.h>

#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <nanomsg/nn.h>
#include <nanomsg/pipeline.h>
#include <nanomsg/pubsub.h>
#include <nanomsg/reqrep.h>
#include <userconfig.h>
#include <jmpp/jmpp.h>
#include <jmpp/jmpp_npu_api.h>
#include <jmpp/jmpp_vo_rgb_api.h>
#include <jmpp/jmpp_rga_api.h>
#include <jmpp/jmpp_raw_stream_api.h>
#include <jmpp/jmpp_api.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#include "rknn_api.h"
#include "ssd.h"

#define NUM_NPU_FD 2
#define MODEL_INPUT_SIZE 300

using namespace std;
using namespace cv;

/*-------------------------------------------
                  Functions
-------------------------------------------*/

/*! Depending on the video decoder pipeline, video input color space can either be NV12 or NV16 */

int mpower_npu_enable;          // mpower process enable or disable
int mpower_2D_graphic_enable;   // mpower helps do some 2D drawing on the frame

int npu_resolution_width;
int npu_resolution_height;
IMAGE_TYPE_E  npu_color_space;

int config_raw_resolution_width;    // raw stream in mild.json
int config_raw_resolution_height;   // raw stream in mild.json
int config_raw_color_space;         // raw stream in mild.json

JMPP_NPU_STREAM_S stream;    // do not modify this data or read structure data directly.
                             // always use API to access information.
                             // this may take a large memory. declare as a global variable


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

static FILE *output_file = NULL;
static uint32_t frame_count = 0;
static char* ssd_path = "model/ssd_inception_v2_rv1109_rv1126.rknn";

static int config_json_get();        // getting NPU and RAW stream configuration
static void wait_jmpp_init_complete();
void rga_chn12_frame_cb(MEDIA_BUFFER mb);


// callback on RGA channel.
// DO NOT do time comsuming operations in the callback.
// RK_MPI_MB_ReleaseBuffer() must be CALLED!
// The purpose of the callback is to get (copy) the data and get out.
// Data rate is coming at 30 FPS. If you do not need every frame,
// pick the frame you want and drop others.
// 
// If this function is blocked for too long, frame lost may happen.

static void printRKNNTensor(rknn_tensor_attr *attr) {
    printf("index=%d name=%s n_dims=%d dims=[%d %d %d %d] n_elems=%d size=%d fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n", 
            attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2], attr->dims[1], attr->dims[0], 
            attr->n_elems, attr->size, 0, attr->type, attr->qnt_type, attr->fl, attr->zp, attr->scale);
}

static unsigned char *load_model(const char *filename, int *model_size){
    FILE *fp = fopen(filename, "rb");
    if(fp == nullptr) {
        printf("fopen %s fail!\n", filename);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    unsigned char *model = (unsigned char*)malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if(model_len != fread(model, 1, model_len, fp)) {
        printf("fread %s fail!\n", filename);
        free(model);
        return NULL;
    }
    *model_size = model_len;
    if(fp) {
        fclose(fp);
    }
    return model;
}

void rga_chn12_frame_cb(MEDIA_BUFFER mb){

	int chn;
	int r;
	MB_IMAGE_INFO_S stImageInfo;

	memset(&stImageInfo, 0, sizeof(stImageInfo));
	
	chn = RK_MPI_MB_GetChannelID(mb);

	if (chn != 12)   // depending on the setup
		printf("RGA channel invalid\n");

	r = RK_MPI_MB_GetImageInfo(mb, &stImageInfo);

	if (r) printf("Warn: Get image info failed! r = %d\n", r);
	
	raw_stream_data_addr = (uint8_t  *) RK_MPI_MB_GetPtr(mb);
	
	raw_stream_data_size = RK_MPI_MB_GetSize(mb);

	raw_stream_data_timestamp = RK_MPI_MB_GetTimestamp(mb);

	raw_stream_data_width = stImageInfo.u32Width;
	raw_stream_data_height = stImageInfo.u32Height;
	raw_stream_data_color_space = stImageInfo.enImgType;

	
	if ((output_file) && (frame_count == 0)){
		fwrite(raw_stream_data_addr, 1, raw_stream_data_size, output_file);
		fclose(output_file);		
	}


	++frame_count;
	
	printf("!!Get RGB888 frame:ptr:%p, size:%zu, mode:%d, channel:%d, "
	       "timestamp:%lld, ImgInfo:<wxh %dx%d, fmt 0x%x>\n",
	       RK_MPI_MB_GetPtr(mb), RK_MPI_MB_GetSize(mb),
	       RK_MPI_MB_GetModeID(mb), RK_MPI_MB_GetChannelID(mb),
	       RK_MPI_MB_GetTimestamp(mb), stImageInfo.u32Width,
	       stImageInfo.u32Height, stImageInfo.enImgType);
	
	// MUST release memory as soon as possible!
	
	RK_MPI_MB_ReleaseBuffer(mb);
	
}

static void wait_jmpp_init_complete(){

	int jmpp_api_sock;

	int delay = 100;
	
	JMPP_API_MSG_REQREP_S jmpp_api_msg;

	int r;
	
	if ((jmpp_api_sock = nn_socket(AF_SP, NN_REQ)) < 0) printf("nn_socket\n");

	// set a 100ms message timeout
	
        if (nn_setsockopt(jmpp_api_sock, NN_SOL_SOCKET, NN_RCVTIMEO, &delay, sizeof(int)) < 0) printf("nn_setsockopt\n");
	
	if (nn_connect (jmpp_api_sock, JMPP_API_INTF_REQREP_URL) < 0) printf("nn_connect\n");


	while(1){
		jmpp_api_msg.msg_type = JMPP_API_MSG_INIT_CMPL_STATUS_GET;
		jmpp_api_msg.rc = -1;
		jmpp_api_msg.data.status = JMPP_API_MSG_STATUS_UNKNOWN;
		
		if (nn_send(jmpp_api_sock, &jmpp_api_msg, sizeof(jmpp_api_msg), 0) < 0){
			printf("nn_send\n");
			continue;
		}
		
		r = nn_recv(jmpp_api_sock, &jmpp_api_msg, sizeof(jmpp_api_msg), 0);
		
		if (r >= 0){
			if (jmpp_api_msg.rc != 0){
				// something is wrong in the message format
				
				printf("jmpp api msg rc != 0\n");
				continue;
			}

			if (jmpp_api_msg.data.status == JMPP_API_MSG_STATUS_INIT_CMPL){
				printf("JMPP init completed\n");
				break;
			}
		}

		
		printf("JMPP INIT CMPL msg status error r %d, rc %d, status %d\n",
		       r,
		       jmpp_api_msg.rc,
		       jmpp_api_msg.data.status);
               // sleep 1 second and send message again
		
		sleep(1);
        }

	nn_close(jmpp_api_sock);
}

int config_json_get(){

	JsonNode *cfg;
	const char *str;

	mpower_npu_enable = 0;
	mpower_2D_graphic_enable = 0;
	npu_resolution_width = 0;
	npu_resolution_height = 0;
	npu_color_space = IMAGE_TYPE_UNKNOW;

	// get first video channel 0 NPU configuration
	
	cfg = user_config_query_to_json("{\"mild\":{\"videoInput\":{\"0\":{}}}}");
	
	if (NULL == cfg){
		printf("cannot find mild video input\n");
		return 0;
	}

	// video input 0
	
	str = json_object_get_string_field(cfg, "mild.videoInput.0.npuStream.status", NULL);
	
	if ((str != NULL) && (strcmp(str, "okay") == 0)) mpower_npu_enable = 1;

	mpower_2D_graphic_enable = json_object_get_int_field(cfg, "mild.videoInput.0.npuStream.2D", 0);

	str = json_object_get_string_field(cfg, "mild.videoInput.0.npuStream.resolution", NULL);

	npu_resolution_width = json_object_get_int_field(cfg, "mild.videoInput.0.npuStream.width", 0);
	if (npu_resolution_width == 0) printf("something is wrong\n");
	
	npu_resolution_height = json_object_get_int_field(cfg, "mild.videoInput.0.npuStream.height", 0);
	if (npu_resolution_height == 0) printf("something is wrong\n");

	str = json_object_get_string_field(cfg, "mild.videoInput.0.npuStream.imageType", NULL);
	
	if ((str != NULL) && (strcmp(str, "NV12")) == 0) npu_color_space = IMAGE_TYPE_NV12;
	else if ((str != NULL) && (strcmp(str, "NV16")) == 0) npu_color_space = IMAGE_TYPE_NV16;
	else if ((str != NULL) && (strcmp(str, "YUV420P") == 0)) npu_color_space = IMAGE_TYPE_YUV420P;
	else printf("unsupported color space %s\n", str);

	printf("NPU video stream: %dx%d, color space %s, 2D graphic %d\n",
	       npu_resolution_width,
	       npu_resolution_height, str,
	       mpower_2D_graphic_enable );

	
	// raw stream parsing

	config_raw_resolution_width = json_object_get_int_field(cfg, "mild.videoInput.0.rawStream.width", 0);
	if (config_raw_resolution_width == 0) printf("something is wrong\n");
	
	config_raw_resolution_height = json_object_get_int_field(cfg, "mild.videoInput.0.rawStream.height", 0);
	if (config_raw_resolution_height == 0) printf("something is wrong\n");

	str = json_object_get_string_field(cfg, "mild.videoInput.0.rawStream.imageType", NULL);
	if ((str != NULL) && (strcmp(str, "NV12")) == 0) config_raw_color_space = IMAGE_TYPE_NV12;

    else if ((str != NULL) && (strcmp(str, "NV16")) == 0) config_raw_color_space = IMAGE_TYPE_NV16;
    else if ((str != NULL) && (strcmp(str, "YUV420P") == 0)) config_raw_color_space = IMAGE_TYPE_YUV420P;
    else printf("unsupported color space %s\n", str);


	printf("RAW video stream: %d x %d, color space %s\n",
	       config_raw_resolution_width,
	       config_raw_resolution_height,
	       str);

	// free memory
	json_object_put(cfg);

	return 0;
}
static void *GetMediaBuffer(void *arg) {

	rknn_context ctx;
	int ret;
	int model_len = 0;
	unsigned char *model;

	printf("Loading model ...\n");
	model = load_model(ssd_path, &model_len);
	ret = rknn_init(&ctx, model, model_len, 0);
	if (ret < 0) {
		printf("rknn_init fail! ret=%d\n", ret);
		return NULL;
	}

	// Get Model Input Output Info
	rknn_input_output_num io_num;
	ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
	if (ret != RKNN_SUCC) {
		printf("rknn_query fail! ret=%d\n", ret);
		return NULL;
	}
	printf("model input num: %d, output num: %d\n", io_num.n_input,
			io_num.n_output);

	printf("input tensors:\n");
	rknn_tensor_attr input_attrs[io_num.n_input];
	memset(input_attrs, 0, sizeof(input_attrs));
	for (unsigned int i = 0; i < io_num.n_input; i++) {
		input_attrs[i].index = i;
		ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]),
						sizeof(rknn_tensor_attr));
		if (ret != RKNN_SUCC) {
		printf("rknn_query fail! ret=%d\n", ret);
		return NULL;
		}
		printRKNNTensor(&(input_attrs[i]));
	}

	printf("output tensors:\n");
	rknn_tensor_attr output_attrs[io_num.n_output];
	memset(output_attrs, 0, sizeof(output_attrs));
	for (unsigned int i = 0; i < io_num.n_output; i++) {
		output_attrs[i].index = i;
		ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]),
						sizeof(rknn_tensor_attr));
		if (ret != RKNN_SUCC) {
		printf("rknn_query fail! ret=%d\n", ret);
		return NULL;
		}
		printRKNNTensor(&(output_attrs[i]));
	}

	MEDIA_BUFFER buffer = NULL;
	while (1) {
		buffer = RK_MPI_SYS_GetMediaBuffer(RK_ID_RGA, 12, -1);
		if (!buffer) {
			continue;
		}
		// printf("Get Frame:ptr:%p, fd:%d, size:%zu, mode:%d, channel:%d, "
		//        "timestamp:%lld\n",
		//        RK_MPI_MB_GetPtr(buffer), RK_MPI_MB_GetFD(buffer),
		//        RK_MPI_MB_GetSize(buffer),
		//        RK_MPI_MB_GetModeID(buffer), RK_MPI_MB_GetChannelID(buffer),
		//        RK_MPI_MB_GetTimestamp(buffer));

		int rga_buffer_model_input_size = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 3;

		// Set Input Data
		rknn_input inputs[1];
		memset(inputs, 0, sizeof(inputs));
		inputs[0].index = 0;
		inputs[0].type = RKNN_TENSOR_UINT8;
		inputs[0].size = rga_buffer_model_input_size;
		inputs[0].fmt = RKNN_TENSOR_NHWC;
		inputs[0].buf = RK_MPI_MB_GetPtr(buffer);

		ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
		if (ret < 0) {
			printf("rknn_input_set fail! ret=%d\n", ret);
			return NULL;
		}

		// Run
		// printf("rknn_run\n");
		ret = rknn_run(ctx, NULL);
		if (ret < 0) {
			printf("rknn_run fail! ret=%d\n", ret);
			return NULL;
		}

		// Get Output
		rknn_output outputs[2];
		memset(outputs, 0, sizeof(outputs));
		outputs[0].want_float = 1;
		outputs[1].want_float = 1;
		ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
		if (ret < 0) {
			printf("rknn_outputs_get fail! ret=%d\n", ret);
			return NULL;
		}

		// Post Process
		detect_result_group_t detect_result_group;
		postProcessSSD((float *)(outputs[0].buf), (float *)(outputs[1].buf), MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, &detect_result_group);
		// Release rknn_outputs
		rknn_outputs_release(ctx, 2, outputs);

		// Dump Objects
		for (int i = 0; i < detect_result_group.count; i++) {
			detect_result_t *det_result = &(detect_result_group.results[i]);
			printf("%s @ (%d %d %d %d) %f\n", det_result->name,
				det_result->box.left,
				det_result->box.top, det_result->box.right,
				det_result->box.bottom,
				det_result->prop);
		}
		RK_MPI_MB_ReleaseBuffer(buffer);
		
	}

	// release
	if (ctx) rknn_destroy(ctx);
	if (model) free(model);

	return NULL;
}

int main(){

	int r;
	const int video_chn = 0;           // camera id 0
	pthread_t read_thread;
  	pthread_create(&read_thread, NULL, GetMediaBuffer, NULL);
	// output_file = fopen("/media/usb0/1.rgb888", "w");

	wait_jmpp_init_complete();
	jmpp_api_init(); // before calling JMPP API. Need to initialize it
	config_json_get();  // read mpower NPU configuration

	
	// open raw stream(224x224)
	// raw stream is in NV16 color space. 
	// User has to call Rockchip native API to get frame data.

	r = jmpp_raw_stream_open(video_chn, &raw_stream); 


	// user can use vi_pipe and vi_chn to call rockchip API to get data.

	// It is possible to get VI data if vi_chn is known.
	// It is possible to bind VI data to one or mutiple RGA if vi_chn is known.
	// 
	// There are different ways to get frame data, either using RK_MPI_SYS_GetMediaBuffer()
	// or RK_MPI_SYS_RegisterOutCb(). But application may have to create thread when calling them.
	
	raw_stream_vi_pipe = jmpp_raw_stream_vi_pipe(video_chn, &raw_stream);
	raw_stream_vi_chn = jmpp_raw_stream_vi_chn(video_chn, &raw_stream);

	
	if (r != 0) printf("cannot open RAW stream\n");


	// RGA[12] color space conversation

	// coverting color space
	
	RGA_ATTR_S stRgaAttr;
	stRgaAttr.bEnBufPool = RK_TRUE;
	stRgaAttr.u16BufPoolCnt = 2;     // Please do not set too many. 2 is default
	stRgaAttr.u16Rotaion = 0;
	stRgaAttr.stImgIn.u32X = 0;
	stRgaAttr.stImgIn.u32Y = 0;
	stRgaAttr.stImgIn.imgType = IMAGE_TYPE_NV16;    // MUST BE NV16. 
	stRgaAttr.stImgIn.u32Width = config_raw_resolution_width;
	stRgaAttr.stImgIn.u32Height = config_raw_resolution_height;
	stRgaAttr.stImgIn.u32HorStride = config_raw_resolution_width;
	stRgaAttr.stImgIn.u32VirStride = config_raw_resolution_height;
	
	stRgaAttr.stImgOut.u32X = 0;
	stRgaAttr.stImgOut.u32Y = 0;
	stRgaAttr.stImgOut.imgType = IMAGE_TYPE_RGB888;
	stRgaAttr.stImgOut.u32Width = 300;
	stRgaAttr.stImgOut.u32Height = 300;
	stRgaAttr.stImgOut.u32HorStride = 300;
	stRgaAttr.stImgOut.u32VirStride = 300;

	// RGA channel 12 .. 15 can be used with rockchip native API
	
	r = RK_MPI_RGA_CreateChn(12, &stRgaAttr);
	if (r) printf("ERROR: Create rga[12] falied! ret=%d\n", r);

	// register a callback when there is a data on rga channel 12
	
	MPP_CHN_S stEncChn;

	stEncChn.enModId = RK_ID_RGA;
	stEncChn.s32ChnId = 12;

	// r = RK_MPI_SYS_RegisterOutCb(&stEncChn, rga_chn12_frame_cb);
	// if (r) printf("ERROR: Register cb for RGA CB error! code:%d\n", r);

	// starting the VI -> RGA[12], VI -> RGA[13] pipeline data.
	// VI automatically starts capturing data after binding is completed.

	MPP_CHN_S stSrcChn;
	stSrcChn.enModId = RK_ID_VI;
	stSrcChn.s32DevId = raw_stream_vi_pipe;
	stSrcChn.s32ChnId = raw_stream_vi_chn;
	MPP_CHN_S stDestChn;
	stDestChn.enModId = RK_ID_RGA;
	stDestChn.s32DevId = 0;
	stDestChn.s32ChnId = 12;
	
	r = RK_MPI_SYS_Bind(&stSrcChn, &stDestChn);
	if (r) printf("ERROR: Bind vi[%d:%d] and rga[12] failed! ret=%d\n", raw_stream_vi_pipe, raw_stream_vi_chn, r);
	
	pthread_join(read_thread, NULL);
	// stop and close should alwys go in pair and in the following order
	
	jmpp_raw_stream_stop(video_chn, &raw_stream);
	jmpp_raw_stream_close(video_chn, &raw_stream);
	
	jmpp_npu_stream_stop(video_chn, &stream);
	jmpp_npu_stream_close(video_chn, &stream);

	return 0;
	
}



		
