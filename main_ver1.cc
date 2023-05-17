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
#include <poll.h>

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

#define NUM_NPU_FD 3


using namespace std;
using namespace cv;

/*-------------------------------------------
                  Functions
-------------------------------------------*/

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

int mpower_npu_enable;          // mpower process enable or disable
int mpower_2D_graphic_enable;   // mpower helps do some 2D drawing on the frame
int npu_resolution_width;
int npu_resolution_height;

IMAGE_TYPE_E  npu_color_space;
JMPP_NPU_STREAM_S stream;    // do not modify this data or read structure data directly.
                             // always use API to access information.
                             // this may take a large memory. declare as a global variable
JMPP_VIDEO_GRAPHIC_S vo_graphic;   // video output graphic configration(context).
                                   // this may take a large memory. Prefer as a global variable


// for N12 and YUV420P, maximum frame size is width * height * 3 / 2

uint8_t data[JMPP_MAX_VI_FRAME_SIZE];     // supporting NV12 or YUV420P video input
static int npu_config_json_get();        // getting mpower configuration
static void wait_jmpp_init_complete();

static void wait_jmpp_init_complete(){

	int jmpp_api_sock;
	int delay = 100;
	JMPP_API_MSG_REQREP_S jmpp_api_msg;
	int r;
	
	if ((jmpp_api_sock = nn_socket(AF_SP, NN_REQ)) < 0){
        printf("nn_socket\n");
    }

	// set a 100ms message timeout
	
    if (nn_setsockopt(jmpp_api_sock, NN_SOL_SOCKET, NN_RCVTIMEO, &delay, sizeof(int)) < 0){
        printf("nn_setsockopt\n");
    }
	
	if (nn_connect (jmpp_api_sock, JMPP_API_INTF_REQREP_URL) < 0){
		printf("nn_connect\n");
    }


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

int npu_config_json_get(){

	JsonNode *cfg;
	const char *str;

	mpower_npu_enable = 0;
	mpower_2D_graphic_enable = 0;
	npu_resolution_width = 0;
	npu_resolution_height = 0;
	npu_color_space = IMAGE_TYPE_UNKNOW;

	// get first video channel 0 NPU configuration
	
	cfg = user_config_query_to_json("{\"mild\":{\"videoInput\":{\"0\":{\"npuStream\":{}}}}}");
	
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
		
	if ((str != NULL) && (strcmp(str, "NV16")) == 0) npu_color_space = IMAGE_TYPE_NV16;
    else if ((str != NULL) && (strcmp(str, "YUV420P") == 0)) npu_color_space = IMAGE_TYPE_YUV420P;
	else printf("unsupported color space %s\n", str);


	printf("MPOWER NPU video stream: %d x %d, color space %s, 2D graphic %d\n",
	       npu_resolution_width,
	       npu_resolution_height, str,
	       mpower_2D_graphic_enable );
	
	// free memory
	json_object_put(cfg);

	return 0;
	
	
}   

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char** argv)
{
    const int img_width = 300;
    const int img_height = 300;
    const int img_channels = 3;

    rknn_context ctx;
    int ret;
    int model_len = 0;
    unsigned char *model;

    const char *model_path = argv[1];

    if (argc != 3) {
        printf("Usage:%s model image\n", argv[0]);
        return -1;
    }

    

    // Load RKNN Model
    printf("Loading model ...\n");
    model = load_model(model_path, &model_len);
    ret = rknn_init(&ctx, model, model_len, 0);
    if(ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    // Get Model Input Output Info
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        printRKNNTensor(&(input_attrs[i]));
    }

    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        printRKNNTensor(&(output_attrs[i]));
    }

    int r;
	struct pollfd pfd[NUM_NPU_FD];	

	int npu_fd;
	const int video_chn = 0;           // camera id 0

	uint32_t data_size = 0;
	uint8_t *data_addr = 0;
	uint64_t data_timestamp = 0;
	uint32_t      width = 0;
	uint32_t      height = 0;
	IMAGE_TYPE_E  color_space = IMAGE_TYPE_UNKNOW;
	int timer_fd;
	struct itimerspec timeout;
	int frame_count = 0;

    // !! need to check JMPP status before calling JMPP API
	wait_jmpp_init_complete();

    timer_fd = timerfd_create(CLOCK_MONOTONIC, 0);
	if (timer_fd < 0) printf("failed to create timer\n");
	
	jmpp_api_init(); // before calling JMPP API. Need to initialize it
	npu_config_json_get();  // read mpower NPU configuration

    // if you want to send frames to EVB LCD screen(1204x600),
	// video graphic needs to be used.
	// this is for debugging.
	// final product may not have LCD screen
	
	const int x_coord = 0;   // should be 0. not support yet
	const int y_coord = 0;   // should be 0. not support yet

	int display_height = 540;  
	int display_width = 960;   // prefer align to 16

	// Using original incoming video resolution and color space
	// to display on LCD. I
	// If you are are scale the original video down or convert to other
	// color space, please change the input image dimension
	// and color space accordingly. 

	int input_width = npu_resolution_width;
	int input_height = npu_resolution_height;
	int input_color_space = npu_color_space;

	assert(input_color_space == IMAGE_TYPE_NV16);
	
	/* in the DVR solution, npu stream color space is NV16, 640x360 */
	
	r = jmpp_video_graphic_open( 0,        // always 0 for now
				     x_coord,  // always 0 for now
				     y_coord,  // always 0 for now
				     input_width,
				     input_height,
				     IMAGE_TYPE_NV16,
				     display_width,
				     display_height,				   
				     &vo_graphic);
	if (r != 0) printf ("cannot open video output\n");
	
	r = jmpp_npu_stream_open(video_chn, &stream);
	if (r != 0) printf("NPU stream open error\n");
	
    // get the file desciptor for incoming video stream
	npu_fd = jmpp_npu_stream_fd(video_chn,  &stream);
		
	pfd[0].fd = npu_fd;
	pfd[0].events = POLLIN;
	pfd[0].revents = 0;


	pfd[1].fd = timer_fd;
	pfd[1].events = POLLIN;
	pfd[1].revents = 0;
	
	timeout.it_value.tv_sec     = 20;    // 5 seconds timer for printing information
	timeout.it_value.tv_nsec    = 0;
	timeout.it_interval.tv_sec  =  20;   
	timeout.it_interval.tv_nsec =  0;

	timerfd_settime(timer_fd, 0, &timeout, NULL);
	
	// start video streaming. this should be the last call before
	// going into while loop

	jmpp_npu_stream_start(0,  &stream);
    
    while(1){
        if (poll(pfd, NUM_NPU_FD, -1) < 0){
			printf("poll error\n");
			continue;
		}
        if (pfd[0].revents & POLLIN){      // check if any NPU data

			// get data, returning number of bytes
			data_size = jmpp_npu_stream_get(video_chn, &stream);

			// user space data address
			data_addr = jmpp_npu_stream_addr(video_chn, &stream);

			// timestamp in micro seconds
			data_timestamp = jmpp_npu_stream_pts(video_chn, &stream);

			color_space = jmpp_npu_stream_color_space(video_chn,  &stream);
			width  =  jmpp_npu_stream_width(video_chn, &stream);
			height =  jmpp_npu_stream_height(video_chn, &stream);

			// color space, width and height should match the NPU stream
			// configuration in mild.json

			// user should copy the data to its own buffer if time consuming
			// task is required.
		
			memcpy(data, data_addr, data_size);

			// after saving data, user should release stream as soon as possible.
			// do not do time consuming task without releasing  the stream
                      			
			jmpp_npu_stream_release(video_chn, &stream);


			// send frame to video graphic. this is like 10 fps

			
			if ((frame_count == 0)  || (frame_count == 3)  ||
			    (frame_count == 6)  || (frame_count == 9)  ||
			    (frame_count == 12) || (frame_count == 15) ||
				(frame_count == 18) || (frame_count == 21) ||
			    (frame_count == 24) || (frame_count == 27)){
                
                // for(int i=0; i<10; ++i) cout << frame_count << " " << i << endl;

				/* DVR solution image color space is NV16, 640x360 */

				// this is a time consuming API.
				// try not do this in the same thread as incoming data thread.
				// Do not send too many frames (per seconds) to video ouput
				
				r = jmpp_video_graphic_frame_send (0,   // always 0
								   data,
								   data_size,
								   &vo_graphic);

				if (r != 0) printf ("cannot send frame to video output\n");
			}

			++frame_count;

			if (frame_count >= 30) frame_count = 0;					
		}



		if (pfd[1].revents & POLLIN){   // print information every 5 seconds
			u_int64_t ticks;
			
			if (read(timer_fd, &ticks, sizeof(ticks)) != sizeof(ticks)) printf("timer error\n");
			
			printf("NPU data[%llu], %d x %d, color space %d, size %d, addr 0x%x\n",
			       data_timestamp, width, height, color_space, data_size,
			       (uint32_t) data_addr);
            
            cv::Size yuyv_size(width, height);

            uint8_t yuyv_buf[width * height]; 
            uint32_t idx = 0;

            for (int i=0; i<width * height*2; i++){
                // if(tmp==460800) break;
                // cout << tmp << " " << unsigned(*(&data + tmp)) << endl;
                // cout << i << " " << input_width*input_height*2 <<endl;;
                yuyv_buf[i + 0] = unsigned(*(&data + idx++));
                yuyv_buf[i + 1] = unsigned(*(&data + input_width*input_height+idx));
                yuyv_buf[i + 2] = unsigned(*(&data + idx++));
                yuyv_buf[i + 3] = unsigned(*(&data + input_width*input_height+idx));
            }
            cv::Mat yuyv(yuyv_size, CV_8UC2, yuyv_buf);

            cv::Mat dst;
            cv::cvtColor(yuyv, dst, cv::COLOR_YUV2BGRA_YUYV); 

            imwrite("./test.bmp", dst);
            cout << "HELLO" << endl;
		}



        // // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        // // Load image
        // // cv::Mat orig_img = cv::imread("./model/video_split/" + img_path, 1);
        // // cv::Mat img = orig_img.clone();
        // if(!orig_img.data) {
        //     printf("cv::imread %s fail!\n", img_path);
        //     return -1;
        // }
        // if(orig_img.cols != img_width || orig_img.rows != img_height) {
        //     printf("resize %d %d to %d %d\n", orig_img.cols, orig_img.rows, img_width, img_height);
        //     cv::resize(orig_img, img, cv::Size(img_width, img_height), (0, 0), (0, 0), cv::INTER_LINEAR);
        // }

        // // Set Input Data
        // rknn_input inputs[1];
        // memset(inputs, 0, sizeof(inputs));
        // inputs[0].index = 0;
        // inputs[0].type = RKNN_TENSOR_UINT8;
        // inputs[0].size = img.cols*img.rows*img.channels();
        // inputs[0].fmt = RKNN_TENSOR_NHWC;
        // inputs[0].buf = img.data;

        // ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
        // if(ret < 0) {
        //     printf("rknn_input_set fail! ret=%d\n", ret);
        //     return -1;
        // }

        // // Run
        // printf("rknn_run\n");
        // ret = rknn_run(ctx, nullptr);
        // if(ret < 0) {
        //     printf("rknn_run fail! ret=%d\n", ret);
        //     return -1;
        // }

        // // Get Output
        // rknn_output outputs[2];
        // memset(outputs, 0, sizeof(outputs));
        // outputs[0].want_float = 1;
        // outputs[1].want_float = 1;
        // ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
        // if(ret < 0) {
        //     printf("rknn_outputs_get fail! ret=%d\n", ret);
        //     return -1;
        // }

        // // Post Process
        // detect_result_group_t detect_result_group;
        // postProcessSSD((float *)(outputs[0].buf), (float *)(outputs[1].buf), orig_img.cols, orig_img.rows, &detect_result_group);
        // // Release rknn_outputs
        // rknn_outputs_release(ctx, 2, outputs);

        // // Draw Objects
        // for (int i = 0; i < detect_result_group.count; i++) {
        //     detect_result_t *det_result = &(detect_result_group.results[i]);
        //     bool truck_flag = false;
        //     bool car_flag = false;

        //     if (string(det_result->name).compare("truck") == 0) truck_flag = true;
        //     if (string(det_result->name).compare("car") == 0) car_flag = true;

        //     if (truck_flag || car_flag){
                
        //         int x1 = det_result->box.left;
        //         int y1 = det_result->box.top;
        //         int x2 = det_result->box.right;
        //         int y2 = det_result->box.bottom;
        //         if (!(x2 <= orig_img.cols/3) && !(x1 >= orig_img.cols*2/3)){
        //             printf("%s @ (%d %d %d %d) %f\n",
        //                 det_result->name,
        //                 det_result->box.left, det_result->box.top, det_result->box.right, det_result->box.bottom,
        //                 det_result->prop);
        //             float focal_length = 1600;
        //             float car_width = 1.75;
        //             float width = x2 - x1;
        //             float distance = int((focal_length * car_width / width)*100+0.5)/100.;
        //             Scalar box_color;
        //             if (width <= orig_img.cols/2){
        //                 if (distance >= 10.) box_color = CV_RGB(50,205,50);
        //                 else if (distance < 10. && distance > 7.) box_color = CV_RGB(255,140,0);
        //                 else box_color = CV_RGB(255,0,0);

        //                 rectangle(orig_img, Point(x1, y1), Point(x2, y2), box_color, 3);
                        
        //                 int fontface = FONT_HERSHEY_SIMPLEX;
        //                 double scale = 0.6;
        //                 int thickness = 1;
        //                 int baseline = 0;
        //                 string label = to_string(distance);
        //                 Point or_point = Point(x1, y2)+ Point(0, baseline);

        //                 label = label.assign(label, 0, label.find("."));
        //                 Size text = getTextSize(label, fontface, scale, thickness, &baseline);
        //                 rectangle(orig_img, or_point, or_point + Point(text.width, -text.height), CV_RGB(255,255,255), CV_FILLED);
        //                 putText(orig_img, label, or_point, fontface, scale, CV_RGB(0,0,0), thickness, 8);
        //             }
        //         }
        //     }
        // }
    }


	// stop and close should alwys go in pair and in the following order
	jmpp_npu_stream_stop(video_chn, &stream);
	jmpp_npu_stream_close(video_chn, &stream);
	jmpp_video_graphic_close(0, &vo_graphic);

    // Release
    if(ctx >= 0) rknn_destroy(ctx);
    if(model) free(model);
    return 0;
}
