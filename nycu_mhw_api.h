#ifndef NYCU_MHW_API_H
#define NYCU_MHW_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include <sys/time.h>
#include <nanomsg/nn.h>
#include <nanomsg/pubsub.h>


#define NYCU_MHW_PUB_URL "ipc:///tmp/nycu.mhw.ipc"    // this is the broadcast URL


typedef struct _nycu_mhw_alarm_event_
{
	struct 
	{
		int width;
		int height;
		
	} frame_dimension;
	
	struct
	{
		int top_x;
		int top_y;
		int bottom_x;
		int bottom_y;
		
	} front_vehicle_coord;    // front vehicle coordinate


	unsigned long long int timestamp;
	
	float front_vehicle_distance;       // -1: invalid

	float car_speed;                    // km/h

} NYCU_MHW_ALARM_EVENT_s;
	

#ifdef __cplusplus
}
#endif


#endif
