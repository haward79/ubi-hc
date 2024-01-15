#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <assert.h>
#include <pthread.h>
#include <sys/timerfd.h>
#include <sys/statfs.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/file.h>
#include <dirent.h>
#include "fs_util.h"
#include <fcntl.h>
#include <unistd.h>
#include <poll.h>
#include <errno.h>
#include <libgen.h>
#include <time.h>
#include <stdlib.h>
#include "gps_db.h"
#include "gps.h"
#include "upload.h"
#include "fs_util.h"
#include "jtime.h"
#include "curl/curl.h"
#include "nanomsg/nn.h"
#include "nanomsg/pipeline.h"
#include "nanomsg/pubsub.h"
#include <nanomsg/reqrep.h>
#include <nanomsg/inproc.h>
#include "json.h"
#include "config_json.h"
#include "upload.h"
#include "event.h"


extern int gpsNgsensor_log_enable;

#define WAITMS(x)  do { struct timeval wait = { 0, (x) * 1000 }; (void)select(0, NULL, NULL, NULL, &wait);} while (0)


typedef enum
{
	HTTP_SEND_STATUS_OK        = 0,
	
	HTTP_SEND_STATUS_TIMEOUT,
	
	HTTP_SEND_STATUS_EXTRA_FDS,
	
	HTTP_SEND_STATUS_CURL_API_ERROR,

	HTTP_SEND_STATUS_SERVER_ERROR,
	
	
} HTTP_SEND_STATUS_E;


typedef enum
{
	GPS_DB_LOAD_INIT  = 0,   
	GPS_DB_LOAD_OK    = 1,
	GPS_DB_LOAD_GT    = 2,
	GPS_DB_LOAD_AGAIN = 3,
	
} GPS_DB_STATUS_E;

const char *GPS_DB_STATUS_STR[4] = {"INIT", "OK", "GT", "AGAIN"};

volatile int nr_http_error;

GPS_DB_S gps_db;

static GPS_DATA_s *gps_data_pop(GPS_BUF_s *buf, GPS_DATA_s *data);
static void       gps_data_push(GPS_BUF_s *buf, const GPS_DATA_s *data);
static int        gps_rec_log_read(GPS_REC_TIME_s *gps_rec);
static int        gps_rec_log_write(const GPS_REC_TIME_s *gps_rec);
static time_t     gps_to_calender_time(const GPS_REC_TIME_s *gps_time) __attribute__((unused));
static void       calender_to_gps_time(const time_t *t, GPS_REC_TIME_s *gps_time) __attribute__((unused));
static void       gps_to_jtime(const GPS_REC_TIME_s *gps_time, JTIME_S *jtm);
static void       jtime_to_gps_time(const JTIME_S *jtm, GPS_REC_TIME_s *gps_time);
static void       mask_sig(void);

static int        gps_db_load_tm_cmp(const JTIME_S *jtm, const char *file, int cmp_flag, int *pos);

static GPS_DB_STATUS_E gps_db_load(void);

static HTTP_SEND_STATUS_E http_url_send(const char *url,
					struct curl_waitfd extra_fds[],
					unsigned int extra_nfds);

static size_t curl_callback(void *contents, size_t size, size_t nmemb, void *userp);


static inline int gps_buf_is_full(const GPS_BUF_s *buf)
{

	return  (buf -> nr_data >= buf -> max_nr_data);

}


static inline int gps_buf_is_empty(const GPS_BUF_s *buf)
{
	return  (buf -> nr_data == 0);
}





static GPS_DATA_s *gps_data_pop(GPS_BUF_s *buf, GPS_DATA_s *data)
{
    int next;

    if (buf -> nr_data > 0)
    {

	    next = buf->tail + 1;  // next is where tail will point to after this read.
	    
	    if (next >= buf->max_nr_data)
	    {
		    next = 0;
	    }

	    *data = buf->data[buf->tail];  // read data and then move
	    
	    buf -> tail = next;              // tail to next offset.
	    
	    --(buf -> nr_data);
	    
	    return data;
    }

    return NULL;

    
}


static void gps_data_push(GPS_BUF_s *buf, const GPS_DATA_s *data)
{
    int next;

    if (gps_buf_is_full(buf))
    {
	    _eprintf ("buf full\n");
	    
	    return;
	    
    }
    
  
    next = buf -> head + 1;  // next is where head will point to after this write.
    
    if (next >= buf->max_nr_data )
    {
	    next = 0;
    }

    buf -> data[buf->head] = *data;  // load data and then move
    buf -> head = next;              // head to next data offset.

    ++(buf -> nr_data);
   
}



static int gps_rec_log_read(GPS_REC_TIME_s *gps_rec)
{

	FILE *fp;

	int i;

	const char *file = upload_gps_rec_fullpath;
	
	if (is_file(file) == 0)
	{
		_eprintf("%s not exist\n", file);
		return -1;
	}


	fp = fopen(file, "r");

	if (fp == NULL)
	{
		_eprintf ("cannot open %s\n", file);
		return -2;
	}

	

	if (flock(fileno(fp), LOCK_SH) == -1)
	{
		_eprintf ("cannot lock %s\n", file);

		fclose(fp);
		
		return -3;
	}



	i = fscanf(fp, "%u %u",
		   &(gps_rec -> date),
		   &(gps_rec -> tm));

	if (i < 2)
	{
		_eprintf ("fscanf %s error %d\n", file,i);
		fclose(fp);
		return -4;
	}

	if (flock(fileno(fp), LOCK_UN) == -1)
	{
		_eprintf ("cannot unlock %s\n", file);
		
		fclose(fp);

		return -5;
	}


	fclose(fp);

	_dprintf("read time %06d, date %08d\n", gps_rec -> tm, gps_rec -> date);

	return 0;
	
}



static int gps_rec_log_write(const GPS_REC_TIME_s *gps_rec)
{

	FILE *fp;
	int i;
	
	char buf[32];

	const char *file = upload_gps_rec_fullpath;

	_eprintf("gps_rec write time %06d, date %08d\n", gps_rec -> tm, gps_rec -> date);
	
	if (is_file(file) == 0)
	{
		_eprintf( "%s not exist\n", file);
		return -1;
	}


	fp = fopen(file, "w");

	if (fp == NULL)
	{
		_eprintf ("cannot open %s\n", file);
		return -2;
	}

	

	if (flock(fileno(fp), LOCK_EX) == -1)
	{
		_eprintf ("cannot lock %s\n", file);

		fclose(fp);
		
		return -3;
	}

	i = sprintf(buf, "%08d %06d\n",
		    gps_rec -> date,
		    gps_rec -> tm);


	if (fwrite(buf, 1, (size_t)i, fp) != (size_t) i)
	{
		_eprintf("fwrite error\n");
	}
	
	if (flock(fileno(fp), LOCK_UN) == -1)
	{
		_eprintf ("cannot unlock %s\n", file);
		
		fclose(fp);

		return -5;
	}

	fsync(fileno(fp));

	fclose(fp);

	return 0;
	
}



static time_t gps_to_calender_time(const GPS_REC_TIME_s *gps_time)
{
	struct tm m;

	int h;
	int min;
	int s;
	
	int y;
	int mon;
	int d;

	h = gps_time -> tm / 10000;
	min = (gps_time -> tm - (h * 10000)) / 100;
	s = (gps_time -> tm - (h * 10000) - (min * 100));

	
	y = gps_time -> date / 10000;
	mon = (gps_time -> date - (y * 10000)) / 100;
	d = (gps_time -> date - (y * 10000) - (mon * 100));


	//printf ("h %d min %d s %d, d %d, mon %d, y %d\n",
	//         h, min, s, d, mon,y);
	
	
	m.tm_isdst = 0;
	m.tm_yday  = 0;
	m.tm_wday  = 0;

	m.tm_year = y - 1900;
	m.tm_mon  = mon - 1;
	m.tm_mday = d;

	m.tm_hour = h;
	m.tm_min  = min;
	m.tm_sec  = s;
	
	return mktime(&m);

}


static void calender_to_gps_time(const time_t *t, GPS_REC_TIME_s *gps_time)
{

	struct tm tm;

	
	// t: number of seconds elapsed since the Epoch, 1970-01-01 00:00:00 +0000 (UTC).

	localtime_r(t, &tm);

	
	// printf ("tm %d %d %d, %d %d %d\n",
	//         tm.tm_year,tm.tm_mon,tm.tm_mday,tm.tm_hour,tm.tm_min,tm.tm_sec);

	
	gps_time -> date = ((tm.tm_mday) +
			    ((tm.tm_mon  + 1) * 100) +
			    (tm.tm_year  + 1900) * 10000);
		
	gps_time -> tm = ((tm.tm_hour * 10000) +
			  (tm.tm_min  * 100)   +
			  tm.tm_sec);

}



static void gps_to_jtime(const GPS_REC_TIME_s *gps_time, JTIME_S *jtm)
{


	jtm -> hour = gps_time -> tm / 10000;
	jtm -> min = (gps_time -> tm - (jtm -> hour * 10000)) / 100;
	jtm -> sec = (gps_time -> tm - (jtm -> hour * 10000) - (jtm -> min * 100));

	
	jtm -> year = gps_time -> date / 10000;
	jtm -> month = (gps_time -> date - (jtm -> year * 10000)) / 100;
	jtm -> day = (gps_time -> date - (jtm -> year * 10000) - (jtm -> month * 100));
	
}



static void jtime_to_gps_time(const JTIME_S *jtm, GPS_REC_TIME_s *gps_time)
{

	gps_time -> tm   = (jtm -> hour * 10000) + (jtm -> min * 100) + jtm -> sec;
	gps_time -> date = (jtm -> year * 10000) + (jtm -> month * 100) + jtm -> day;	
	
}
 


static void mask_sig(void)
{
	sigset_t mask;
	sigemptyset(&mask); 
        sigaddset(&mask, SIGINT);
	sigaddset(&mask, SIGTERM);
	sigaddset(&mask, SIGQUIT);
        pthread_sigmask(SIG_BLOCK, &mask, NULL);
}






static int gps_db_load_tm_cmp(const JTIME_S *jtm, const char *file, int cmp_flag,
			      int *pos)
{

	GPS_KEY_TIMESTAMP_s gps_key;

	GPS_REC_TIME_s gps_time;
	
	GPS_DB_DATA_s *pData;

	int p;
	int n;

	_dprintf("%s %d %d/%d/%d/%d\n",
		 file, cmp_flag,
		 jtm -> year, jtm -> month, jtm -> day, jtm -> hour);
	
	if (file_size(file) > 0)
	{
		
	       n = gps_db_init(file, gps_db.data, MAX_NUM_GPS_DB_DATA);

	       if (n > 0)
	       {
		       jtime_to_gps_time(jtm, &gps_time);
		       
		       gps_key.tm       = gps_time.tm;
		       gps_key.date     = gps_time.date;
		       gps_key.cmp_flag = cmp_flag; 
		       
		       
		       pData = gps_db_bsearch(&gps_key,
					      gps_db.data,
					      (size_t) n,
					      sizeof(GPS_DB_DATA_s),
					      gps_db_time_cmp, &p);
		       
		       if (pData)
		       {
			       _dprintf ("%s, pos %d(%d), tm %d, date %d\n",
					 file,
					 p, n,
					 pData -> sys_tm, pData -> sys_date);
			       
			       strcpy(gps_db.file, file);
			       
			       *pos = p;
			       
			       return n;
			       
		       } else
		       {
			       return 0;
		       }
		       
	       }
	}

	return 0;

}


static GPS_DB_STATUS_E gps_db_load()     // should be called by data acq thread only.
{
        
	JTIME_S jtm;
	time_t t0;
	time_t t1;

	int pos;
	int nr_data;

	const char *pFile;

	char file[FS_FULLPATH_MAX_LEN];
	
	gps_db.pFile = NULL;
	gps_db.num_data = 0;
	gps_db.db_pos = -1;


	t1 = gps_to_calender_time(&(gps_db.rtm_upload));
	t0 = gps_to_calender_time(&(gps_db.rtm_init));

	if (t1 != t0)
	{
		_eprintf ("GPS possible rtm bug, %d %d, %d %d\n",
			  gps_db.rtm_upload.date,
			  gps_db.rtm_upload.tm,
			  gps_db.rtm_init.date,
			  gps_db.rtm_init.tm);

		if (t1 > t0)
		{
			gps_to_jtime(&(gps_db.rtm_upload), &jtm);
			
		} else
		{
			gps_to_jtime(&(gps_db.rtm_init), &jtm);
		}
	} else
	{
		gps_to_jtime(&(gps_db.rtm_init), &jtm);
	}

	pFile = find_file_eq(upload_gps_dir_fullpath,
			     jtm.year, jtm.month, jtm.day, jtm.hour,
			     file);

	_dprintf ("%d/%d/%d/%d\n", jtm.year, jtm.month, jtm.day, jtm.hour);
	
	if (pFile)
	{
		
		
		
		if (file_size(pFile) > 0)
		{

			nr_data = gps_db_load_tm_cmp(&jtm, file, 0, &pos);

			if (nr_data > 0)
			{
				if ((pos + 1) < nr_data)
				{
					gps_db.db_pos = pos + 1;
					gps_db.num_data = nr_data;
					strcpy(gps_db.file, file);
					gps_db.pFile = gps_db.file;
					
					return GPS_DB_LOAD_OK;
				} 
			}
		}
	}


	
	pFile = find_file_eq(cache_gps_dir_fullpath,
			     jtm.year, jtm.month, jtm.day, jtm.hour,
			     file);
	
	if (pFile)
	{
		if (file_size(pFile) > 0)
		{
						
			nr_data = gps_db_load_tm_cmp(&jtm, file, 0, &pos);
						
			if (nr_data > 0)
			{
				if ((pos + 1) < nr_data)
				{
					gps_db.db_pos = pos + 1;

					gps_db.num_data = nr_data;
					strcpy(gps_db.file, file);
					gps_db.pFile = gps_db.file;
					
					return GPS_DB_LOAD_OK;
				}
			}
			
			nr_data = gps_db_load_tm_cmp(&jtm, file, 1, &pos);

			if (nr_data > 0)
			{
				if (pos < nr_data)
				{
					gps_db.db_pos = pos;

					gps_db.num_data = nr_data;
					strcpy(gps_db.file, file);
					gps_db.pFile = gps_db.file;
					
					return GPS_DB_LOAD_OK;
				}
			}
			
		} 
	}
	
	return GPS_DB_LOAD_GT;
	
}




static GPS_DB_STATUS_E gps_db_load_gt()     // should be called by data acq thread only.
{                                            


	JTIME_S jtm;
	time_t t0;
	time_t t1;

	int pos;
	int nr_data;

	const char *pFile;

	char file[FS_FULLPATH_MAX_LEN];
	
	gps_db.pFile = NULL;
	gps_db.num_data = 0;
	gps_db.db_pos = -1;


	t1 = gps_to_calender_time(&(gps_db.rtm_upload));
	t0 = gps_to_calender_time(&(gps_db.rtm_init));

	if (t1 != t0)
	{
		_eprintf ("GPS possible rtm bug, %d %d, %d %d\n",
			  gps_db.rtm_upload.date,
			  gps_db.rtm_upload.tm,
			  gps_db.rtm_init.date,
			  gps_db.rtm_init.tm);

		if (t1 > t0)
		{
			gps_to_jtime(&(gps_db.rtm_upload), &jtm);
			
		} else
		{
			gps_to_jtime(&(gps_db.rtm_init), &jtm);
		}
	} else
	{
		gps_to_jtime(&(gps_db.rtm_init), &jtm);
	}
	
	pFile = find_file_gt(upload_gps_dir_fullpath,
			     jtm.year, jtm.month, jtm.day, jtm.hour,
			     file);
	
	_dprintf ("%d/%d/%d/%d\n", jtm.year, jtm.month, jtm.day, jtm.hour);

	
	if (pFile)
	{
		
		if (file_size(pFile) > 0)
		{

			nr_data = gps_db_load_tm_cmp(&jtm, file, 1, &pos);

			if (nr_data > 0)
			{
				if (pos < nr_data)
				{
					gps_db.db_pos = pos;
					gps_db.num_data = nr_data;
					strcpy(gps_db.file, file);
					gps_db.pFile = gps_db.file;

					gps_db.rtm_init.tm = gps_db.data[pos].sys_tm;
					gps_db.rtm_init.date = gps_db.data[pos].sys_date;
					
					return GPS_DB_LOAD_OK;
				}
			}

		} 	       
	}


	pFile = find_file_gt(cache_gps_dir_fullpath,
			     jtm.year, jtm.month, jtm.day, jtm.hour,
			     file);

	if (pFile)
	{
		if (file_size(pFile) > 0)
		{
						
			nr_data = gps_db_load_tm_cmp(&jtm, file, 1, &pos);
						
			if (nr_data > 0)
			{
				if (pos < nr_data)
				{
					gps_db.db_pos = pos;

					gps_db.num_data = nr_data;
					strcpy(gps_db.file, file);
					gps_db.pFile = gps_db.file;

					gps_db.rtm_init.tm = gps_db.data[pos].sys_tm;
					gps_db.rtm_init.date = gps_db.data[pos].sys_date;
					
					return GPS_DB_LOAD_OK;
				}
			}
		}
	}

	return GPS_DB_LOAD_INIT;
	
       
}



static void db_data_to_gpsbuf(GPS_BUF_s *gbuf)
{

	uint64_t n = 0;
	int r;
	GPS_DATA_s data;
	GPS_DB_DATA_s *pData;

	// _dprintf ("enter %s, %d, %d\n", __FUNCTION__, gps_buf_is_full(gbuf), gps_db.db_pos);
	
	pthread_mutex_lock(&(gbuf -> mutex));

	while ((0 == gps_buf_is_full(gbuf)) && (gps_db.db_pos < gps_db.num_data))
	{
		pData = &(gps_db.data[gps_db.db_pos]);
		
		data.sys_tm   = pData -> sys_tm;
		data.sys_date = pData -> sys_date;
		
		data.gps_tm   = pData -> gps_tm;
		data.gps_date = pData -> gps_date;


		strncpy(data.latitude, pData -> latitude, 16);
		strncpy(data.longitude, pData -> longitude, 16);
		strncpy(data.speed, pData -> speed, 12);
		strncpy(data.altitude, pData -> altitude,12);
		strncpy(data.hdop, pData -> hdop, 12);

		data.satellites_tracked = pData -> satellites_tracked;

		if (gpsNgsensor_log_enable == 1)
		{

			data.gX = pData -> gX;
			data.gY = pData -> gY;
			data.gZ = pData -> gZ;
		}

		data.course = pData -> course;
		data.car_speed = pData -> car_speed;
		data.rpm = pData -> rpm;
		data.acc = pData -> acc;
		data.upload_flag = pData->upload_flag;

		gps_data_push(gbuf, &data);
		
		// _dprintf ("push %d %d\n", pData -> sys_tm, pData -> sys_date);
		//printf ("push %08u %06u, %08u %06u\n", pData -> sys_date, pData -> sys_tm, pData -> gps_date, pData -> gps_tm);

		++(gps_db.db_pos);
		++n;
	}

	pthread_mutex_unlock(&(gbuf -> mutex));
	
	if (n > 0)
	{
		r = write(gbuf -> can_consume_eventfd, &n, sizeof(uint64_t));
		
		if (r != sizeof(uint64_t))
		{
			_eprintf("write eventfd error\n");
		}

	}
	
}


void *data_acq_gps_thread(void *arg)    // gps data acquisition from files (jlogger)
{


#define ACQ_NUM_FDS 3

	
	
	GPS_BUF_s *gps_data_buf = (GPS_BUF_s *)arg;

	int nm_sub_sock;
	int nm_sub_fd;

	int gps_timer_fd;
	int rec_timer_fd;

	struct itimerspec timeout;
	
	struct pollfd pfds[ACQ_NUM_FDS];

	int r;
	
	size_t sz;

	u_int64_t ticks;
	
	GPS_REC_TIME_s rtm;
	JTIME_S jtm;

	GPS_DB_STATUS_E old_db_status = GPS_DB_LOAD_INIT;
	GPS_DB_STATUS_E db_status     = GPS_DB_LOAD_INIT;

	char file[FS_FULLPATH_MAX_LEN];
	char file1[FS_FULLPATH_MAX_LEN];

	const char *fp;
	const char *fp1;

	_eprintf("GPS uploading starts(data acq)...\n");
	
	pthread_mutex_init(&(gps_db.mutex), NULL);
	
	nm_sub_sock   = -1;
	rec_timer_fd  = 0;
	gps_timer_fd  = 0;

	mask_sig();

	

	_eprintf("GPS uploading to %s\n", domain_name_str);
       
	if ((nm_sub_sock = nn_socket(AF_SP, NN_SUB)) < 0)
	{
		_eprintf("nn_sock error\n");
		goto gps_acq_exit;
        }

	// subscribe to everything ("" means all topics)

	if (nn_setsockopt(nm_sub_sock, NN_SUB, NN_SUB_SUBSCRIBE, "", 0) < 0)
	{
		_eprintf("nn_setsockopt error\n");
		goto gps_acq_exit;
        }
	
	if (nn_connect(nm_sub_sock, PUBSUB_URL) < 0)
	{
                _eprintf("nn_bind error\n");
		goto gps_acq_exit;
	}


	sz = sizeof(nm_sub_fd);
	
	r = nn_getsockopt (nm_sub_sock, NN_SOL_SOCKET,
			   NN_RCVFD, &nm_sub_fd, &sz);

	if (r < 0)
	{
		_eprintf("nn_getsockopt error\n");
		goto gps_acq_exit;
	}


	/* create gps data timer */
	
	gps_timer_fd = timerfd_create(CLOCK_MONOTONIC, 0);
	
	if (gps_timer_fd <= 0)
	{
		_eprintf("failed to create timer\n");
		
	}

	rec_timer_fd = timerfd_create(CLOCK_MONOTONIC, 0);
	
	if (rec_timer_fd <= 0)
	{
		_eprintf("failed to create timer\n");
		
	}

	memset(pfds, 0, sizeof(pfds));
	
	pfds[0].fd = nm_sub_fd;         
	pfds[0].events = POLLIN;

	pfds[1].fd = gps_timer_fd;    // gps db data loading or send to uploading thread
	pfds[1].events = POLLIN;

	pfds[2].fd = rec_timer_fd;    // update gps.rec file
	pfds[2].events = POLLIN;

	gps_db.rtm_upload.tm   = 0;
	gps_db.rtm_upload.date = 0;

	gps_db.rtm_init.tm   = 0;
	gps_db.rtm_init.date = 0;

	gps_db.num_data   = 0;
	gps_db.db_pos = 0;		
	gps_db.pFile = NULL;
	gps_db.file[0] = '\0';

	if (!is_file(upload_gps_rec_fullpath))
	{

		_eprintf("new %s\n", upload_gps_rec_fullpath);
		
		new_file(upload_gps_rec_fullpath);
		
		// find the oldest file
		
		// gps.rec data format:
		// 
		// <year><month><day> <hour><min><sec> 
		// 20200302 040009 
		// hour:minute:second = 04:00:09
		// year:month:day = 2020:03:02
		//
		// <year><month><day> <hour><min><sec> is at the beginning of very line
		// in the gps file.
		
		gps_db.rtm_init.tm   = 0;
		gps_db.rtm_init.date = 0;
		
		r = gps_rec_log_write(&(gps_db.rtm_init));
		
	}
			
	r = gps_rec_log_read(&(gps_db.rtm_init));
			
	if (r != 0)
	{
		_eprintf("cannot read gps rec\n");
	}

	
	if (gps_db.rtm_init.date == 0)
	{
		gps_db.rtm_init.tm = 0;
	}

	gps_db.rtm_upload = gps_db.rtm_init;

	_eprintf("GPS init timestamp: %d %d\n",
		 gps_db.rtm_upload.date,
		 gps_db.rtm_upload.tm);
	
	/* set gps data acq timer */
	
	timeout.it_value.tv_sec     =  5;
	timeout.it_value.tv_nsec    =  0;
	timeout.it_interval.tv_sec  =  5;
	timeout.it_interval.tv_nsec =  0;
	
	r = timerfd_settime(gps_timer_fd, 0, &timeout, NULL);

	/* set gps.rec write timer */

	timeout.it_value.tv_sec     =  10 * 60; 
	timeout.it_value.tv_nsec    =  0;
	timeout.it_interval.tv_sec  =  10 * 60;    
	timeout.it_interval.tv_nsec =  0;
	
	r = timerfd_settime(rec_timer_fd, 0, &timeout, NULL);
	
	while(1)
	{

		r = poll(pfds, ACQ_NUM_FDS, -1);   // no timeout

		if (r < 0)
		{
			_eprintf("poll error\n");
			
			continue;
		}


		if (pfds[0].revents & POLLIN)
		{

			PUB_MSG_S msg;

			if ((r = nn_recv(nm_sub_sock, &msg, sizeof(msg), 0)) < 0)
			{
				_eprintf("nn_recv error\n");
			}

			if (r == sizeof(PUB_MSG_S))
			{
				if ((msg.type == THREAD_EXIT) ||
				    (msg.type == THREAD_GPS_EXIT))
				{
					goto gps_acq_exit;
				}
			}
			
			
		}
		
		
		if (pfds[1].revents & POLLIN)   // gps data timer
		{
				
			if (read(gps_timer_fd, &ticks, sizeof(ticks)) != sizeof(ticks))
			{
				_eprintf("timer fd read error\n");
				continue;
			}
			
			pthread_mutex_lock(&(gps_db.mutex));

			_lprintf (0, "db status %s, %d(%d)\n", GPS_DB_STATUS_STR[db_status],
				  gps_db.db_pos,
				  gps_db.num_data);

			if (gps_db.num_data > 0)
			{
				r = gps_db.num_data - 1;

				// check if last dat is uploaded to server

				_dprintf ("### %d %d %d %d %d %d\n",
					  gps_db.db_pos,
					  gps_db.num_data,
					  gps_db.rtm_upload.tm,
					  gps_db.rtm_upload.date,
					  gps_db.data[r].sys_tm,
					  gps_db.data[r].sys_date);


				if ((gps_db.rtm_upload.tm   == gps_db.data[r].sys_tm) &&
				    (gps_db.rtm_upload.date == gps_db.data[r].sys_date))
				{

					
					_lprintf (0, "reload db, %d %d\n",
						  gps_db.rtm_upload.tm,
						  gps_db.rtm_upload.date);

					assert(gps_buf_is_empty(gps_data_buf));

					gps_db.pFile = NULL;
					gps_db.num_data = 0;
					gps_db.db_pos = -1;
					gps_db.rtm_init = gps_db.rtm_upload; 

					// WARNING!! do not reset rtm_upload

					db_status = GPS_DB_LOAD_INIT;
					
				}
			}
			
			pthread_mutex_unlock(&(gps_db.mutex));
			
			if (db_status == GPS_DB_LOAD_INIT)
			{
				db_status = gps_db_load();

				_lprintf(0, "db status(INIT) %s\n", GPS_DB_STATUS_STR[db_status]);

				if (db_status == GPS_DB_LOAD_OK)
				{
					_lprintf (0, "load db %s, pos %d(%d), tm %d, date %d\n",
						  gps_db.file,
						  gps_db.db_pos,
						  gps_db.num_data,
						  gps_db.data[gps_db.db_pos].sys_tm,
						  gps_db.data[gps_db.db_pos].sys_date);
				}
				
				
				
			} else if (db_status == GPS_DB_LOAD_GT)
			{

				db_status = gps_db_load();  // check again

				_lprintf(0, "db status(GT) %s\n", GPS_DB_STATUS_STR[db_status]);

				if (db_status == GPS_DB_LOAD_GT)
				{
				
					gps_to_jtime(&(gps_db.rtm_init), &jtm);

					fp = find_file_gt (upload_gps_dir_fullpath,
						   jtm.year, jtm.month, jtm.day, jtm.hour,
						   file);

					fp1 = find_file_gt (cache_gps_dir_fullpath,
							    jtm.year, jtm.month, jtm.day, jtm.hour,
							    file1);

					if (fp != NULL)
					{
						_lprintf(0, "%s, %d/%d/%d, y %d, %d\n", upload_gps_dir_fullpath,
							 jtm.month, jtm.day, jtm.hour, jtm.year,
							 file_size(fp));
					}

					if (fp1 != NULL)
					{
						_lprintf(0, "%s, %d/%d/%d, y %d, %d\n", cache_gps_dir_fullpath,
							 jtm.month, jtm.day, jtm.hour, jtm.year,
							 file_size(fp1));
					}
				
				
					if (((fp != NULL) && (file_size(fp) > 0)) ||
					    ((fp1 != NULL) && (file_size(fp1) > 0)))
					{
						db_status = gps_db_load_gt();

						_lprintf(0, "db status(load gt) %d\n",  db_status);

						if (db_status == GPS_DB_LOAD_OK)
						{
							_lprintf (0, "load db %s, pos %d(%d), tm %d, date %d\n",
								  gps_db.file,
								  gps_db.db_pos,
								  gps_db.num_data,
								  gps_db.data[gps_db.db_pos].sys_tm,
								  gps_db.data[gps_db.db_pos].sys_date);
						}
					}
				}
			}


			if (db_status == GPS_DB_LOAD_OK)
			{

				strcpy(gps_db.file, file);
				
				db_data_to_gpsbuf(gps_data_buf);
				
				
			} else if (db_status == GPS_DB_LOAD_AGAIN)
			{
				db_status = old_db_status;
			}
		}


		if (pfds[2].revents & POLLIN)   // gps.rec update timer
		{

			time_t t0;
			time_t t1;
			
			if (read(rec_timer_fd, &ticks, sizeof(ticks)) != sizeof(ticks))
			{
				_eprintf("timer fd read error\n");
				continue;
			}

			
			_eprintf("gps.rec upated, %d %d\n",
				 gps_db.rtm_upload.tm,
				 gps_db.rtm_upload.date);
			
			pthread_mutex_lock(&(gps_db.mutex));

			t1 = gps_to_calender_time(&(gps_db.rtm_upload));
			t0 = gps_to_calender_time(&(gps_db.rtm_init));

			rtm = gps_db.rtm_upload;

			if (t1 < t0)
			{
				// something bug has happened.
				// at least try to recovery from the initial record

				_eprintf("GSP rec time error, %d-%d, %d-%d\n",
					 gps_db.rtm_upload.date,
					 gps_db.rtm_upload.tm,
					 gps_db.rtm_init.date,
					 gps_db.rtm_init.tm);
				
				rtm = gps_db.rtm_init;
			}


			pthread_mutex_unlock(&(gps_db.mutex));

			gps_rec_log_write(&rtm);
			
		}

		
	}
	
gps_acq_exit:

	pthread_mutex_lock(&(gps_db.mutex));
	
	gps_rec_log_write(&(gps_db.rtm_upload));

	pthread_mutex_unlock(&(gps_db.mutex));

	if (gps_timer_fd > 0)
	{
		close(gps_timer_fd);
	}

	if (rec_timer_fd > 0)
	{
		close(rec_timer_fd);
	}

	if (nm_sub_sock >= 0)
	{
		nn_close(nm_sub_sock);
	}



	
	_eprintf ("jupload gps acq exit\n");

	pthread_exit(NULL);
	
}



#define CURL_CALLBACK_BUF_LEN 64

static char curl_callback_data_buf[CURL_CALLBACK_BUF_LEN];

static int cdb_len = 0;         // curl callback data buf length

#define MAX_URL_LEN GPS_HTTP_MAX_URL_LEN

static char url[MAX_URL_LEN];

extern int upload_simulate;

void *upload_gps_thread(void *arg)
{

#define UPLOAD_NUM_FDS  2
	
	int nm_sub_sock = -1;
	int nm_sub_fd;

	
	struct pollfd pfds[UPLOAD_NUM_FDS];

	
	
	GPS_BUF_s *gps_data_buf = (GPS_BUF_s *)arg;
	GPS_DATA_s upload_data;
	GPS_DATA_s *pData;

	int r;
	
	size_t sz;
	struct curl_waitfd curl_extra_fds[1];

	_eprintf("GPS uploading starts(send)...\n");
	

	if ((nm_sub_sock = nn_socket(AF_SP, NN_SUB)) < 0)
	{
		_eprintf("nn_sock error\n");
		goto gps_upload_exit;
        }

	// subscribe to everything ("" means all topics)

	if (nn_setsockopt(nm_sub_sock, NN_SUB, NN_SUB_SUBSCRIBE, "", 0) < 0)
	{
		_eprintf("nn_setsockopt error\n");
		goto gps_upload_exit;
        }
	
	if (nn_connect(nm_sub_sock, PUBSUB_URL) < 0)
	{
                _eprintf("nn_bind error\n");
		goto gps_upload_exit;
	}


	sz = sizeof(nm_sub_fd);
	
	r = nn_getsockopt (nm_sub_sock, NN_SOL_SOCKET,
			   NN_RCVFD, &nm_sub_fd, &sz);

	if (r < 0)
	{
		_eprintf("nn_getsockopt error\n");
		goto gps_upload_exit;
	}

	memset(pfds, 0, sizeof(pfds));
	
	pfds[0].fd = nm_sub_fd;         
	pfds[0].events = POLLIN;

	pfds[1].fd = gps_data_buf -> can_consume_eventfd;
	pfds[1].events = POLLIN;

	curl_extra_fds[0].fd = nm_sub_fd;
	curl_extra_fds[0].events = CURL_WAIT_POLLIN;
	curl_extra_fds[0].revents = 0;
       

	while(1)
	{

		r = poll(pfds, UPLOAD_NUM_FDS, -1);   // no timeout

		if (r < 0)
		{
			_eprintf("poll error\n");
			
			continue;
		}


		if (pfds[0].revents & POLLIN)
		{

			PUB_MSG_S msg;
			
			if ((r = nn_recv(nm_sub_sock, &msg, sizeof(msg), 0)) < 0)
			{
				_eprintf("nn_recv error\n");
			}

			if (r == sizeof(PUB_MSG_S))
			{
				// printf ("%s(%d): got msg %d\n", PROC_NAME, __LINE__, msg -> type);
				
				if ((msg.type == THREAD_EXIT) ||
				    (msg.type == THREAD_GPS_EXIT))
				{
					goto gps_upload_exit;
				}
			}
		
		}


		if (pfds[1].revents & POLLIN)     // gps data available
		{

			uint64_t u;

			ssize_t s;

			int nr_data;

			s = read(gps_data_buf -> can_consume_eventfd, &u, sizeof(uint64_t));
			
			if (s != sizeof(uint64_t))
			{
				_eprintf("read event fd error, %d\n", (int)s);
				
				continue;
			}

	
			for (nr_data = 0; nr_data < (int)u; nr_data++)
			{
			
				pthread_mutex_lock(&(gps_data_buf->mutex));
				
				if (!gps_buf_is_empty(gps_data_buf))
				{
					pData = gps_data_pop(gps_data_buf, &upload_data);
				} else
				{
					pData = NULL;
					break;
				}

				pthread_mutex_unlock(&(gps_data_buf->mutex));
				
				if (pData)
				{
					
					JTIME_S jtm_gps;
					JTIME_S jtm_sys;
					
					GPS_REC_TIME_s rtm;
					
					HTTP_SEND_STATUS_E http_status;
					
					rtm.tm = pData -> sys_tm;
					rtm.date = pData -> sys_date;
					gps_to_jtime(&rtm, &jtm_sys);
					
					
					rtm.tm = pData -> gps_tm;
					rtm.date = pData -> gps_date;
					gps_to_jtime(&rtm, &jtm_gps);
					
					
					// quectel is spitting out latitude and longitude
					// in ddmm.mmmmmm and dddmm.mmmmmm (precision can
					// be adjusted to 4 or 6 digits after decimal)
					
					// Due to the way floating point in stored in minmea.h
					// we can only store MAX 5 digits after decimal.
					// the lost of precision is less than 1cm. 
					
					// currently, quectel is set for ddmm.mmmmmm and
					// dddmm.mmmmmm. 
					
					
				
					if (gpsNgsensor_log_enable == 0)
					{
						
				
						s = snprintf(url,MAX_URL_LEN,
							     "%sdevice_id=%s&lat=%s&lon=%s&"
							     "speed=%s&"
							     "GPS_DATE_TIME=%08d%02d:%02d:%02d&"
							     "DEVICE_DATE_TIME=%08d%02d:%02d:%02d",
							     pURL_config -> gps_upload_url_addr,
							     device_id_str,
							     pData -> latitude,
							     pData -> longitude,
							     pData -> speed,
							     pData -> gps_date,
							     jtm_gps.hour,
							     jtm_gps.min,
							     jtm_gps.sec,
							     pData -> sys_date,
							     jtm_sys.hour,
							     jtm_sys.min,
							     jtm_sys.sec);
					} else
					{

						s = snprintf(url,MAX_URL_LEN,
							     "%sdevice_id=%s&lat=%s&lon=%s&"
							     "speed=%s&"
							     "GPS_DATE_TIME=%08d%02d:%02d:%02d&"
							     "DEVICE_DATE_TIME=%08d%02d:%02d:%02d&"
							     "gsensorX=%d&gsensorY=%d&gsensorZ=%d&"
							     "distance=%.1f",
							     pURL_config -> gps_upload_url_addr,
							     device_id_str,
							     pData -> latitude,
							     pData -> longitude,
							     pData -> speed,
							     pData -> gps_date,
							     jtm_gps.hour,
							     jtm_gps.min,
							     jtm_gps.sec,
							     pData -> sys_date,
							     jtm_sys.hour,
							     jtm_sys.min,
							     jtm_sys.sec,
							     pData -> gX,
							     pData -> gY,
							     pData -> gZ);

					}
				
printf("dd gps url: %s\n", url);

					
					if (print_lev > 0)
					{
						struct timeval tv;
						
						gettimeofday(&tv, NULL);
						
						_eprintf ("http url send: %s, %u:%u\n", url, (uint32_t)(tv.tv_sec), (uint32_t)(tv.tv_usec));
					}
					
					nr_http_error = 0;
					
					do
					{
						curl_extra_fds[0].revents = 0;

						
						// http_status = http_url_send(url, curl_extra_fds, 0);  // blocking call

						http_status = http_url_send(url, NULL, 0);  // blocking call
						
						
						if ( http_status != HTTP_SEND_STATUS_OK)
						{
							++nr_http_error;
							
							if (http_status == HTTP_SEND_STATUS_EXTRA_FDS)
							{
								_eprintf ("uploading interrupt\n");
								
								// goto gps_upload_exit;
								
								goto upload_break;
								
							} else
							{
								if (nr_http_error <= 10)
								{
									WAITMS(10);
									
								} 
								{
									if (nr_http_error > 20)
									{
										WAITMS(1000);
										nr_http_error = 20;
										_eprintf("GPS upload error\n");
										
									} else
									{
										WAITMS(100);
									}
								}
							}
						} else
						{
							nr_http_error = 0;
						}
					
					} while (http_status != HTTP_SEND_STATUS_OK);
					
				}
				
				
#warning "bbbb"

upload_break:

			
			
				pthread_mutex_lock(&(gps_db.mutex));
				
				assert(pData -> sys_date != 0);
				
				gps_db.rtm_upload.tm   = pData -> sys_tm;
				gps_db.rtm_upload.date = pData -> sys_date;
				
				pthread_mutex_unlock(&(gps_db.mutex));
				
				
			}
		}
	}
	

gps_upload_exit:

	
	_eprintf ("jupload gps upload exit\n");


	
	if (nm_sub_sock >= 0)
	{
		nn_close(nm_sub_sock);
	}
	
	
	pthread_exit(NULL);
}






static HTTP_SEND_STATUS_E http_url_send(const char *url,
					struct curl_waitfd extra_fds[],
					unsigned int extra_nfds)
{

#define CURLOPT_UPLOAD_TIMEOUT  (10)    // sec
	
#define CURL_UPLOAD_TIMEOUT_MS (CURLOPT_UPLOAD_TIMEOUT * 1000)



	CURL *http_handle;
	CURLM *multi_handle;

	struct json_object *root;
	struct json_object *rtn_code;
	const char *str;

	HTTP_SEND_STATUS_E r;
	int i;
	int curl_still_running;


	int upload_timeo_ms;

	r = HTTP_SEND_STATUS_OK;
	
	if (upload_simulate)
	{
		_eprintf("http send url: %s\n", url);
		
		return r;
	}
	

	// printf ("%s\n", __FUNCTION__);
	
	http_handle = curl_easy_init();

	if (http_handle == NULL)
	{
		_eprintf("curl easy init error\n");
		
		return HTTP_SEND_STATUS_CURL_API_ERROR;
	}
       

	multi_handle = curl_multi_init();

	if (multi_handle == NULL)
	{
		_eprintf("curl multi init error\n");
		
		curl_easy_cleanup(http_handle);
		return HTTP_SEND_STATUS_CURL_API_ERROR;
	}

	
	curl_callback_data_buf[0] = '\0';

	
	/* First set the URL that is about to receive our POST. This URL can
	   just as well be a https:// URL if that is what should receive the
	   data. */
	
	cdb_len = 0;
	
	curl_easy_setopt(http_handle, CURLOPT_URL, url);
	
	curl_easy_setopt(http_handle, CURLOPT_WRITEFUNCTION, curl_callback);

	curl_easy_setopt(http_handle, CURLOPT_CONNECTTIMEOUT, 5);
	curl_easy_setopt(http_handle, CURLOPT_TIMEOUT, CURLOPT_UPLOAD_TIMEOUT);

	// curl_easy_setopt(http_handle, CURLOPT_NOSIGNAL, 1);
	
	curl_easy_setopt(http_handle, CURLOPT_LOW_SPEED_LIMIT, 100L);
	curl_easy_setopt(http_handle, CURLOPT_LOW_SPEED_TIME, 5L);
	
	curl_multi_add_handle(multi_handle, http_handle);

	curl_still_running = 1;

	upload_timeo_ms = 0;
	
 
	do
	{
		CURLMcode mc; /* curl_multi_poll() return code */ 
		int numfds;
		long curl_timeo = -1;
		
		/* we start some action by calling perform right away */ 
		mc = curl_multi_perform(multi_handle, &curl_still_running);
		
		curl_multi_timeout(multi_handle, &curl_timeo);
		
		if(curl_still_running)
		{
			
			if(curl_timeo >= 0)
			{
				if(curl_timeo > 1000)
				{
					curl_timeo = 1000;
				}
				
			} else
			{
				curl_timeo = 100;
			}
				

			mc = curl_multi_poll (multi_handle,
					      extra_fds, extra_nfds,
					      curl_timeo, &numfds);

			_dprintf("curl_multi_poll %d, %d, %d %d\n",
				 mc, numfds,
				 upload_timeo_ms, curl_timeo);
				
			if(mc != CURLM_OK)
			{
				_eprintf("curl_multi_poll() failed, code %d.\n", mc);
				r = HTTP_SEND_STATUS_CURL_API_ERROR;
				goto curl_cleanup;
			}
			
			/* 'numfds' being zero means either a timeout or no file descriptors to
			 * wait for. Try timeout on first occurrence, then assume no file
			 * descriptors and no file descriptors to wait for means wait for 100
			 * milliseconds. 
			 *
			 * This number can include both libcurl internal descriptors 
			 * as well as descriptors provided in extra_fds.
			 */ 
			
			if (numfds == 0)
			{
				upload_timeo_ms += curl_timeo;
				
				if (upload_timeo_ms >= CURL_UPLOAD_TIMEOUT_MS)
				{
					_lprintf(0, "upload timeout\n");
					
					r = HTTP_SEND_STATUS_TIMEOUT;
					
					goto curl_cleanup;
				}
				
				
			} else
			{
				for( i = 0; i < (int) extra_fds; i++)
				{
					if (extra_fds[i].revents & CURL_WAIT_POLLIN)
					{
						r = HTTP_SEND_STATUS_EXTRA_FDS;
						goto curl_cleanup;
					}
				} 
				
			}
			
		} else
		{
			_dprintf ("http stopped running, %d\n",mc);
		}
		
		
	} while (curl_still_running);
	
		
	if ((curl_still_running == 0) && (r == HTTP_SEND_STATUS_OK))
	{
		// check callback data
		
		//_lprintf (0, "callback data: %s\n", curl_callback_data_buf);


		if (print_lev > 0)
		{
			struct timeval tv;
			
			gettimeofday(&tv, NULL);
			
			_eprintf ("rtn time %d:%d\n", (uint32_t)(tv.tv_sec), (uint32_t)(tv.tv_usec));
		}
		
		
		root = json_tokener_parse(curl_callback_data_buf);
		
		if (root)
		{
			// { "code": "0", "msg": "success"}
			// { "code": "1", "msg": "error"}
			// { "code": "2", "msg": [string]}

			_lprintf(0, "rtn status: %s\n", curl_callback_data_buf);
			
			rtn_code = json_object_object_get(root, "code");
			
			if (rtn_code)
			{
				int n;
				
				str = json_object_get_string(rtn_code);
				
				n = atoi(str);
				
				if ( n != 0)
				{
					r = HTTP_SEND_STATUS_SERVER_ERROR;
				}
			}
			
			json_object_put(root);
			
		} else
		{
			_lprintf(0, "invalid callback json obj: %s\n", curl_callback_data_buf);
			
			r = HTTP_SEND_STATUS_SERVER_ERROR;
			
		}
		
	}
	
curl_cleanup:

	// _dprintf("curl cleanup\n");


	// it is posssible remove_handle takes up to 1 minute

	
	curl_multi_remove_handle(multi_handle, http_handle);  

	
	curl_easy_cleanup(http_handle);

	
	curl_multi_cleanup(multi_handle);

	// _eprintf ("url send exit %d\n", r);

	return r;

}



static size_t curl_callback(void *contents, size_t size, size_t nmemb, void *userp)
{
	// callback as soon as there is data, even just 1 byte
	// can be called mulitple times
	
	size_t n = size * nmemb;   // size is always  1
	int i;

	(void) userp;
	
	if ((cdb_len + (int)n) < CURL_CALLBACK_BUF_LEN)
	{

		for (i = 0; i < (int)n; i++)
		{
			curl_callback_data_buf[cdb_len++] = *((char *)contents + i);
		}
		
		curl_callback_data_buf[cdb_len] = '\0';
	} else
	{
		_eprintf ("curl data buff too small\n");
		
	}
	
	return n;
}
