/*
    Copyright 2016 Garrett D'Amore <garrett@damore.org>

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom
    the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
    IN THE SOFTWARE.

    "nanomsg" is a trademark of Martin Sustrik
*/

/*  This program serves as an example for how to write a simple PUB SUB
    service, The server is just a single threaded for loop which broadcasts
    messages to clients, every so often.  The message is a binary format
    message, containing two 32-bit unsigned integers.  The first is UNIX time,
    and the second is the number of directly connected subscribers.

    The clients stay connected and print a message with this information
    along with their process ID to standard output.

    To run this program, start the server as pubsub_demo <url> -s
    Then connect to it with the client as pubsub_demo <url>
    For example:

    % ./pubsub_demo tcp://127.0.0.1:5555 -s &
    % ./pubsub_demo tcp://127.0.0.1:5555 &
    % ./pubsub_demo tcp://127.0.0.1:5555 &
    11:23:54 <pid 1254> There are 2 clients connected.
    11:24:04 <pid 1255> There are 2 clients connected.
    ..
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <netinet/in.h>  /* For htonl and ntohl */
#include <unistd.h>

#include <nanomsg/nn.h>
#include <nanomsg/pubsub.h>

#include "nycu_mhw_api.h"

int client(const char *url, const char *name)
{
    int sock;

    if((sock = nn_socket(AF_SP, NN_SUB)) < 0) {
        printf("Socket error at chkpt 1.\n");
        exit(0);
    }

    // subscribe to everything ("" means all topics)
    if(nn_setsockopt(sock, NN_SUB, NN_SUB_SUBSCRIBE, "", 0) < 0)
    {
        printf("Socket error at chkpt 2.\n");
        exit(0);
    }

    if(nn_connect(sock, url) < 0)
    {
        printf("Socket error at chkpt 3.\n");
        exit(0);
    }

    for(;;)
    {
        NYCU_MHW_ALARM_EVENT_s *buf = NULL;
        int bytes = nn_recv(sock, &buf, NN_MSG, 0);

        if (bytes < 0) {
            printf("Socket error at chkpt 4.\n");
        }
        
        printf("CLIENT (%s): RECEIVED\n", name); 
        nn_freemsg(buf);
    }
}

int main (int argc, char **argv)
{
    int rc = 0;

    rc = client("tcp://192.168.1.5:5555", "receiver0");

    exit(rc == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}
