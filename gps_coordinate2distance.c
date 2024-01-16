
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


double degree_to_radian(double);
double gps_location_distance(double, double, double, double);


int main()
{
    double lon1 = 0, lat1 = 0, lon2 = 0, lat2 = 0, distance = 0;

    while(1)
    {
        printf("\n[Location 1] Latitude Longitude: ");
        scanf("%lf %lf", &lat1, &lon1);

        printf("\n[Location 2] Latitude Longitude: ");
        scanf("%lf %lf", &lat2, &lon2);

        distance = gps_location_distance(lon1, lat1, lon2, lat2);

        printf("\nDistance between two locations is %lf meters.\n\n", distance);
    }

    return 0;
}


double degree_to_radian(double degree)
{
    return degree / 180 * M_PI;
}


double gps_location_distance(double lon1, double lat1, double lon2, double lat2)
{
    /*
     *  Calculate physical distance between two gps locations.
     *  Unit: degree and meter.
     *  Reference: https://reurl.cc/G4A4yD
     */

    return acos((sin(degree_to_radian(lat1)) * sin(degree_to_radian(lat2))) + (cos(degree_to_radian(lat1)) * cos(degree_to_radian(lat2))) * (cos(degree_to_radian(lon2) - degree_to_radian(lon1)))) * 6371 * 1000;
}
