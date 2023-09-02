"""
Downloads all data needed
"""

from data_getter import *
from data_getter.complete_listener import subscribe

## Youbike Data
if __name__ == "__main__":

    def print_finish(msg):
        print("Finish Downloading: "+ msg)

    subscribe('finish', print_finish)

    get_youbike_OD_data()

    get_youbike_coord_data()

    get_youbike_address_data()

    get_demograhic_data()

    get_income_data()

    get_metro_exit_coord_data()

    get_metro_station_OD_data()

    get_bus_stop_coord_data()

    get_college_coord_data()

    get_highschool_coor_data()

    get_711_coord_data()

    get_taipei_tourist_spot_coord_data()

    get_weather_data()





