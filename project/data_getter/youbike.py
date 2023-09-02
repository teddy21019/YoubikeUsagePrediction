import requests as rq
from data_getter.complete_listener import announce

YB_OB_COLLECTION_URL = "https://quality.data.gov.tw/dq_download_json.php?nid=150635&md5_url=e51f13f6f3f14df40f17c175161be3cc"


def get_youbike_OD_data():

    res = rq.get(YB_OB_COLLECTION_URL).json()
    print(res)

    for yb_data in res:
        print(yb_data['fileURL'])


    announce("finish", "Youbike OD Data")

def get_youbike_coord_data():
    announce("finish", "Youbike Coordination Data")

def get_youbike_address_data():
    ...