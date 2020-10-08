"""utility module providing http manipulation"""

from enum import Enum
from http.client import HTTPSConnection
import json
import xmltodict

class HttpConstants(Enum):

    SUCCESS        = 200
    REDIRECTION    = 300
    CLIENT_ERR     = 400
    SERVER_ERR     = 500
    
    TYPE_REQUEST   = 'GET'
    TYPE_DECODE    = 'utf8'

def get_response(host, req):
    result = None
    conn = HTTPSConnection(host)
    while True:
        conn.request(HttpConstants.TYPE_REQUEST.value, req)
        res = conn.getresponse()
        if res.status == HttpConstants.SERVER_ERR.value:
            continue
        else:
            if res.status == HttpConstants.SUCCESS.value:
                result = res.read().decode(HttpConstants.TYPE_DECODE.value)
            break
    conn.close()
    return result

def convert_json_to_dict(json_str):
    return json.loads(json_str)

def convert_xml_to_dict(xml_str):
    return xmltodict.parse(xml_str)
