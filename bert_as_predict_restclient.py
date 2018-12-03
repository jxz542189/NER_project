from flask import request
import requests
import json

#单线程运行
headers = {"Content-Type":"application/json"}
data = {"msg": "计算机博士的话|隔壁实验室有去腾讯开80w的|当然这应该是比较优秀的博士"}
url = "http://0.0.0.0:8888/predict"
d = requests.post(url=url, json=data, headers=headers)
json_data = d.json()
res = json_data['res']
print(type(res[0]))
#多线程运行  存在问题还没有解决 requests.exceptions.ConnectionError: ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response',))
# headers = {"Content-Type":"application/json"}
# data = {"msg": "计算机博士的话|隔壁实验室有去腾讯开80w的|当然这应该是比较优秀的博士", "batch_size": 1}
# url = "http://0.0.0.0:8888/multithreads_predict"
# d = requests.post(url=url, json=data, headers=headers)
# json_data = d.json()
# res = json_data['res']
# print(type(res[0]))