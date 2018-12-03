from flask import request
import requests
import json

#单线程运行
lines = ['习近平抵达巴拿马城开始对巴拿马进行国事访问','我们变而以书会友,以书结缘，把欧美、港台流行的食品类图谱、画册、工具书汇集一堂',
        '中国国家主席习近平应邀同美国总统特朗普在阿根廷首都布宜诺斯艾利斯共进晚餐并举行会晤。',
        '中美在促进世界和平和繁荣方面共同肩负着重要责任。一个良好的中美关系符合两国人民根本利益，也是国际社会的普遍期待。']
headers = {"Content-Type":"application/json"}
data = {"msg": lines}
url = "http://0.0.0.0:8888/predict"
d = requests.post(url=url, json=data, headers=headers)
json_data = d.json()
res = json_data['res']
print(res)

#多线程运行  存在问题还没有解决 requests.exceptions.ConnectionError: ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response',))
# headers = {"Content-Type":"application/json"}
# data = {"msg": "计算机博士的话|隔壁实验室有去腾讯开80w的|当然这应该是比较优秀的博士", "batch_size": 1}
# url = "http://0.0.0.0:8888/multithreads_predict"
# d = requests.post(url=url, json=data, headers=headers)
# json_data = d.json()
# res = json_data['res']
# print(type(res[0]))