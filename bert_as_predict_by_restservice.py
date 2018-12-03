from bert.bert.extract_features import model_fn_builder
from bert.bert import modeling
import tensorflow as tf
import os
import json
from bert.utils.import_util import cls_from_str
from tensorflow.python.estimator.estimator import Estimator
from bert.bert.extract_features import convert_lst_to_features
from bert.bert import tokenization
from flask import Flask, request, jsonify
import logging
import traceback
from bert.utils.get_request_ip import get_ip
import collections
import threading
from bert.utils.threads_util import MyThread

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='log.txt')
logger = logging.getLogger(__name__)
config_path = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), "bert_blstm_crf"), 'config')
params_path = os.path.join(config_path, 'params.json')

with open(params_path) as param:
    params_dict = json.load(param)
params = tf.contrib.training.HParams(**params_dict)
root_path = params.root_path
bert_path = params.bert_path
config_fp = os.path.join(bert_path, params.bert_config_file)
checkpoint_fp = os.path.join(bert_path,params.init_checkpoint)
pooling_strategy = cls_from_str(params.pooling_strategy)
pooling_layer = params.pooling_layer
model_fn = model_fn_builder(
    modeling.BertConfig.from_json_file(config_fp),
    init_checkpoint=checkpoint_fp,
    pooling_strategy=pooling_strategy,
    pooling_layer=pooling_layer
)
estimator = Estimator(model_fn)
vocab_fp = os.path.join(bert_path, params.vocab_file)
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_fp)


def get_input_fn(msg):
    res = []
    def gen():
        tmp_f = list(convert_lst_to_features(msg, params.max_seq_length, tokenizer))
        tmp_dict = {}
        tmp_dict['input_ids'] = [f.input_ids for f in tmp_f]
        tmp_dict['input_mask'] = [f.input_mask for f in tmp_f]
        tmp_dict['input_type_ids'] = [f.input_type_ids for f in tmp_f]
        yield tmp_dict


    def input_fn():
        return (tf.data.Dataset.from_generator(
            gen,
            output_types={'input_ids': tf.int32,
                          'input_mask': tf.int32,
                          'input_type_ids': tf.int32},
            output_shapes={
                'input_ids': (None, params.max_seq_length),
                'input_mask': (None, params.max_seq_length),
                'input_type_ids': (None, params.max_seq_length)}))
    for r in estimator.predict(input_fn=input_fn):
        res.append(r['encodes'])
    return res
# headers = {"Content-Type":"application/json"}
# data = {"msg": "计算机博士的话|隔壁实验室有去腾讯开80w的|当然这应该是比较优秀的博士"}
# url = "http://0.0.0.0:8888/predict"
@app.route('/predict', methods=['post'])
def predict_server():
    try:
        try:
            temp_data = request.get_data()
            json_data = json.loads(temp_data)
            logger.info(json_data)
            ip = get_ip()
            logger.info("ip: {}".format(ip))
        except Exception as e:
            logger.warning("request failed or request load failed!!!" + traceback.format_exc())
            return jsonify({"state": "request failed or request load failed!!!",
                            'trace': traceback.format_exc()})
        if 'msg' not in json_data:
            logger.warning('msg field must be in json request!!!')
            return jsonify({'state': "msg field must be in json request!!!"})
        else:
            msg = json_data['msg']
            data = msg.split('|')
            res = get_input_fn(data)
            res = [a.tolist() for a in res]
            return jsonify({'state': 'success',
                            'res': res})
    except Exception as e:
        logger.warning("state: "+ traceback.format_exc())
        return jsonify({"state": "predict failed!!!",
                        'trace': traceback.format_exc()})

# headers = {"Content-Type":"application/json"}
# data = {"batch_size":64,"msg": "计算机博士的话|隔壁实验室有去腾讯开80w的|当然这应该是比较优秀的博士"}
# url = "http://0.0.0.0:8888/predict"

@app.route('/multithreads_predict', methods=['post'])
def multithreads_predict_server():
    try:
        try:
            temp_data = request.get_data()
            json_data = json.loads(temp_data)
            logger.info(json_data)
            ip = get_ip()
            logger.info("ip: {}".format(ip))
        except Exception as e:
            logger.warning("request failed or request load failed!!!" + traceback.format_exc())
            return jsonify({"state": "request failed or request load failed!!!",
                            'trace': traceback.format_exc()})
        if 'msg' not in json_data:
            logger.warning('msg field must be in json request!!!')
            return jsonify({'state': "msg field must be in json request!!!"})
        else:
            msg = json_data['msg']
            data = msg.split('|')
        if 'batch_size' not in json_data:
            batch_size = 64
        else:
            batch_size = json_data['batch_size']
        batchs = int(len(data) / batch_size)
        print("========================")
        print("batchs: ", batchs)
        print("========================")
        threads = []
        for i in range(batchs):
            batch_data = data[i * batch_size: (i + 1) * batch_size]
            print(batch_data)
            t = MyThread(get_input_fn, batch_data, get_input_fn.__name__)
            threads.append(t)
            t.start()
            print(threads)
        other = len(data) - batch_size * batchs
        if other != 0:
            batch_data = data[batch_size * batchs:]
            t = MyThread(get_input_fn, batch_data, get_input_fn.__name__)
            threads.append(t)
            t.start()
        for i in range(len(threads)):
            threads[i].join(100000)
        res = []
        for i in range(len(threads)):
            tmp = [a.tolist() for a in threads[i].get_result()]
            res.extend(tmp)
        except_num = [1 for r in res if r == None]
        if len(except_num) >= 1:
            logger.warning("state: threads except ")
            logger.warning("trace: {}".format(traceback.format_exc()))
            return jsonify({'state': 'threads except',
                            'trace': traceback.format_exc()})
        else:
            return jsonify({'state': 'success',
                            'res': res})
    except Exception as e:
        logger.warning("state: "+ traceback.format_exc())
        return jsonify({"state": "predict failed!!!",
                        'trace': traceback.format_exc()})



if __name__ == '__main__':
    msg = ["计算机博士的话", "隔壁实验室有去腾讯开80w的", "当然这应该是比较优秀的博士"]
    res = get_input_fn(msg)
    print(res[0])
    batch_size = 1
    batchs = int(len(msg) / batch_size)
    threads = []
    for i in range(batchs):
        batch_data = msg[i * batch_size: (i + 1) * batch_size]
        # print(batch_data)
        t = MyThread(get_input_fn, batch_data, get_input_fn.__name__)
        threads.append(t)
        t.start()
        print(threads)
    other = len(msg) - batch_size * batchs
    if other != 0:
        batch_data = msg[batch_size * batchs:]
        t = MyThread(get_input_fn, batch_data, get_input_fn.__name__)
        threads.append(t)
        t.start()
    for i in range(len(threads)):
        threads[i].join(100000)
    res = []
    for i in range(len(threads)):
        tmp = [a.tolist() for a in threads[i].get_result()]
        res.extend(tmp)
    except_num = [1 for r in res if r == None]
    if len(except_num) >= 1:
       print("==failed==")
    else:
        print(res[0])





