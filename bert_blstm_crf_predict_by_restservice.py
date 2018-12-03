from bert.bert import modeling, optimization, tokenization
import tensorflow as tf
from blstm_crf.utils.util import convert_single_example
from blstm_crf.utils.data_processor import NerProcessor, InputExample, InputFeatures
import os
import json
import codecs
from blstm_crf.utils.util import file_based_input_fn_builder, filed_based_convert_examples_to_features
import logging
import pickle
from blstm_crf.utils.conlleval import return_report
import traceback
from blstm_crf.bert_blstm_crf import model_fn_builder
from flask import Flask, jsonify, request
import traceback
import random
import re
from bert.utils.get_request_ip import get_ip

#gunicorn -c blstm_crf_gun.py bert_blstm_crf_predict_by_restservice:app
app = Flask(__name__)
config_path = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'blstm_crf'), 'config')
params_path = os.path.join(config_path, 'params.json')
log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'log.txt')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename=log_path)
logger = logging.getLogger(__name__)

processors = {
    "ner": NerProcessor
}
logger.info("load param")
with open(params_path) as param:
    params_dict = json.load(param)
params = tf.contrib.training.HParams(**params_dict)
bert_path = params.bert_path
root_path = params.root_path
bert_config_file = os.path.join(bert_path, params.bert_config_file)
logger.info("load bert config")
bert_config = modeling.BertConfig.from_json_file(bert_config_file)
if params.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (params.max_seq_length, bert_config.max_position_embeddings))
logger.info("clean train's output_dir")
data_config_path = os.path.join(root_path, params.data_config_path)
output_dir = os.path.join(root_path, params.output_dir)
task_name = params.task_name.lower()
if task_name not in processors:
    raise ValueError('Task not found: %s' % (task_name))

processor = processors[task_name]()
label_list = processor.get_labels()
vocab_file = os.path.join(bert_path, params.vocab_file)
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file,
                                       do_lower_case=params.do_lower_case)

logger.info('estimator runconfig')
run_config = tf.estimator.RunConfig(model_dir=output_dir,
                                    save_checkpoints_steps=params.save_checkpoints_steps,
                                    tf_random_seed=19830610)
if os.path.exists(data_config_path):
    with codecs.open(data_config_path) as fd:
        data_config = json.load(fd)
else:
    data_config = {}
data_dir = os.path.join(root_path, params.data_dir)
init_checkpoint = os.path.join(bert_path, params.init_checkpoint)
logger.info("achieve model_fn")
model_fn = model_fn_builder(
    bert_config=bert_config,
    num_labels=len(label_list) + 1,
    init_checkpoint=init_checkpoint,
    learning_rate=params.learning_rate,
    num_train_steps=None,
    num_warmup_steps=None,
    use_one_hot_embeddings=False,
    out_params=params
)

batch_size = params.predict_batch_size
estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    config=run_config,
    params={'batch_size': batch_size},
    model_dir=output_dir
)

if not os.path.exists(data_config_path):
    with codecs.open(data_config_path, 'a', encoding='utf-8') as fd:
        json.dump(data_config, fd)
with codecs.open(os.path.join(output_dir, 'label2id.pkl'), 'rb') as rf:
    label2id = pickle.load(rf)
    id2label = {value: key for key, value in label2id.items()}
label_map = {}
for (i, label) in enumerate(label_list, 1):
    label_map[label] = i
max_seq_length = params.max_seq_length
chars = [chr(i) for i in range(97,123)] + [str(i) for i in range(10)]

def random_chars():
    num = random.sample(chars, 32)
    return ''.join(num)

def predict(lines):

    predict_examples = []

    for line in lines:
        print(line)
        words = []
        labels = []
        line = re.sub(' ', '', line)
        for i in range(len(line)):
            words.append(line[i])
            labels.append('O')
        words = ' '.join(words)
        labels = ' '.join(labels)
        text = tokenization.convert_to_unicode(words)
        label = tokenization.convert_to_unicode(labels)
        # print('text len: ', len(text))
        # print('label len: ', len(label))
        example = InputExample(guid='predict', text=text, label=label)
        predict_examples.append(example)
    predict_file = os.path.join(output_dir, "predict.tf_record_" +random_chars())
    filed_based_convert_examples_to_features(predict_examples, label_list,
                                             params.max_seq_length, tokenizer,
                                             predict_file, mode="test", output_dir=output_dir)

    predict_drop_remainder = False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=params.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    result = estimator.predict(input_fn=predict_input_fn)
    res = result_to_pair(predict_examples, result)
    if os.path.exists(predict_file):
        os.remove(predict_file)
    return res


#["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]
def result_to_pair(predict_examples, result):
    ner_result = []
    for predict_line, prediction in zip(predict_examples, result):
        idx = 0
        line = ''
        res = []
        # print("prediction_line: ", predict_line)
        line_token = str(predict_line.text).split(' ')
        label_token = str(predict_line.label).split(' ')
        if len(line_token) != len(label_token):
            logger.info(predict_line.text)
            logger.info(predict_line.label)
        labels = []
        words = line_token
        for id in prediction['pred_ids']:
            if id == 0:
                labels.append('X')
            else:
                labels.append(id2label[id])
        labels = labels[1:len(words)+1]
        # print("words: ", words)
        # print("labels: ", labels)
        i = 0
        while(i < len(labels)):
            if labels[i] == 'B-PER':
                start = i
                while (i + 1 < len(labels)):
                    if labels[i+1] == 'I-PER':
                        i += 1
                    else:
                        i += 1
                        break
                res.append(''.join(words[start:i]))
            elif labels[i] == 'B-ORG':
                start = i
                while (i + 1 < len(labels)):
                    if labels[i+1] == 'I-ORG':
                        i += 1
                    else:
                        i += 1
                        break
                res.append(''.join(words[start:i]))
            elif labels[i] == 'B-LOC':
                start = i
                while (i + 1 < len(labels)):
                    if labels[i+1] == 'I-LOC':
                        i += 1
                    else:
                        i +=1
                        break
                res.append(''.join(words[start:i]))
            else:
                i += 1
        # print("res: ", res)
        ner_result.append(' '.join(res))
    return ner_result


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
            res = predict(msg)
            return jsonify({'state': 'success',
                            'res': res})
    except Exception as e:
        logger.warning("state: "+ traceback.format_exc())
        return jsonify({"state": "predict failed!!!",
                        'trace': traceback.format_exc()})



# if __name__ == '__main__':
# #['我 变 而 以 书 会 友 以 ， 把 欧 美 、 港 台 流 行 的 食 品 类 图 谱 、 画 册 、 工 具 书 汇 集 一 堂']
#     line = ['习近平抵达巴拿马城开始对巴拿马进行国事访问','我们变而以书会友,以书结缘，把欧美、港台流行的食品类图谱、画册、工具书汇集一堂',
#             '中国国家主席习近平应邀同美国总统特朗普在阿根廷首都布宜诺斯艾利斯共进晚餐并举行会晤。',
#             '中美在促进世界和平和繁荣方面共同肩负着重要责任。一个良好的中美关系符合两国人民根本利益，也是国际社会的普遍期待。']
#
#     res = predict(line)
#     print(res)
    # tf.app.run()