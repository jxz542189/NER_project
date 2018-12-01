from bert_blstm_crf.bert.extract_features import model_fn_builder
from bert_blstm_crf.bert import modeling
import tensorflow as tf
import os
import json
from bert_blstm_crf.utils.import_util import cls_from_str
from tensorflow.python.estimator.estimator import Estimator
from bert_blstm_crf.bert.extract_features import convert_lst_to_features
from bert_blstm_crf.bert import tokenization


if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'config')
    params_path = os.path.join(config_path, 'params.json')

    with open(params_path) as param:
        params_dict = json.load(param)
    params = tf.contrib.training.HParams(**params_dict)
    root_path = params.root_path
    bert_path = params.bert_path
    print(params.root_path)
    config_fp = os.path.join(bert_path, params.bert_config_file)
    checkpoint_fp = os.path.join(bert_path,params.init_checkpoint)
    pooling_strategy = cls_from_str(params.pooling_strategy)
    pooling_layer = params.pooling_layer
    print(type(pooling_strategy))
    model_fn = model_fn_builder(
        modeling.BertConfig.from_json_file(config_fp),
        init_checkpoint=checkpoint_fp,
        pooling_strategy=pooling_strategy,
        pooling_layer=pooling_layer
    )
    estimator = Estimator(model_fn)
    vocab_fp = os.path.join(bert_path, params.vocab_file)
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_fp)
    msg = ["计算机博士的话", "隔壁实验室有去腾讯开80w的","当然这应该是比较优秀的博士"]
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
    # for r in estimator.predict(input_fn=input_fn):
    #     print(r['encodes'])
    #     print(len(r['encodes']))
    #     break
    embedding = model.get_sequence_output()





