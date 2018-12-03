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
import collections
import numpy as np
import random


config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config')
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
    words = []
    labels = []
    for line in lines:
        for i in range(len(line)):
            words.append(line[i])
            labels.append('O')
        words = ' '.join(words)
        labels = ' '.join(labels)
        text = tokenization.convert_to_unicode(words)
        label = tokenization.convert_to_unicode(labels)
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
        print("prediction_line: ", predict_line)
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
        labels = labels[:len(words)]
        i = 0
        while(i < len(labels)):
            if labels[i] == 'B-PER':
                if i + 1 < len(labels) and labels[i+1] == 'I-PER':
                    res.append(words[i:i+2])
                    i += 2
                else:
                    res.append(words[i])
                    i += 1
            elif labels[i] == 'B-ORG':
                if i + 1 < len(labels) and labels[i+1] == 'I-ORG':
                    res.append(words[i:i+2])
                    i += 2
                else:
                    res.append(words[i])
                    i += 1
            elif labels[i] == 'B-LOC':
                if i + 1 < len(labels) and labels[i + 1] == 'I-LOC':
                    res.append(words[i:i + 2])
                    i += 2
                else:
                    res.append(words[i])
                    i += 1
            else:
                i += 1
        ner_result.append(' '.join(res))
    return ner_result



def main(_):


    token_path = os.path.join(output_dir, "token_test.txt")
    if os.path.exists(token_path):
        os.remove(token_path)


    predict_examples = processor.get_test_tmp_examples(data_dir)

    predict_file = os.path.join(output_dir, "predict.tf_record")
    filed_based_convert_examples_to_features(predict_examples, label_list,
                                             params.max_seq_length, tokenizer,
                                             predict_file, mode="test", output_dir=output_dir)

    logger.info("***** Running prediction*****")
    logger.info("  Num examples = %d", len(predict_examples))
    logger.info("  Batch size = %d", params.predict_batch_size)
    predict_drop_remainder = False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=params.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    predicted_result = estimator.evaluate(input_fn=predict_input_fn)
    output_eval_file = os.path.join(output_dir, "predicted_results.txt")
    with codecs.open(output_eval_file, "w", encoding='utf-8') as writer:
        tf.logging.info("***** Predict results *****")
        for key in sorted(predicted_result.keys()):
            tf.logging.info("  %s = %s", key, str(predicted_result[key]))
            writer.write("%s = %s\n" % (key, str(predicted_result[key])))

    if os.path.exists(predict_file):
        os.remove(predict_file)

    result = estimator.predict(input_fn=predict_input_fn)
    print("result: ", result)
    print(type(result))
    output_predict_file = os.path.join(output_dir, "label_test.txt")

    def result_to_pair(writer):
        for predict_line, prediction in zip(predict_examples, result):
            idx = 0
            line = ''
            # print("prediction: ", prediction)
            line_token = str(predict_line.text).split(' ')
            label_token = str(predict_line.label).split(' ')
            if len(line_token) != len(label_token):
                logger.info(predict_line.text)
                logger.info(predict_line.label)
            for id in prediction['pred_ids']:
                if id == 0:
                    continue
                curr_labels = id2label[id]
                if curr_labels in ['[CLS]', '[SEP]']:
                    continue
                try:
                    line += line_token[idx] + ' ' + label_token[idx] + ' ' + curr_labels + '\n'
                except Exception as e:
                    logger.info(e)
                    logger.info(predict_line.text)
                    logger.info(predict_line.label)
                    line = ''
                    break
                idx += 1
            writer.write(line + '\n')

    with codecs.open(output_predict_file, 'w', encoding='utf-8') as writer:
        result_to_pair(writer)

    eval_result = return_report(output_predict_file)
    logger.info(eval_result)

if __name__ == '__main__':

    line = ['我们变而以书会友,以书结缘，把欧美、港台流行的食品类图谱、画册、工具书汇集一堂']

    res = predict(line)
    print(res)
    # tf.app.run()