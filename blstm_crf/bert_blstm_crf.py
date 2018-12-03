from bert.bert import modeling, optimization, tokenization
import tensorflow as tf
from blstm_crf.model.lstm_crf_layer import BLSTM_CRF
from blstm_crf.utils import tf_metrics
from tensorflow.contrib.layers.python.layers import initializers
from blstm_crf.utils.data_processor import NerProcessor
import os
import json
import codecs
from blstm_crf.utils.util import file_based_input_fn_builder, filed_based_convert_examples_to_features
import logging
import pickle
from blstm_crf.utils.conlleval import return_report
import traceback


config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config')
params_path = os.path.join(config_path, 'params.json')
log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'log.txt')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename=log_path)
logger = logging.getLogger(__name__)

def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings, params):
    model = modeling.BertModel(config=bert_config,
                               is_training=is_training,
                               input_ids=input_ids,
                               input_mask=input_mask,
                               token_type_ids=segment_ids,
                               use_one_hot_embeddings=use_one_hot_embeddings)
    embedding = model.get_sequence_output()
    max_seq_length = embedding.shape[1].value

    used = tf.sign(tf.abs(input_ids))
    lengths = tf.reduce_sum(used, reduction_indices=1)

    blstm_crf = BLSTM_CRF(embedded_chars=embedding, hidden_unit=params.lstm_size, cell_type=params.cell,
                          num_layers=params.num_layers,
                          droupout_rate=params.dropout_rate, initializers=initializers, num_labels=num_labels,
                          seq_length=max_seq_length, labels=labels, lengths=lengths, is_training=is_training)
    rst = blstm_crf.add_blstm_crf_layer()
    return rst


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps,
                     use_one_hot_embeddings, out_params):
    """
    构建模型
    :param bert_config:
    :param num_labels:
    :param init_checkpoint:
    :param learning_rate:
    :param num_train_steps:
    :param num_warmup_steps:
    :param use_tpu:
    :param use_one_hot_embeddings:
    :return:
    """

    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        print('shape of input_ids', input_ids.shape)
        # label_mask = features["label_mask"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        # 使用参数构建模型,input_idx 就是输入的样本idx表示，label_ids 就是标签的idx表示
        (total_loss, logits, trans, pred_ids) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings, out_params)

        tvars = tf.trainable_variables()
        # 加载BERT模型
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                       init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")

        # 打印加载模型的参数
        # for var in tvars:
        #     init_string = ""
        #     if var.name in initialized_variable_names:
        #         init_string = ", *INIT_FROM_CKPT*"
        #     tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
        #                     init_string)
        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, None)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op)  # 钩子，这里用来将BERT中的参数作为我们模型的初始值
        elif mode == tf.estimator.ModeKeys.EVAL:
            # 针对NER ,进行了修改
            def metric_fn(label_ids, logits, trans):
                # 首先对结果进行维特比解码
                # crf 解码

                weight = tf.sequence_mask(out_params.max_seq_length)
                precision = tf_metrics.precision(label_ids, pred_ids, num_labels, [2, 3, 4, 5, 6, 7], weight)
                recall = tf_metrics.recall(label_ids, pred_ids, num_labels, [2, 3, 4, 5, 6, 7], weight)
                f = tf_metrics.f1(label_ids, pred_ids, num_labels, [2, 3, 4, 5, 6, 7], weight)

                return {
                    "eval_precision": precision,
                    "eval_recall": recall,
                    "eval_f": f,
                    # "eval_loss": loss,
                }

            eval_metrics = metric_fn(label_ids, logits, trans)
            # eval_metrics = (metric_fn, [label_ids, logits])
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics)  #
        else:
            predictions = {
                'pred_ids': pred_ids
            }
            export_outputs = {
                'prediction': tf.estimator.export.PredictOutput(predictions)
            }
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs=export_outputs
            )
        return output_spec

    return model_fn


def main(_):
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
    if params.max_seq_length >bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (params.max_seq_length, bert_config.max_position_embeddings))
    logger.info("clean train's output_dir")
    data_config_path = os.path.join(root_path, params.data_config_path)
    print(data_config_path)
    output_dir = os.path.join(root_path, params.output_dir)
    print(output_dir)
    if params.clean and params.do_train:
        if os.path.exists(output_dir):
            def del_file(path):
                ls = os.listdir(path)
                for i in ls:
                    c_path = os.path.join(path, i)
                    if os.path.isdir(c_path):
                        del_file(c_path)
                    else:
                        os.remove(c_path)
            try:
                del_file(output_dir)
            except Exception as e:
                print("output_dir:{}  ".format(output_dir) + traceback.format_exc())
                print('pleace remove the files of output dir and data.conf')
                exit(-1)
        if os.path.exists(data_config_path):
            try:
                os.remove(data_config_path)
            except Exception as e:
                print("data_config_path:{}  ".format(data_config_path) + traceback.format_exc())
                print('pleace remove the files of output dir and data.conf')
                exit(-1)
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
                                        tf_random_seed=19830610,
                                        save_checkpoints_steps=params.save_checkpoints_steps)
    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if os.path.exists(data_config_path):
        with codecs.open(data_config_path) as fd:
            data_config = json.load(fd)
    else:
        data_config = {}
    data_dir = os.path.join(root_path, params.data_dir)
    print("data_dir: {}".format(data_dir))
    if params.do_train:
        logger.info("load train data")
        if len(data_config) == 0:

            train_examples = processor.get_train_examples(data_dir)
            num_train_steps = int(
                len(train_examples) / params.train_batch_size * params.num_train_epochs)
            num_warmup_steps = int(num_train_steps * params.warmup_proportion)

            data_config['num_train_steps'] = num_train_steps
            data_config['num_warmup_steps'] = num_warmup_steps
            data_config['num_train_size'] = len(train_examples)
        else:
            num_train_steps = int(data_config['num_train_steps'])
            num_warmup_steps = int(data_config['num_warmup_steps'])
    init_checkpoint = os.path.join(bert_path, params.init_checkpoint)
    logger.info("achieve model_fn")
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list) + 1,
        init_checkpoint=init_checkpoint,
        learning_rate=params.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_one_hot_embeddings=False,
        out_params=params
    )
    if params.do_train:
        batch_size = params.train_batch_size
    elif params.do_eval:
        batch_size = params.eval_batch_size
    else:
        batch_size = params.predict_batch_size
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={'batch_size':batch_size}
    )
    if params.do_train:
        logger.info("convert data type to tfrecord")
        if data_config.get('train.tf_record_path', '') == '':
            train_file = os.path.join(output_dir, "train.tf_record")
            filed_based_convert_examples_to_features(
                train_examples, label_list, params.max_seq_length, tokenizer, train_file, mode=None, output_dir=output_dir)
        else:
            train_file = data_config.get('train.tf_record_path')
        num_train_size = num_train_size = int(data_config['num_train_size'])
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", num_train_size)
        logger.info("  Batch size = %d", params.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        logger.info("read train batch data")
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=params.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    if params.do_eval:
        logger.info("achieve eval data")
        if data_config.get('eval.tf_record_path', '') == '':
            eval_examples = processor.get_dev_examples(data_dir)
            eval_file = os.path.join(output_dir, "eval.tf_record")
            filed_based_convert_examples_to_features(
                eval_examples, label_list, params.max_seq_length, tokenizer, eval_file, output_dir=output_dir)
            data_config['eval.tf_record_path'] = eval_file
            data_config['num_eval_size'] = len(eval_examples)
        else:
            eval_file = data_config['eval.tf_record_path']
        num_eval_size = data_config.get('num_eval_size', 0)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", num_eval_size)
        logger.info("  Batch size = %d", params.eval_batch_size)
        eval_steps = None
        eval_drop_remainder = False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=params.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)
        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
        output_eval_file = os.path.join(output_dir, "eval_results.txt")
        with codecs.open(output_eval_file, "w", encoding='utf-8') as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
    if not os.path.exists(data_config_path):
        with codecs.open(data_config_path, 'a', encoding='utf-8') as fd:
            json.dump(data_config, fd)
    if params.do_predict:
        token_path = os.path.join(output_dir, "token_test.txt")
        if os.path.exists(token_path):
            os.remove(token_path)
        with codecs.open(os.path.join(output_dir, 'label2id.pkl'), 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}
        predict_examples = processor.get_test_examples(data_dir)
        predict_file = os.path.join(output_dir, "predict.tf_record")
        filed_based_convert_examples_to_features(predict_examples, label_list,
                                                 params.max_seq_length, tokenizer,
                                                 predict_file, mode="test",output_dir=output_dir)

        logger.info("***** Running prediction*****")
        logger.info("  Num examples = %d", len(predict_examples))
        logger.info("  Batch size = %d", params.predict_batch_size)
        predict_drop_remainder = False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=params.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder
        )

        predicted_result = estimator.evaluate(input_fn=predict_input_fn)
        output_eval_file = os.path.join(output_dir, "predicted_results.txt")
        with codecs.open(output_eval_file, "w", encoding='utf-8') as writer:
            tf.logging.info("***** Predict results *****")
            for key in sorted(predicted_result.keys()):
                tf.logging.info("  %s = %s", key, str(predicted_result[key]))
                writer.write("%s = %s\n" % (key, str(predicted_result[key])))

        result = estimator.predict(input_fn=predict_input_fn)
        output_predict_file = os.path.join(output_dir, "label_test.txt")

        def result_to_pair(writer):
            for predict_line, prediction in zip(predict_examples, result):
                idx = 0
                line = ''
                line_token = str(predict_line.text).split(' ')
                label_token = str(predict_line.label).split(' ')
                if len(line_token) != len(label_token):
                    tf.logging.info(predict_line.text)
                    tf.logging.info(predict_line.label)
                for id in prediction:
                    if id == 0:
                        continue
                    curr_labels = id2label[id]
                    if curr_labels in ['[CLS]', '[SEP]']:
                        continue
                    try:
                        line += line_token[idx] + ' ' + label_token[idx] + ' ' + curr_labels + '\n'
                    except Exception as e:
                        tf.logging.info(e)
                        tf.logging.info(predict_line.text)
                        tf.logging.info(predict_line.label)
                        line = ''
                        break
                    idx += 1
                writer.write(line + '\n')

        with codecs.open(output_predict_file, 'w', encoding='utf-8') as writer:
            result_to_pair(writer)

        eval_result = return_report(output_predict_file)
        print(eval_result)

if __name__ == '__main__':
    print(log_path)
    tf.app.run()