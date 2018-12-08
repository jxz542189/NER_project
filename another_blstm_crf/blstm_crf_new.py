from another_blstm_crf.utils.covert import process_label, process_text, Process, input_fn
import tensorflow as tf
#from blstm_crf.bert_blstm_crf import create_model
from bert.bert import modeling, optimization, tokenization
import os
from blstm_crf.utils import tf_metrics
import shutil
from datetime import datetime
import json
from blstm_crf.model.lstm_crf_layer import BLSTM_CRF
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib import crf


os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(path, "data")
TARGET_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]
num_labels = len(TARGET_LABELS)
VOCAB_FILE = "/root/PycharmProjects/BERT-BiLSTM-CRF-NER/chinese_L-12_H-768_A-12/vocab.txt"
BERT_PATH = "/root/PycharmProjects/BERT-BiLSTM-CRF-NER/chinese_L-12_H-768_A-12"
BERT_CONFIG_FILE = "bert_config.json"
INIT_CHECKPOINT = "bert_model.ckpt"
INIT_CHECKPOINT = os.path.join(BERT_PATH, INIT_CHECKPOINT)
LEARNING_RATE = 0.01
train_examples = 20863
test_examples = 4637
eval_examples = 2317
BATCH_SIZE = 2
NUM_EPOCHS = 3
WARMUP_PROPORTION = 0.1
num_train_steps = int(
    train_examples / BATCH_SIZE * NUM_EPOCHS)
num_eval_steps = int(eval_examples / BATCH_SIZE)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)
MAX_SEQ_LENGTH = 128
TOTAL_STEPS = int((train_examples/BATCH_SIZE) * NUM_EPOCHS)
EVAL_AFTER_SEC = 60
MODEL_NAME = 'blstm_crf'
model_dir = 'trained_models/{}'.format(MODEL_NAME)
RESUME_TRAINING = False
LOG_STEP_COUNT_STEPS = 5000
TF_RANDOM_SEED = 19830610
params_path = "/root/PycharmProjects/NER_project/another_blstm_crf/config/params.json"


def model_fn(features, labels, mode, params):
    input_ids, input_mask, segment_ids = process_text(features)
    label_ids = labels
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    bert_config_file = os.path.join(BERT_PATH, BERT_CONFIG_FILE)
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    with open(params_path) as param:
        params_dict = json.load(param)
    config = tf.contrib.training.HParams(**params_dict)
    # (total_loss, logits, trans, pred_ids) = create_model(
    #     bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
    #     num_labels, use_one_hot_embeddings=False, params=config)
    model = modeling.BertModel(config=bert_config,
                               is_training=is_training,
                               input_ids=input_ids,
                               input_mask=input_mask,
                               token_type_ids=segment_ids,
                               use_one_hot_embeddings=False)
    embedding = model.get_sequence_output()
    max_seq_length = embedding.shape[1].value

    used = tf.sign(tf.abs(input_ids))
    lengths = tf.reduce_sum(used, reduction_indices=1)

    blstm_crf = BLSTM_CRF(embedded_chars=embedding, hidden_unit=config.lstm_size, cell_type=config.cell,
                          num_layers=config.num_layers,
                          droupout_rate=config.dropout_rate, initializers=initializers, num_labels=num_labels,
                          seq_length=max_seq_length, labels=labels, lengths=lengths, is_training=is_training)
    lstm_output = blstm_crf.blstm_layer(blstm_crf.embedded_chars)
    logits = blstm_crf.project_bilstm_layer(lstm_output)
    tvars = tf.trainable_variables()
    if INIT_CHECKPOINT:
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                   INIT_CHECKPOINT)
        tf.train.init_from_checkpoint(INIT_CHECKPOINT, assignment_map)
    if mode == tf.estimator.ModeKeys.TRAIN:
        total_loss, trans = blstm_crf.crf_layer(logits)
        pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=trans, sequence_length=blstm_crf.lengths)
        # (total_loss, logits, trans, pred_ids) = blstm_crf.add_blstm_crf_layer()
        train_op = optimization.create_optimizer(
            total_loss, LEARNING_RATE, num_train_steps, num_warmup_steps, None)
        output_spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=total_loss,
            train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:
        def metric_fn(label_ids, pred_ids, trans):

            weight = tf.sequence_mask(MAX_SEQ_LENGTH)
            precision = tf_metrics.precision(label_ids, pred_ids, num_labels, [1, 2, 3, 4, 5, 6], weight)
            recall = tf_metrics.recall(label_ids, pred_ids, num_labels, [1, 2, 3, 4, 5, 6], weight)
            f = tf_metrics.f1(label_ids, pred_ids, num_labels, [1, 2, 3, 4, 5, 6], weight)

            return {
                "eval_precision": precision,
                "eval_recall": recall,
                "eval_f": f
            }

        total_loss, trans = blstm_crf.crf_layer(logits)
        with tf.variable_scope("crf_loss", reuse=tf.AUTO_REUSE):
            trans = tf.get_variable(
                "transitions",
                shape=[blstm_crf.num_labels, blstm_crf.num_labels],
                initializer=blstm_crf.initializers.xavier_initializer())
        pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=trans, sequence_length=blstm_crf.lengths)
        eval_metrics = metric_fn(label_ids, pred_ids, trans)
        output_spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=total_loss,
            eval_metric_ops=eval_metrics)
    else:
        with tf.variable_scope("crf_loss", reuse=tf.AUTO_REUSE):
            trans = tf.get_variable(
                "transitions",
                shape=[blstm_crf.num_labels, blstm_crf.num_labels],
                initializer=blstm_crf.initializers.xavier_initializer())
        pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=trans, sequence_length=blstm_crf.lengths)
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


def create_estimator(run_config, hparams=None):
    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       params=hparams,
                                       config=run_config)
    print("")
    print("Estimator Type: {}".format(type(estimator)))
    print("")

    return estimator


def serving_input_fn():
    receiver_tensor = {
        'instances': tf.placeholder(tf.string, [None])

    }
    features = {
        key:tensor for key, tensor  in receiver_tensor.items()
    }
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)


if __name__ == '__main__':
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(os.path.join(data_path, 'train.csv'),
                                  mode=tf.estimator.ModeKeys.TRAIN,
                                  num_epochs=NUM_EPOCHS,
                                  batch_size=BATCH_SIZE),
        max_steps=TOTAL_STEPS,
        hooks=None
    )
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn(os.path.join(data_path, 'test.csv'),
                                  mode=tf.estimator.ModeKeys.EVAL,
                                  batch_size=BATCH_SIZE//2),
        exporters=[tf.estimator.LatestExporter("predict",
                                               serving_input_receiver_fn=serving_input_fn,
                                               as_text=True,
                                               exports_to_keep=1)],
        steps=num_eval_steps,
        throttle_secs=EVAL_AFTER_SEC
    )
    if not RESUME_TRAINING:
        print("Removing previous artifacts...")
        shutil.rmtree(model_dir, ignore_errors=True)
    else:
        print("Resuming training...")

    time_start = datetime.utcnow()
    print("Experiment started at {}".format(time_start.strftime("%H:%M:%S")))
    run_config = tf.estimator.RunConfig(log_step_count_steps=LOG_STEP_COUNT_STEPS,
                                        tf_random_seed=TF_RANDOM_SEED,
                                        model_dir=model_dir,
                                        save_checkpoints_steps=1000)
    estimator = create_estimator(run_config)

    tf.estimator.train_and_evaluate(estimator=estimator,
                                    train_spec=train_spec,
                                    eval_spec=eval_spec)
    time_end = datetime.utcnow()
    print(".......................................")
    print("Experiment finished at {}".format(time_end.strftime("%H:%M:%S")))
    print("")
    time_elapsed = time_end - time_start
    print("Experiment elapsed time: {} seconds".format(time_elapsed.total_seconds()))