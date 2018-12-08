import pandas as pd
import os
import codecs
import tensorflow as tf
import multiprocessing
from tensorflow import data



HEADER_DEFAULTS = [["NA"], ["NA"]]
TARGET_NAME = 'labels'
HEADER = ["instances", "labels"]
TARGET_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]
num_labels = len(TARGET_LABELS)
MULTI_THREADING = True
PAD_WORD = "[PAD]"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 2
VOCAB_FILE = "/root/PycharmProjects/BERT-BiLSTM-CRF-NER/chinese_L-12_H-768_A-12/vocab.txt"
path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_path = os.path.join(path, "data")


def save_csv(list_dict, filename):
    df = pd.DataFrame(list_dict)
    df.to_csv(filename, sep='\t')

def read_csv(filename, header=0, index_col=0):
    data = pd.read_csv(filename, header=header, index_col=index_col, skiprows=0)
    return data


class Process(object):

    def get_train_file(self, data_dir):
        list_dict = self.convert_txt_to_csv(os.path.join(data_dir, 'train.txt'))
        save_csv(list_dict, os.path.join(data_dir, 'train.csv'))

    def get_eval_file(self, data_dir):
        list_dict = self.convert_txt_to_csv(os.path.join(data_dir, 'dev.txt'))
        save_csv(list_dict, os.path.join(data_dir, 'dev.csv'))

    def get_test_file(self, data_dir):
        list_dict = self.convert_txt_to_csv(os.path.join(data_dir, 'test.txt'))
        save_csv(list_dict, os.path.join(data_dir, 'test.csv'))

    @classmethod
    def convert_txt_to_csv(cls, input_file):
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            lines = []
            words = []
            labels = []
            words.append('[CLS]')
            labels.append('[CLS]')
            for line in f:
                contends = line.strip()
                tokens = contends.split(' ')
                if len(tokens) == 2:
                    word = line.strip().split(' ')[0]
                    label = line.strip().split(' ')[-1]
                else:
                    if len(contends) == 0:
                        words.append('[SEP]')
                        labels.append('[SEP]')
                        l = ' '.join([label for label in labels if len(label) > 0])
                        w = ' '.join([word for word in words if len(word) > 0])
                        dict = {}
                        dict['instances'] = w
                        dict['labels'] = l
                        lines.append(dict)
                        words = []
                        labels = []
                        words.append('[CLS]')
                        labels.append('[CLS]')
                        continue
                if contends.startswith("-DOCSTART-"):
                    words.append('')
                    continue
                words.append(word)
                labels.append(label)
            return lines


def parse_tsv_row(tsv_row):
    columns = tf.decode_csv(tsv_row, record_defaults=HEADER_DEFAULTS, field_delim='\t', select_cols=[1, 2])
    features = dict(zip(HEADER, columns))
    target = features.pop(TARGET_NAME)
    return features, target


def parse_label_column(label_string_tensor):
    table = tf.contrib.lookup.index_table_from_tensor(tf.constant(TARGET_LABELS))
    return table.lookup(label_string_tensor)


def input_fn(files_name, mode=tf.estimator.ModeKeys.EVAL,
             skip_header_lines=1,
             num_epochs=1,
             batch_size=32):
    shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False
    num_threads = multiprocessing.cpu_count() if MULTI_THREADING else 1
    buffer_size = 2 * batch_size + 1

    print("")
    print("* data input_fn:")
    print("================")
    print("Input file(s): {}".format(files_name))
    print("Batch size: {}".format(batch_size))
    print("Epoch Count: {}".format(num_epochs))
    print("Mode: {}".format(mode))
    print("Thread Count: {}".format(num_threads))
    print("Shuffle: {}".format(shuffle))
    print("================")
    print("")

    dataset = data.TextLineDataset(filenames=files_name)
    dataset = dataset.skip(skip_header_lines)

    if shuffle:
        dataset = dataset.shuffle(buffer_size)

    # dataset = dataset.map(lambda tsv_row: parse_tsv_row(tsv_row),
    #                       num_parallel_calls=num_threads)
    # dataset = dataset.batch(batch_size)
    dataset = dataset.apply(tf.contrib.data.map_and_batch(
        lambda tsv_row: parse_tsv_row(tsv_row),
        batch_size=batch_size,
        drop_remainder=False
    ))
    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size)

    iterator = dataset.make_one_shot_iterator()
    features, target = iterator.get_next()
    print("===================================")
    return features, process_label(target)


def process_text(text_feature):

    vocab_table = tf.contrib.lookup.index_table_from_file(vocabulary_file=VOCAB_FILE,
                                                            num_oov_buckets=0,
                                                            default_value=0)
    smss = text_feature['instances']
    words = tf.string_split(smss)
    dense_words = tf.sparse_tensor_to_dense(words, default_value=PAD_WORD)
    word_ids = vocab_table.lookup(dense_words)
    padding = tf.constant([[0, 0], [0, MAX_SEQ_LENGTH]])
    word_ids_padded = tf.pad(word_ids, padding)
    word_id_vector = tf.slice(word_ids_padded, [0, 0],
                              [-1, MAX_SEQ_LENGTH])
    zeros = tf.zeros_like(word_id_vector)
    word_mask = tf.greater(word_id_vector, zeros)
    word_mask = tf.cast(word_mask, dtype=tf.int32)

    segment_id = tf.zeros_like(word_id_vector)
    word_id_vector = tf.cast(word_id_vector, dtype=tf.int32)
    segment_id = tf.cast(segment_id, dtype=tf.int32)

    return word_id_vector, word_mask, segment_id


def process_label(labels):
    vocab_table = tf.contrib.lookup.index_table_from_tensor(mapping=tf.constant(TARGET_LABELS))
    words = tf.string_split(labels)
    dense_words = tf.sparse_tensor_to_dense(words, default_value="X")
    word_ids = vocab_table.lookup(dense_words)
    padding = tf.constant([[0, 0], [0, MAX_SEQ_LENGTH]])
    word_ids_padded = tf.pad(word_ids, padding)
    label_id_vector = tf.slice(word_ids_padded, [0, 0], [-1, MAX_SEQ_LENGTH])
    return label_id_vector




if __name__ == '__main__':
    pass


    #================测试process_text这个函数===================
    # sess = tf.Session()
    # features, labels = input_fn(os.path.join(data_path, 'train.csv'), batch_size=BATCH_SIZE)
    # inputs_id, inputs_mask, segment_id = process_text(features, sess)
    # table_init = tf.tables_initializer()
    # sess.run(table_init)
    #
    # res = sess.run([inputs_id, inputs_mask, segment_id])
    # print(res[0])
    # print(res[1])
    # print(res[2])





    #=================测试process_label这个函数================
    # sess = tf.Session()
    # features, labels = input_fn(os.path.join(data_path, 'train.csv'), batch_size=BATCH_SIZE)
    # labels_id = process_label(labels)
    # table_init = tf.tables_initializer()
    # sess.run(table_init)
    # res = sess.run(labels_id)
    # print(res.tolist()[0:2])

    #==================将txt文本转化为csv==============
    # process = Process()
    # process.get_train_file(data_path)
    # process.get_eval_file(data_path)
    # process.get_test_file(data_path)
    # inp = [{'c1': 10, 'c2': 100}, {'c1': 11, 'c2': 110}, {'c1': 12, 'c2': 120}]
    # save_csv(inp, 'tmp.csv')