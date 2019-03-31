import tensorflow as tf
from typing import List
#do something in decoder output text
# def get_words_from_chars(characters_list: List[str], sequence_lengths: List[int], name='chars_conversion'):
#     with tf.name_scope(name=name):
#         def join_characters_fn(coords):
#             return tf.reduce_join(characters_list[coords[0]:coords[1]])
#
#         def coords_several_sequences():
#             end_coords = tf.cumsum(sequence_lengths)
#             start_coords = tf.concat([[0], end_coords[:-1]], axis=0)
#             coords = tf.stack([start_coords, end_coords], axis=1)
#             coords = tf.cast(coords, dtype=tf.int32)
#             return tf.map_fn(join_characters_fn, coords, dtype=tf.string)
#
#         def coords_single_sequence():
#             return tf.reduce_join(characters_list, keep_dims=True)
#
#         words = tf.cond(tf.shape(sequence_lengths)[0] > 1,
#                         true_fn=lambda: coords_several_sequences(),
#                         false_fn=lambda: coords_single_sequence())
#
#     return words

def ctc_loss(prob, labels, input_shape, alphabet, alphabet_codes, batch_size,
    n_pools=2*2, decode=True):
    # Compute seq_len from image width
    # 2x2 pooling in dimension W on layer 1 and 2 -> n-pools = 2*2
    seq_len_inputs = tf.divide([input_shape[1]]*batch_size, n_pools,
                               name='seq_len_input_op') - 1

    # Get keys (letters) and values (integer stand ins for letters)
    # Alphabet and codes
    keys = [c for c in alphabet] # the letters themselves
    values = alphabet_codes # integer representations


    # Create non-string labels from the keys and values above
    # Convert string label to code label
    with tf.name_scope('str2code_conversion'):
        table_str2int = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(keys, values), -1)
        splited = tf.string_split(labels, delimiter='')  # TODO change string split to utf8 split in next tf version
        codes = table_str2int.lookup(splited.values)
        sparse_code_target = tf.SparseTensor(splited.indices, codes, splited.dense_shape)

    seq_lengths_labels = tf.bincount(tf.cast(sparse_code_target.indices[:, 0], tf.int32),
                                     minlength=tf.shape(prob)[1])


    # Use ctc loss on probabilities from lstm output
    # Loss
    # ----
    # >>> Cannot have longer labels than predictions -> error
    with tf.control_dependencies([tf.less_equal(sparse_code_target.dense_shape[1], tf.reduce_max(tf.cast(seq_len_inputs, tf.int64)))]):
        loss_ctc = tf.nn.ctc_loss(labels=sparse_code_target,
                                  inputs=prob,
                                  sequence_length=tf.cast(seq_len_inputs, tf.int32),
                                  preprocess_collapse_repeated=False,
                                  ctc_merge_repeated=True,
                                  ignore_longer_outputs_than_inputs=True,  # returns zero gradient in case it happens -> ema loss = NaN
                                  time_major=True)
        loss_ctc = tf.reduce_mean(loss_ctc)

    if decode:
        with tf.name_scope('code2str_conversion'):
            keys = tf.cast(alphabet_codes, tf.int64)
            values = [c for c in alphabet]
            table_int2str = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(keys, values), '?')
            word_beam_search_module = tf.load_op_library('../data/TFWordBeamSearch.so')
            # prepare information about language (dictionary, characters in dataset, characters forming words)
            wordChars = open('../data/wordChars.txt').read().splitlines()[0]
            corpus = open('../data/corpus.txt').read()
            # decode using the "Words" mode of word beam search
            sparse_code_pred = word_beam_search_module.word_beam_search(prob, 50, 'NGrams',
                                                                    0.0, corpus.encode('utf8'), alphabet.encode('utf8'),
                                                                    wordChars.encode('utf8'))

    return loss_ctc, sparse_code_pred
