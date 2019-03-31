import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import time
import editdistance


#############################################################################################################
# run epochs
def run_epochs(saver,
               restore_model_nm,
               n_epochs_per_bunch,
               iterator,
               n_batches,
               next_batch,
               train_op,
               loss_ctc,
               input_tensor,
               labels,
               trg,
               data_batch,
               data_image,
               output_model_dir,
               oldnew,
               pred,
               sparse_code_pred,
               alphabet):
    with tf.Session() as sess:
        # initialize variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        # restore model if it exists
        if restore_model_nm != "":
            saver.restore(sess, restore_model_nm)

        start_time = time.time() # time everything
        for i in range(n_epochs_per_bunch):
            sess.run(iterator.initializer) # need to initialize iterator for each epoch
            print("---------------------------------------------------------")
            print("Starting epoch", i)
            for j in range(0, n_batches):
                err = False
                input_tensor_b, labels_b, filenames_b = sess.run(next_batch)
                ground_truth = []
                for i in range(labels_b.shape[0]):
                    ground_truth.append(labels_b[i].decode("utf-8"))
                if train_op is not None:# do training
                    _, decoder, loss = sess.run([train_op, sparse_code_pred, loss_ctc],
                                                        feed_dict={input_tensor: input_tensor_b, labels: labels_b})

                else:# do prediction
                    decoder, loss = sess.run([sparse_code_pred, loss_ctc],
                                                    feed_dict={input_tensor: input_tensor_b, labels: labels_b})
                words = decoderOutputToText(decoder, ground_truth.shape[0], alphabet)
                #Calculate CER and wordAccuracy
                out = getCERandAccuracy(words, ground_truth)
                cer, acc = out
                try:
                    #create output for new row in data frames
                    new_bat = {"tr_group":trg,
                               "oldnew":oldnew,
                               "pred":pred,
                               "epoch":i,
                               "batch":j,
                               "loss":loss,
                               "cer":cer,
                               "accuracy":[[acc]],
                               "time":time.time()-start_time}
                    new_img = {"tr_group":[trg for _ in range(len(labels_b))],
                               "oldnew":[oldnew for _ in range(len(labels_b))],
                               "pred":[pred for _ in range(len(labels_b))],
                               "epoch":[i for _ in range(len(labels_b))],
                               "batch":[j for _ in range(len(labels_b))],
                               "labels":[str(ddd, "utf-8") for ddd in labels_b],
                               "words":[str(ddd, "utf-8") for ddd in words],
                               "filenames":[str(ddd, "utf-8") for ddd in filenames_b]}
                    tim = (time.time()-start_time)/(i*n_batches + j + 1)
                    print('batch: {0}:{1}:{2}, time per batch: {5}\n\tloss: {3}, CER: {4}'.format(trg, i, j, loss, cer, tim), flush=True)
                except: # deal with CTC loss errors that occur semi-regularly
                    #create dummy output for new row in data frames
                    new_bat = {"tr_group":trg,
                               "oldnew":oldnew,
                               "pred":pred,
                               "epoch":i,
                               "batch":j,
                               "loss":-1,
                               "cer":-1,
                               "accuracy":[[-1, -1]],
                               "time":time.time()-start_time}
                    new_img = {"tr_group":[trg for _ in range(len(labels_b))],
                               "oldnew":[oldnew for _ in range(len(labels_b))],
                               "pred":[pred for _ in range(len(labels_b))],
                               "epoch":[i for _ in range(len(labels_b))],
                               "batch":[j for _ in range(len(labels_b))],
                               "labels":[str(ddd, "utf-8") for ddd in labels_b],
                               "words":["" for _ in range(len(labels_b))],
                               "filenames":[str(ddd, "utf-8") for ddd in filenames_b]}
                    print("Error at {0}:{1}:{2}".format(trg, i, j), flush=True)
                    err = True
                #save data
                new_bat = pd.DataFrame.from_dict(new_bat)
                new_img = pd.DataFrame.from_dict(new_img)
                data_batch = data_batch.append(new_bat)
                data_image = data_image.append(new_img)
                data_batch.to_csv(output_model_dir+"metrics_batch" + str(trg) + ".csv", index=False)
                data_image.to_csv(output_model_dir+"metrics_image" + str(trg) + ".csv", index=False)
                saver.save(sess, output_model_dir+"model" + str(trg) + ".ckpt")
                # uncomment this line if you want to break each epoch after the first successful batch
                #if not err: break
            print('Avg Epoch time: {0} seconds'.format((time.time() - start_time)/(1.0*(i+1))))
    return data_batch, data_image

def decoderOutputToText(ctcOutput, batchSize, alphabet):
    encodedLabelStrs = [[] for i in range(batchSize)]
    blank = len(alphabet)
    for b in range(batchSize):
        for label in ctcOutput[b]:
            if label==blank:
                break
            encodedLabelStrs[b].append(label)
    return [str().join([alphabet[c] for c in labelStr]) for labelStr in encodedLabelStrs]

def getCERandAccuracy(words, ground_truth):
    numCharErr = 0
    numCharTotal = 0
    numWordOK = 0
    numWordTotal = 0
    print('Ground truth -> Recognized')
    for i in range(len(words)):
        numWordOK += 1 if ground_truth[i] == words[i] else 0
        numWordTotal += 1
        dist = editdistance.eval(words[i], ground_truth[i])
        numCharErr += dist
        numCharTotal += len(ground_truth[i])
    CER = numCharErr / numCharTotal
    wordAccuracy = numWordOK / numWordTotal
    return CER, wordAccuracy
