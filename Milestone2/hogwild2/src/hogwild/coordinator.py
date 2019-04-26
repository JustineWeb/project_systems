import json
import multiprocessing
import random
from multiprocessing.sharedctypes import Array
from multiprocessing.sharedctypes import Array
from datetime import datetime
from hogwild import hogwild_pb2, hogwild_pb2_grpc, ingest_data
from hogwild import settings as s
from hogwild.EarlyStopping import EarlyStopping
from hogwild.svm import SVM
from hogwild.utils import calculate_accs
from time import time

def calculate_svm_update(data,svm=svm):
    # Select random subset
    subset_indices = random.sample(range(len(targets_train)), s.subset_size)
    data_train,targets_train = data
    data_stoc = [data_train[x] for x in subset_indices]
    targets_stoc = [targets_train[x] for x in subset_indices]
    # Calculate weight updates
    train_loss = svm.fit(data_stoc, targets_stoc, s.synchronous)


if __name__ == '__main__':
    # Step 1: Calculate the indices reserved for the validation set
    dataset_size = sum(1 for line in open(s.TRAIN_FILE))
    val_indices = random.sample(range(dataset_size), int(s.validation_split * dataset_size))
    print('Data set size: {}, Train set size: {}, Validation set size: {}'.format(dataset_size,
                                                                                  dataset_size - len(val_indices),
                                                                                  len(val_indices)))
    # Step 2: create the coordinator SVM process
    print('Loading training data')
    data, targets = ingest_data.load_large_reuters_data(s.TRAIN_FILE,
                                                        s.TOPICS_FILE,
                                                        s.TEST_FILES,
                                                        selected_cat='CCAT',
                                                        train=True)
    data_train = [data[x] for x in range(len(targets)) if x not in val_indices]
    targets_train = [targets[x] for x in range(len(targets)) if x not in val_indices]
    data_val = [data[x] for x in val_indices]
    targets_val = [targets[x] for x in val_indices]
    print('Number of train datapoints: {}'.format(len(targets_train)))
    print('Number of validation datapoints: {}'.format(len(targets_val)))

    global w
    if s.lock_free:
        w = Array('d',[0.0]*dim,lock=False)
    else:
        w = Array('d',[0.0]*dim,lock=True)

    dim = max([max(k) for k in data]) + 1
    svm = SVM(learning_rate=s.learning_rate, lambda_reg=s.lambda_reg, dim=dim,w)

    # Step 3: depending on the chosen mode, create the pool of workers and the shared array
    pool = multiprocessing.Pool(processes= s.N_WORKERS)

    # Early stopping
    early_stopping = EarlyStopping(s.persistence)
    stopping_crit_reached = False

    n_epochs = 0
    start_time = time()
    losses_val = []
    while n_epochs < s.epochs:
        
        # update weights vector
        pool.map(calculate_svm_update,(data_train,targets_train))

        # calculate validation loss
        val_loss = svm.loss(data_val,targets_val)
        losses_val.append({'time': datetime.utcfromtimestamp(time()).strftime("%Y-%m-%d %H:%M:%S.%f"),'loss_val': val_loss})
        print('Val loss: {:.4f}'.format(val_loss))

        # check for early stopping

        end_time = time()
        print('All SGD epochs done!')

        ### Calculating final accuracies on train, validation and test sets ###
        data_test, targets_test = ingest_data.load_large_reuters_data(s.TRAIN_FILE,
                                                                      s.TOPICS_FILE,
                                                                      s.TEST_FILES,
                                                                      selected_cat='CCAT',
                                                                      train=False)

        # Calculate the predictions on the validation set
        task_queue.put({'type': 'predict', 'values': data_test})
        prediction = response_queue.get()

        a = sum([1 for x in zip(targets_test, prediction) if x[0] == 1 and x[1] == 1])
        b = sum([1 for x in targets_test if x == 1])
        print('Val accuracy of Label 1: {:.2f}%'.format(a / b))

        # Load the train dataset
        print('Loading train and validation sets to calculate final accuracies')
        data, targets = ingest_data.load_large_reuters_data(s.TRAIN_FILE,
                                                            s.TOPICS_FILE,
                                                            s.TEST_FILES,
                                                            selected_cat='CCAT',
                                                            train=True)
        data_train, targets_train, data_val, targets_val = ingest_data.train_val_split(data, targets, val_indices)

        # Calculate the predictions on the train set
        task_queue.put({'type': 'predict', 'values': data_train})
        preds_train = response_queue.get()
        acc_pos_train, acc_neg_train, acc_tot_train = calculate_accs(targets_train, preds_train)
        print('Train accuracy of Label 1: {:.2f}%'.format(acc_pos_train))
        print('Train accuracy of Label -1: {:.2f}%'.format(acc_neg_train))
        print('Train accuracy: {:.2f}%'.format(acc_tot_train))

        # Calculate the predictions on the validation set
        task_queue.put({'type': 'predict', 'values': data_val})
        preds_val = response_queue.get()
        acc_pos_val, acc_neg_val, acc_tot_val = calculate_accs(targets_val, preds_val)
        print('Val accuracy of Label 1: {:.2f}%'.format(acc_pos_val))
        print('Val accuracy of Label -1: {:.2f}%'.format(acc_neg_val))
        print('Val accuracy: {:.2f}%'.format(acc_tot_val))

        # Load the test dataset
        print('Loading test set')
        data_test, targets_test = ingest_data.load_large_reuters_data(s.TRAIN_FILE,
                                                                      s.TOPICS_FILE,
                                                                      s.TEST_FILES,
                                                                      selected_cat='CCAT',
                                                                      train=False)

        # Calculate the predictions on the test set
        task_queue.put({'type': 'predict', 'values': data_test})
        preds_test = response_queue.get()
        acc_pos_test, acc_neg_test, acc_tot_test = calculate_accs(targets_test, preds_test)
        print('Test accuracy of Label 1: {:.2f}%'.format(acc_pos_test))
        print('Test accuracy of Label -1: {:.2f}%'.format(acc_neg_test))
        print('Test accuracy: {:.2f}%'.format(acc_tot_test))

        # Save results in a log
        log = [{'start_time': datetime.utcfromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S.%f"),
                'end_time': datetime.utcfromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S.%f"),
                'running_time': end_time - start_time,
                'n_workers': s.N_WORKERS,
                'running_mode': s.running_mode,
                'sync_epochs': n_epochs - 1,
                'accuracy_train': acc_tot_train,
                'accuracy_1_train': acc_pos_train,
                'accuracy_-1_train': acc_neg_train,
                'accuracy_val': acc_tot_val,
                'accuracy_1_val': acc_pos_val,
                'accuracy_-1_val': acc_neg_val,
                'accuracy_test': acc_tot_test,
                'accuracy_1_test': acc_pos_test,
                'accuracy_-1_test': acc_neg_test,
                'losses_val': losses_val,
                'losses_train': hws.train_losses}]

        with open('log.json', 'w') as outfile:
            json.dump(log, outfile)

        # Send poison pill to SVM process
        task_queue.put(None)

        # Close queues and join processes
        task_queue.close()
        task_queue.join_thread()
        response_queue.close()
        response_queue.join_thread()
        svm_proc.join()

    except KeyboardInterrupt:
        server.stop(0)

