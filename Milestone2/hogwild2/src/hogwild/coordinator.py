import json
import multiprocessing
import random
from multiprocessing.sharedctypes import Array
from datetime import datetime
import settings as s
from EarlyStopping import EarlyStopping
from time import time
import ingest_data
from utils import dotproduct, sign, calculate_accs

def calculate_svm_update(w, data_train,targets_train):
    # Select random subset
	subset_indices = random.sample(range(len(targets_train)), s.subset_size)
	data_stoc = [data_train[x] for x in subset_indices]
	targets_stoc = [targets_train[x] for x in subset_indices]
	# Calculate weight updates
	train_loss = fit(data_stoc, targets_stoc,w)

def fit(data, labels,w):
	'''
	Calculates the gradient and train loss.
	'''
	train_loss = 0

	for x, label in zip(data, labels):
		xw = dotproduct(x,w)
		if misclassification(xw, label):
			delta_w = gradient(x, label,w)
		else:
			delta_w = regularization_gradient(x,w)
		for k, v in delta_w.items():
			w[k] += s.learning_rate * v
		train_loss += max(1 - label * xw, 0)
		train_loss += regularizer(x,w)
	return train_loss / len(labels)

def misclassification(x_dot_w, label):
	''' Returns true if x is misclassified. '''
	return x_dot_w * label < 1

def gradient(x, label,w):
	''' Returns the gradient of the loss with respect to the weights '''
	regularizer = regularizer_g(x,w)
	return {k: (v * label - regularizer) for k, v in x.items()}

def regularizer_g(x,w):
	'''Returns the gradient of the regularization term  '''
	return 2 * s.lambda_reg * sum([w[i] for i in x.keys()]) / len(x)

def regularization_gradient(x,w):
	''' Returns the gradient of the regularization term for each datapoint '''
	regularizer = regularizer_g(x,w)
	return {k: regularizer for k in x.keys()}

def regularizer(x,w):
	''' Returns the regularization term '''
	return s.lambda_reg* sum([w[i] ** 2 for i in x.keys()]) / len(x)

def loss(data, labels,w):
	''' Returns the MSE loss of the data with the true labels. '''
	total_loss = 0
	for x, label in zip(data, labels):
		xw = dotproduct(x, w)
		total_loss += max(1 - label * xw, 0)
		total_loss += regularizer(x,w)
	return total_loss / len(labels)

def predict(data,w):
	''' Predict the labels of the input data '''
	return [sign(dotproduct(x,w)) for x in data]

if __name__ == '__main__':
	# Step 1: Calculate the indices reserved for the validation set
	dataset_size = sum(1 for line in open(s.TRAIN_FILE))
	val_indices = random.sample(range(dataset_size), int(s.validation_split * dataset_size))
	print('Data set size: {}, Train set size: {}, Validation set size: {}'.format(dataset_size,
	                                                                              dataset_size - len(val_indices),
	                                                                              len(val_indices)))
	# Step 2: create the coordinator SVM process
	print('Loading training data')
	data, targets = ingest_data.load_large_reuters_data(s.TRAIN_FILE,s.TOPICS_FILE,s.TEST_FILES,selected_cat='CCAT', train= True)
	data_train = [data[x] for x in range(len(targets)) if x not in val_indices]
	targets_train = [targets[x] for x in range(len(targets)) if x not in val_indices]
	data_val = [data[x] for x in val_indices]
	targets_val = [targets[x] for x in val_indices]
	print('Number of train datapoints: {}'.format(len(targets_train)))
	print('Number of validation datapoints: {}'.format(len(targets_val)))

	dim = max([max(k) for k in data]) + 1
	if s.lock_free:
		w = Array('d',[0.0]*dim,lock=False)
	else:
		w = Array('d',[0.0]*dim,lock=True)


	# Early stopping
	early_stopping = EarlyStopping(s.persistence)
	stopping_crit_reached = False

	# Step 3: training using the pool of threads
	n_epochs = 0
	start_time = time()
	losses_val = []

	while n_epochs < s.epochs:
		workers = [multiprocessing.Process(target=calculate_svm_update, args = (w,data_train,targets_train)) for i in range(s.N_WORKERS)]
		# update weights vector
		for p in workers:
			p.start()
		for p in workers:
			p.join()
		for p in workers:
			p.terminate()
		n_epochs += 1

		# calculate validation loss
		val_loss = loss(data_val,targets_val,w)
		losses_val.append({'time': datetime.utcfromtimestamp(time()).strftime("%Y-%m-%d %H:%M:%S.%f"),'loss_val': val_loss})
		print('Val loss: {:.4f}'.format(val_loss))

		# check for early stopping
		stopping_crit_reached = early_stopping.stopping_criterion(val_loss)
		if stopping_crit_reached:
			break

		end_time = time()
	print('All SGD epochs done!')
	print("final time is: {}".format(end_time))
	print("time per iteration: {}".format(end_time/n_epochs))

    ### Calculating final accuracies on train, validation and test sets ###
	data_test, targets_test = ingest_data.load_large_reuters_data(s.TRAIN_FILE,
																s.TOPICS_FILE,
																s.TEST_FILES,
																selected_cat='CCAT',
																train=False)

	# Calculate the predictions on the validation set
	prediction = predict(data_test,w)

	a = sum([1 for x in zip(targets_test, prediction) if x[0] == 1 and x[1] == 1])
	b = sum([1 for x in targets_test if x == 1])
	print('Val accuracy of Label 1: {:.2f}%'.format(a / b))

	# load the train dataset
	print('Loading train and validation sets to calculate final accuracies')
	data, targets = ingest_data.load_large_reuters_data(s.TRAIN_FILE,
																s.TOPICS_FILE,
																s.TEST_FILES,
																selected_cat='CCAT',
																train=True)
	data_train, targets_train, data_val, targets_val = ingest_data.train_val_split(data, targets, val_indices)
	# Calculate the predictions on the train set
	preds_train = predict(data_train,w)
	acc_pos_train, acc_neg_train, acc_tot_train = calculate_accs(targets_train, preds_train)
	print('Train accuracy of Label 1: {:.2f}%'.format(acc_pos_train))
	print('Train accuracy of Label -1: {:.2f}%'.format(acc_neg_train))
	print('Train accuracy: {:.2f}%'.format(acc_tot_train))

	# Calculate the predictions on the validation set
	pred_vals = predict(data_val,w)
	acc_pos_val, acc_neg_val, acc_tot_val = calculate_accs(targets_val, pred_vals)
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
	preds_test = predict(data_test, w)
	acc_pos_test, acc_neg_test, acc_tot_test = calculate_accs(targets_test, preds_test)
	print('Test accuracy of Label 1: {:.2f}%'.format(acc_pos_test))
	print('Test accuracy of Label -1: {:.2f}%'.format(acc_neg_test))
	print('Test accuracy: {:.2f}%'.format(acc_tot_test))

	# Save results in a log
	log = [{'start_time': datetime.utcfromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S.%f"),
			'end_time': datetime.utcfromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S.%f"),
			'running_time': end_time - start_time,
			'n_workers': s.N_WORKERS,
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
			'losses_val': losses_val,}]

	with open('log.json', 'w') as outfile:
		json.dump(log, outfile)