
def main():
    extra_args = {'target':[str,'Name of parameter to be predicted.'],
    'task':[str,'Choose between regression and classification.'],
    'train_in':[str,'Filename of input data for training.'],
    'train_out':[str,'Filename of output target for training.'],
    'train_list':[str,'File containing list of training samples.'],
    'train_pred':[str,'File to store predicted values from training.'],
    'train_summ':[str,'File to store ML performance metrics from train.'],
    'test_in':[str,'Filename of input data for testing.'],
    'test_out':[str,'Filename of output target for testing.'],
    'test_list':[str,'File containing list of testing samples.'],
    'test_pred':[str,'File to store predicted values from testing.'],
    'test_summ':[str,'File to store ML performance metrics from test.'],
    'lr':[float,'Learning rate.'],
    'batch':[int,'Batch size.'],
    'epochs':[int,'Number of epochs.']}

    ARGS = get_args(extra_args)

