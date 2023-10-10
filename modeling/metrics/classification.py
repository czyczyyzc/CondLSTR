import numpy as np


class ClassificationMetric(object):
    
    def __init__(self, **kwargs):
        # Initializing counters for correct predictions and total predictions
        self.correct_predictions = 0
        self.total_predictions = 0

    def add_batch(self, pred_dict, data_dict):
        # Extracting predicted labels and ground truth labels from the dictionaries
        predicted_labels = np.asarray(pred_dict['class'])
        true_labels = data_dict['img_label'].cpu().numpy()

        # Counting how many predictions are correct
        self.correct_predictions += np.sum(predicted_labels == true_labels)
        self.total_predictions += len(true_labels)

    def reset(self):
        # Resetting the counters
        self.correct_predictions = 0
        self.total_predictions = 0
    
    def dump_state(self, state):
        state = np.array([self.correct_predictions, self.total_predictions])
        return state

    def load_state(self, state):
        self.correct_predictions, self.total_predictions = list(state[-2:])
        assert len(state[:-2]) == 0

    def __call__(self):
        # Computing the accuracy
        if self.total_predictions == 0:
            return 0.0  # To handle the case of division by zero
        accuracy = self.correct_predictions / self.total_predictions
        
        # Resetting the counters for future computations
        self.reset()
        eval_dict = dict()
        eval_dict['accuracy'] = accuracy
        eval_dict['avg'] = eval_dict['accuracy']
        return eval_dict
    