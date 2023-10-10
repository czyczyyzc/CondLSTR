import numpy as np


class SemanticSegMetric(object):

    def __init__(self, num_classes=4, **kwargs):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((self.num_classes,) * 2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        mIoU = np.nanmean(MIoU[1:])
        MIoU[0] = mIoU
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_classes)
        label = self.num_classes * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_classes**2)
        confusion_matrix = count.reshape(self.num_classes, self.num_classes)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        return
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes,) * 2)

    def dump_state(self, state):
        state = np.concatenate([state, self.confusion_matrix.reshape(-1)], axis=0)
        return state

    def load_state(self, state):
        position = self.confusion_matrix.reshape(-1).shape[0]
        self.confusion_matrix = state[-position:].reshape(self.confusion_matrix.shape)
        return state[:-position]

    def __call__(self):
        MIoU = self.Mean_Intersection_over_Union()
        self.reset()
        eval_dict = {'MIoU': MIoU}
        eval_dict['avg'] = eval_dict['MIoU']
        return eval_dict
