import numpy as np
import math


class FixedList(list):
    def __init__(self, size=10):
        super(FixedList, self).__init__()
        self.size = size

    def append(self, obj):
        if len(self) >= self.size:
            self.pop(0)
        super().append(obj)


class TopAverage(object):
    def __init__(self, top_k=10):
        self.scores = []
        self.top_k = top_k

    def get_top_average(self):
        if len(self.scores) > 0:
            return np.mean(self.scores)
        else:
            return 0

    def get_average(self, score):
        if len(self.scores) > 0:
            avg = np.mean(self.scores)
        else:
            avg = 0
        print("Top %d average: %f" % (self.top_k, avg))
        self.scores.append(score)
        self.scores.sort(reverse=True)
        self.scores = self.scores[:self.top_k]
        return avg

    def get_reward(self, score):
        reward = score - self.get_average(score)
        return np.clip(reward, -0.5, 0.5)

    def return_average(self):
        if len(self.scores) > 0:
            avg = np.mean(self.scores)
        else:
            avg = 0
        return avg

class EarlyStop(object):
    def __init__(self, size=10):
        self.size = size
        self.train_loss_list = FixedList(size)
        self.train_score_list = FixedList(size)
        self.val_loss_list = FixedList(size)
        self.val_score_list = FixedList(size)
        self.maxAcc_val = 0
        self.minLoss_val = 100
        self.minLoss_train = 100
        self.cur_step = 0
        self.init_list()

    def init_list(self):
        self.train_loss_list.append(0)
        self.train_score_list.append(0)
        self.val_loss_list.append(0)
        self.val_score_list.append(0)

    def should_stop(self, train_loss, train_score, val_loss, val_score):
        flag = False

        if math.isnan(train_loss) or math.isnan(val_loss):
            return True
        else:
            if val_loss < np.mean(self.val_loss_list) or val_score > np.mean(self.val_score_list):
                self.cur_step = 0
            else:
                self.cur_step += 1
        if self.cur_step >= self.size:
            flag = True
        self.train_loss_list.append(train_loss)
        self.train_score_list.append(train_score)
        self.val_loss_list.append(val_loss)
        self.val_score_list.append(val_score)
        self.maxAcc_val = max(self.maxAcc_val, val_score)
        self.minLoss_val = min(self.minLoss_val, val_loss)
        self.minLoss_train = min(self.minLoss_train, train_loss)
        return flag

    def should_save(self, train_loss, train_score, val_loss, val_score):
        # if train_loss < min(self.train_loss) and val_score > max(self.val_score_list):
        if math.isnan(train_loss) or math.isnan(val_loss):
            return False
        # elif val_loss < self.minLoss_val and val_score > self.maxAcc_val:
        # elif train_loss < self.minLoss_train and val_score > self.maxAcc_val:
        elif val_score > self.maxAcc_val:
            return True
        else:
            return False


def process_action(actions, args):
    actual_action = actions
    actual_action[-1] = args.num_class
    return actual_action


def calc_f1(output, labels, sigmoid=True):
    y_true = labels.cpu().data.numpy()
    y_pred = output.cpu().data.numpy()
    from sklearn import metrics
    if not sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")
