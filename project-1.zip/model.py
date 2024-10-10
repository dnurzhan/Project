import numpy as np
import re

class Model:
    def __init__(self, alpha=1):
        self.vocab = set() # словарь, содержащий все уникальные слова из набора train
        self.spam = {} # словарь, содержащий частоту слов в спам-сообщениях из набора данных train.
        self.ham = {} # словарь, содержащий частоту слов в не спам-сообщениях из набора данных train.
        self.alpha = alpha # сглаживание
        self.label2num = None # словарь, используемый для преобразования меток в числа
        self.num2label = None # словарь, используемый для преобразования числа в метки
        self.Nvoc = None # общее количество уникальных слов в наборе данных train
        self.Nspam = None # общее количество уникальных слов в спам-сообщениях в наборе данных train
        self.Nham = None # общее количество уникальных слов в не спам-сообщениях в наборе данных train
        self._train_X, self._train_y = None, None
        self._val_X, self._val_y = None, None
        self._test_X, self._test_y = None, None

    def fit(self, dataset):
        '''
        dataset - объект класса Dataset
        Функция использует входной аргумент "dataset", 
        чтобы заполнить все атрибуты данного класса.
        '''
        # Начало вашего кода
        self.label2num = dataset.label2num
        self.num2label = dataset.num2label
        self._train_X, self._train_y = dataset.train[0], dataset.train[1]
        self._val_X, self._val_y = dataset.val[0], dataset.val[1]
        self._test_X, self._test_y = dataset.test[0], dataset.test[1]

        self.vocab = set()
        self.spam = {}
        self.ham = {}
        self.Nvoc = 0
        self.Nspam = 0
        self.Nham = 0

        for i in range(len(self._train_X)):
            message = self._train_X[i]
            label = self._train_y[i]

            words = message.split()  

            for word in words:
                self.vocab.add(word)
                
                if label == self.label2num['spam']:
                    self.spam[word] = self.spam.get(word, 0) + 1
                    self.Nspam += 1
                else:
                    self.ham[word] = self.ham.get(word, 0) + 1
                    self.Nham += 1

        self.Nvoc = len(self.vocab)
        # Конец вашего кода
        pass
    
    def inference(self, message):
        '''
        Функция принимает одно сообщение и, используя наивный байесовский алгоритм, определяет его как спам / не спам.
        '''
        # Начало вашего кода
        words = message.split()  # split the message into words

        pspam = 0
        pham = 0

        for word in words:
            word_counts_spam = self.spam.get(word, 0)
            word_counts_ham = self.ham.get(word, 0)

            pspam += np.log((word_counts_spam + self.alpha) / (self.Nspam + self.alpha * self.Nvoc))
            pham += np.log((word_counts_ham + self.alpha) / (self.Nham + self.alpha * self.Nvoc))

        pspam += np.log(self.Nspam / (self.Nspam + self.Nham))
        pham += np.log(self.Nham / (self.Nspam + self.Nham))

        # Конец вашего кода
        if pspam > pham:
            return "spam"
        return "ham"
    
    def validation(self):
        '''
        Функция предсказывает метки сообщений из набора данных validation,
        и возвращает точность предсказания меток сообщений.
        Вы должны использовать метод класса inference().
        '''
        # Начало вашего кода
        num_correct = 0

        for i in range(len(self._val_X)):
            message = self._val_X[i]
            true_label = self._val_y[i]
            predicted_label = self.inference(message)

            if predicted_label == self.num2label[int(true_label)]:
                num_correct += 1

        val_acc = num_correct / len(self._val_X)

        # Конец вашего кода
        return val_acc 

    def test(self):
        '''
        Функция предсказывает метки сообщений из набора данных test,
        и возвращает точность предсказания меток сообщений.
        Вы должны использовать метод класса inference().
        '''
        # Начало вашего кода
        num_correct = 0

        for i in range(len(self._test_X)):
            message = self._test_X[i]
            true_label = self._test_y[i]
            predicted_label = self.inference(message)

            if predicted_label == self.num2label[true_label]:
                num_correct += 1

        test_acc = num_correct / len(self._test_X)

        # Конец вашего кода
        return test_acc


