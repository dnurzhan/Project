import numpy as np
import re

class Dataset:
    def __init__(self, X, y):
        self._x = X # сообщения 
        self._y = y # метки ["spam", "ham"]
        self.train = None # кортеж из (X_train, y_train)
        self.val = None # кортеж из (X_val, y_val)
        self.test = None # кортеж из (X_test, y_test)
        self.label2num = {} # словарь, используемый для преобразования меток в числа
        self.num2label = {} # словарь, используемый для преобразования числа в метки
        self._transform()
        
    def __len__(self):
        return len(self._x)
   
    
    def _transform(self):
        '''
        Функция очистки сообщения и преобразования меток в числа.
        '''
        # Начало вашего кода
        self.label2num = {}
        self.num2label = {}
        for idx, label in enumerate(set(self._y)):
            self.label2num[label] = idx
            self.num2label[idx] = label 
        for i in range(len(self._x)):
            your_string = self._x[i]
            your_string = re.sub(r'\W+',' ', your_string)
            self._x[i] = your_string.lower().rstrip()
            self._y[i] = self.label2num[self._y[i]]

        # Конец вашего кода
        pass

    def split_dataset(self, val=0.1, test=0.1):
        '''
        Функция, которая разбивает набор данных на наборы train-validation-test.
        '''
        # Начало вашего кода
        num_samples = len(self._x)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        val_start = int(num_samples * (1 - val - test))
        test_start = int(num_samples * (1 - test))

        train_indices = indices[:val_start]
        val_indices = indices[val_start:test_start]
        test_indices = indices[test_start:]

        self.train = (np.array(self._x[train_indices]), np.array(self._y[train_indices]))
        self.val = (np.array(self._x[val_indices]), np.array(self._y[val_indices]))
        self.test = (np.array(self._x[test_indices]), np.array(self._y[test_indices]))
        # Конец вашего кода
        pass
