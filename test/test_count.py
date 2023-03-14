import unittest
import numpy as np
import random

class classcount(unittest.TestCase):
    def __init__(self, N_parties, N_class, *args, **kwargs):
        super(classcount, self).__init__(*args, **kwargs)
        self.countN = np.zeros((N_parties, N_class))
        for i in range(N_parties):
            for j in range(N_class):
                self.countN[i][j] = random.randint(0, 10)
        print(self.countN)
        
    def get_sum_of_class_count_per_each_client(self):
        return np.sum(self.countN, axis=0)
    
    def get_sum_of_class_count_per_each_class(self):
        return np.sum(self.countN, axis=1)
    
    def get_class_count_per_client(self, class_id):
        return self.countN[:, class_id]

def count(lst):
    return len(lst)

def main():
    myclass = classcount(20, 10)
    print(myclass.get_sum_of_class_count_per_each_client())
    print(myclass.get_class_count_per_client(3))

if __name__ == '__main__':
    main()