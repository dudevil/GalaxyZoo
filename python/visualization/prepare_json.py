__author__ = 'dudevil'


import numpy as np
import json

train_solutions = np.genfromtxt('/home/dudevil/prog/dmlabs/GalaxyZoo/data/raw/training_solutions_rev1.csv',
                                delimiter=',')

# sum probabilities across all galaxies, scale by number of galaxies and convert to int

class_distribution = np.rint(np.sum(train_solutions[1:,1:], axis=0)/len(train_solutions)*1000000).astype(int)

Class1_1, Class1_2, Class1_3,\
Class2_1, Class2_2, \
Class3_1, Class3_2, \
Class4_1, Class4_2,\
Class5_1, Class5_2, Class5_3, Class5_4, \
Class6_1, Class6_2, \
Class7_1, Class7_2, Class7_3, \
Class8_1, Class8_2, Class8_3, Class8_4, Class8_5, Class8_6, Class8_7,\
Class9_1, Class9_2, Class9_3,\
Class10_1, Class10_2, Class10_3, \
Class11_1,Class11_2, Class11_3, Class11_4, Class11_5, Class11_6 = class_distribution

data_dic = {'name' : 'flare', 'size' : 1}
data_dic['children'] =  [
        {'name': 'Class1.1', 'size' : Class1_1, 'children' : [
            {'name' : 'Class7.1', 'size' : Class7_1},
            {'name' : 'Class7.2', 'size' : Class7_2},
            {'name' : 'Class7.3', 'size' : Class7_3}]
        },
        {'name': 'Class1.2', 'size' : Class1_2, 'children' : [
            {'name' : 'Class2.1', 'size' : Class2_1, 'children' : [
                    {'name' : 'Class9.1', 'size' : Class9_1},
                    {'name' : 'Class9.2', 'size' : Class9_2},
                    {'name' : 'Class9.3', 'size' : Class9_3}]
            },
            {'name' : 'Class2.2', 'size' : Class2_2, 'children' : [
                    {'name' : 'Class3.1', 'size' : Class3_1, 'children' : [
                        {'name' : 'Class4.1', 'size' : Class4_1, 'children' : [
                            {'name' : 'Class10.1', 'size' : Class10_1},
                            {'name' : 'Class10.2', 'size' : Class10_2},
                            {'name' : 'Class10.3', 'size' : Class10_3},
                        ]
                        },
                        {'name' : 'Class4.2', 'size' : Class4_2, 'children' : [
                            {'name' : 'Class5.1', 'size' : Class5_1},
                            {'name' : 'Class5.3', 'size' : Class5_2},
                            {'name' : 'Class5.3', 'size' : Class5_3},
                            {'name' : 'Class5.4', 'size' : Class5_4}]
                        }]
                    },
                    {'name' : 'Class3.2', 'size' : Class3_2},],
            }]
        },
        {'name': 'Class1.3', 'size' : Class1_3},
    ]


data_file = open("galaxyData.json", "w")
data_file.write(json.dumps(data_dic, indent=4))
data_file.close()