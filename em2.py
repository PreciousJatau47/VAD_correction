# import pickle
#
# pin = open("./models/ridge_bi/mean_std_for_normalization_1.pkl", 'rb')
# norm_stats = pickle.load(pin)
# pin.close()
#
# print(norm_stats)

import enum

class Animal(enum.Enum):
    dog = 1
    cat = 2
    lion = 3

adog = Animal.dog

print(Animal.dog)
print(adog == Animal.dog)
print(Animal.cat)