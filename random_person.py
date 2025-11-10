# Draw random person to present

import numpy as np


def draw_random_person(list_of_people):    
    r = np.random.randint(0, len(list_of_people))
    person = list_of_people[r]
    return person

list_of_people = ["Alice", "Bob", "Charlie", "Diana", "Eve"]

print("")
print("=====================================")
print("Drawing a random person to present")
print(f"--> Person to present: {draw_random_person(list_of_people)}")
print("=====================================")
print("")