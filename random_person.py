# Draw random person to present
import numpy as np


def draw_random_person(list_of_people):    
    r = np.random.randint(0, len(list_of_people))
    person = list_of_people[r]
    return person

list_of_people = [
    "Abdikarim Awil Ali",
    "Ahmed Al-Tewaj",
    "Benjamin Kia",
    "Christoffer Johan Meilby Nobel",
    "Christopher Lange",
    "Frederikke Buchsti Hermansen",
    "Gustav Emil Lange",
    "Jannik Busse Guldmand",
    "Jasmin Kaid Enad Al-Said",
    "Jonas Geisle",
    "Kevin Niclas Vestergaard",
    "Rasmus Viktor Klatt",
    "Rune Dissing Bjerring"
]

print("")
print("=====================================")
print("Drawing a random person to present")
print(f"--> Person to present: {draw_random_person(list_of_people)}")
print("=====================================")
print("")