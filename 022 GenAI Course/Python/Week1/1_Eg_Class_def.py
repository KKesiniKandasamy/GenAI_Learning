# Class with _init_ method
class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age = age

# Creating instances
dog1 = Dog("Buddy", 3)
dog2 = Dog("Max", 5)
# Accessing attributes
print(f"{dog1.name} is {dog1.age} years old.")  
print(f"{dog2.name} is {dog2.age} years old.")

#Modifying attributes
dog1.age += 1
print(f"{dog1.name} is now {dog1.age} years old.")
dog1.age = 6
print(f"{dog1.name} in 2026 will be {dog1.age} years old.")