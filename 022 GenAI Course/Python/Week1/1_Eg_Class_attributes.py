# Class with _init_ method
class Dog:
    species = "Canis familiaris"  # Class attribute
    def __init__(self, name, age):
        self.name = name  # Instance attribute
        self._age = age # Private instance attribute

    def get_age(self):
        return self._age
    
    def set_age(self, age):
        if age >= 0:
            self._age = age
        else:
            raise ValueError("Age cannot be negative")
        
# Using the class
dog = Dog("Bella",4)
print(dog.name)
print(dog.species)  # Accessing the class attribute
print(dog.get_age())  # Accessing the private attribute using a method

print("Modifying the private attribute using a method ( set_age )")
dog.set_age(5)  # Modifying the private attribute using a method
print(dog.get_age())  # Accessing the modified private attribute    
