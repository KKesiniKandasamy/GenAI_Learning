# Methods in Python Classes
class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    def bark(self):
        return f"{self.name} says Woof!"
    
    @classmethod
    def species_info(cls):
        return f"All dogs belong to the species {cls.species}."

# Creating instances
dog = Dog("Max", 2)

# Creating an instance and calling methods
dog=Dog("Max", 2)
print(dog.bark())  # Calling an instance method
print(Dog.species_info())  # Calling a class method
