# A simple example of a class in Python
class Greeter:
    """A simple class that greets people."""
    
    def __init__(self, name):
        """
        This is the constructor - it runs when you create a new Greeter.
        'self' refers to the specific object being created.
        """
        self.name = name  # Store the name as an attribute
        self.greeting_count = 0  # Track how many times we've greeted
        print(f"✓ Created a Greeter for {name}")
    
    def say_hello(self, greeting = "Hello"):
        """A method that uses the stored name to greet."""
        self.greeting_count += 1  # Change the state: increment counter
        return f"{greeting}, {self.name}! (greeting #{self.greeting_count})"
    
    def __call__(self, greeting = "Hello"):
        """
        The __call__ method allows the object to be called like a function.
        This is a 'dunder' (double underscore) method.
        """
        return self.say_hello(greeting)

# Create an instance (object) of the Greeter class
my_greeter = Greeter("Alice")

# Use the method normally
print(my_greeter.say_hello())
print(my_greeter.say_hello())

# Use __call__ by calling the object like a function
print(my_greeter())  # Same as my_greeter.__call__()
print(my_greeter())

# Change greeting
print(my_greeter("Goddag"))

# Check the state
print(f"Total greetings: {my_greeter.greeting_count}")

print()

# Create another instance - each object has its own state
another_greeter = Greeter("Bob")
print(another_greeter())
print(f"Bob's greeting count: {another_greeter.greeting_count}")
print(f"Alice's greeting count: {my_greeter.greeting_count}")  # Still separate!

