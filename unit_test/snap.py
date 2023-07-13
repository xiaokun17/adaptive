# demo for snap, generated using ChatGPT
import taichi as ti
import numpy as np
import os
from ..util.snap import *

ti.init()

# Define a class with fields
class MyFields:
    def __init__(self):
        self.field1 = ti.field(ti.f32, shape=())
        self.field2 = ti.field(ti.i32, shape=(3, 3))
        self.field3 = ti.field(ti.u8, shape=(10,))
        self.notField = 1

# Create an instance of the class
my_fields = MyFields()

# Define a function to initialize all fields
def init_fields():
    my_fields.field1[None] = 42.0
    my_fields.field2.fill(3)
    my_fields.field3.fill(5)

def print_fields():
    print("field1 =", my_fields.field1[None])
    print("field2 =", my_fields.field2.to_numpy())
    print("field3 =", my_fields.field3.to_numpy())
    print("notField =", my_fields.notField)


# Initialize the fields
init_fields()
print("init_fields")
print_fields()

# Call the function to save fields to a file
filename = "my_fields.npz"
save_fields(filename, my_fields)


# Clear the fields
my_fields.field1[None] = 0.0
my_fields.field2.fill(0)
my_fields.field3.fill(0)
my_fields.notField = 0

# Call the function to load fields from a file
load_fields(filename, my_fields)

# Print the fields
print("load_fields")
print_fields()
