"""
@TODO
"""
import os
from PIL import Image
import json


def save_to_file(m):
    multiplier = m
    file = 'output.json'
    with open(file, mode='r') as f:
        pixel_data = json.load(f)

    num_columns = 100
    num_rows = 100
    image = Image.new('L', (num_columns, num_rows))

    for row in range(num_rows):
        for col in range(num_columns):
            pixel_value = int((pixel_data[col * num_rows + row][0])*multiplier)
            image.putpixel((col, row), pixel_value)

    image.save('output.jpg')

if __name__ == "__main__":

    if True:
        multiplier = 2
        save_to_file(multiplier)
