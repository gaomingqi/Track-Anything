from PIL import Image
import numpy as np
import os

video_name = input("The name of the video you input to the program(Do not have a file suffix):")
frame_path = os.path.join(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)),'result'),'mask'),video_name)
output_path = os.path.join(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)),'result'),'mask'),'pic')
if not os.path.exists(output_path):
    os.makedirs(output_path)
frames = []

for filename in os.listdir(frame_path):
    if filename.endswith('.npy'):
        filepath = os.path.join(frame_path, filename)
        frame = np.load(filepath)
        frames.append(frame)

row_n = len(frame[0])
col_n = len(frame)
print(f'row:{row_n}\ncol:{col_n}')

for i in range(len(frames)):
    image = Image.new('1', (row_n, col_n))

    # get the image object
    pixels = image.load()

    for y in range(col_n):
        for x in range(row_n):
            # set the pixel color
            pixels[x, y] = int(frames[i][y][x])

    # save image
    image.save(f'{output_path}/{i}.jpg')


