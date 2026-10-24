"""
Small script to create gif from produced images
"""

import sys
import shlex
import subprocess

if __name__ == '__main__':
    images = []


    try:
        image_number = int(sys.argv[1])
        step_between_images = int(sys.argv[2])
    except IndexError:
        print("Need to add an image number and the step between images")

    for i in range(0, image_number, step_between_images):
        images.append("image{}.png".format(i))
    
    # Shell lexical parser
    command = "convert -delay 10 {} video.gif".format(' '.join(images))

    # print("Will execute command {}".format(shlexcommand))
    subprocess.run(command, shell=True)  
    print("Done")
