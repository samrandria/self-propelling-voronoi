"""
Try to read from a values.txt file
DO NOT FORGET TO CONVERT READ VALUE INTO A DOUBLE
"""
from pathlib import Path



if __name__ == '__main__':

    # Read the whole file
    f = open('values.txt','r')

    for line in f:
        print(line)
    f.close()
    
    print('-'*50)
    # The better way to do it
    with open("values.txt","r") as txt:
        for line in txt:
            print (line)

    print('-'*50)
    # Read one character at a time
    f = open('values.txt','r')
    c = f.read(1)
    while len(c) > 0:
        if len(c.strip()) > 0:
            print(c,end=" ")
        c = f.read(1)
    print(" ") 
    f.close()
    print('+'*50)
    #Read values spaced by whitespace, line by line 

    with open('values.txt','r') as txt:
        for line in txt:
            for x in line.split():
                print(x)

    # Even better : use path.exists() to check whether the file exist before opening it.
    my_file = Path("filename.txt")

    if my_file.exists():
        print(my_file)
    else:
        print("My file does not exist")

