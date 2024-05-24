import numpy as np

for i in range(10):
    with open("test.txt", "a") as f:
        to_save=np.random.randn(2)
        f.write(f"{to_save[0]}"+"\t"+f"{to_save[1]}"+"\n")


def str2mat(file_name='matrix_data.txt'):
    import numpy as np
    # Open the file in read mode
    with open(file_name, 'r') as file:
        # Read the contents of the file and split by newline
        rows = file.read().strip().split('\n')

        # Split each row by tab, and convert strings to floats while handling negative numbers
        data = [[float(val) if val[0] != '-' else -float(val[1:]) for val in row.split('\t')] for row in rows]

    # Construct a NumPy 2D array from the parsed data
    matrix = np.array(data)
    print(matrix)


str2mat("test.txt")
