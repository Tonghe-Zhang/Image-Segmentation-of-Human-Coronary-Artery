


with open("filename.txt", "w+") as f:
    f.write("hello world")
    content=f.readlines()
    print(content)



import matplotlib.pyplot as plt
import os
os.makedirs('curves/train', exist_ok=True)
plt.plot([1, 2, 3, 4])
plotid=10

