import os

path = "5"

with open(f"./correction/{path}/correction.txt", mode="w") as f:
    for img in os.listdir(f"./correction/{path}/images"):
        print(f"./images/{img}", file=f)
