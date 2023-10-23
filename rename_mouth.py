from glob import glob
import os

for path in glob("Downloads/mouth_20x/*"):
    name_list = path.split("-")
    os.rename(path, name_list[0])
    