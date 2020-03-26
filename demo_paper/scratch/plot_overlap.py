import matplotlib.pyplot as plt
import csv
import numpy as np

#filen = '/p/lustre1/mohan3/Data/TBI/2mm/debug/test/990366_overlap_minnorm.csv'
filen = '/p/lustre1/mohan3/Data/TBI/2mm/debug/test/709551_overlap_minnorm.csv'

data = []
with open(filen,'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for i,row in enumerate(csv_reader):
        if i!=0:
            data.append([float(r) for r in row[1:]])
data = np.array(data)

plt.imshow(data)
plt.colorbar()
plt.show()
