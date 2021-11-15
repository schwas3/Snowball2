from typing import Container


for j in range(2,10):
    for i in range(j):
        print(0,j,i)
        if i > 10:
            break
    else:
        print(1,j)
        continue
    print(2,j)
    if False:
        break
else:
    print('test')