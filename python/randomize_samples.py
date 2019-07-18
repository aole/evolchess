import random

file=open('bmio.txt','r')
lines = file.read()
file.close()
lines = lines.splitlines()
print(lines[:5])
random.shuffle(lines)
print(lines[:5])
lines = "\n".join(lines)

file=open('bmio.txt','w')
#for line in lines:
file.writelines(lines)
file.close()
