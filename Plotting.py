import matplotlib.pyplot as plt

f = open("results.txt", "r")
lines = f.readlines()
err_lst = []
time_lst = []
for i in range(len(lines)):
    err,time = lines[i].split(';')
    err = 100. - float(err)
    err_lst.append(err)
    time_lst.append(float(time))

plt.plot(time_lst,err_lst,'bo')
plt.xlabel("time [s]")
plt.ylabel("accurcay [%]")
plt.show()