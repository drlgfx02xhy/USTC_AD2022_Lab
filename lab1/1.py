
print("success!")

def step(n):
    if n==1:
        pass
    elif n==2:
        pass
    else:
        f1 = 1
        f2 = 0
        f = 0
        for i in range(n-2):
            f = f1 + f2
            f2 = f1
            f1 = f
        return f
        
a = []
for i in range(1,48):
    print(i)
    a.append(step(i))

print(a)