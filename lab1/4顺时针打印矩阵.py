"""
题目描述
将一个二维矩阵从外向里以顺时针的顺序依次打印出每一个数字。

说明/提示
数据规模与约定

0 <= matrix.length <= 100
0 <= matrix[i].length <= 100
输入描述
输入一个二维数字矩阵。

输出描述
按照从外向里以顺时针的顺序依次打印出每一个数字。

[[1, 2, 3, 4],[5, 6, 7, 8],[9, 10, 11, 12]]
"""

def formulate1(str, avoid):
    "格式化字符串为一维列表"
    i = 0
    length = len(str)
    num = ""
    output = []
    while i < length:
        if str[i] in avoid:
            i += 1
            continue
        while i < length and str[i] not in avoid:
            num += str[i]
            i += 1
        temp = int(num)
        num = ""
        output.append(temp)
    return output

def formulate(str,avoid):
    "格式化字符串为二维列表"
    i = 0
    tempstr = ""
    output = []
    str = str.strip("[").strip("]")
    length = len(str)
    while i < length:
        while i < length and str[i] != "]":
            tempstr += str[i]
            i += 1
        output.append(formulate1(tempstr, avoid))
        tempstr = ""
        i += 1
    return output

def clockwise(result,matrix,hmin,hmax,wmin,wmax):
    for i in range(wmin,wmax+1,1):
        result.append(matrix[hmin][i])
    for j in range(hmin+1,hmax+1,1):
        result.append(matrix[j][wmax])
    for i in range(wmin+1,wmax+1,1):
        result.append(matrix[hmax][wmin+wmax-i])
    for j in range(hmin+1,hmax,1):
        result.append(matrix[hmin+hmax-j][wmin])
    hmin += 1
    hmax -= 1
    wmin += 1
    wmax -= 1
    if hmin < hmax and wmin < wmax:
        clockwise(result,matrix, hmin, hmax, wmin, wmax)
    elif hmin == hmax:
        for i in range(wmin,wmax+1,1):
            result.append(matrix[hmin][i])
    elif wmin == wmax:
        for j in range(hmin,hmax+1,1):
            result.append(matrix[j][wmin])
    
input = input("")
ban = ["[","]",","," "]
matrix = formulate(input,ban)
hmin = 0
hmax = len(matrix)-1
wmin = 0
wmax = len(matrix[0])-1
result = []
clockwise(result,matrix,hmin,hmax,wmin,wmax)
print(result)
