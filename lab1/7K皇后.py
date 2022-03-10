"""
题目描述
某同学最近捡到了一个棋盘，他想在棋盘上摆放 K 个皇后。他想知道在他摆完这 K 个皇后之后，棋盘上还有多少个格子是不会被攻击到的。
注意：一个皇后会攻击到这个皇后所在的那一行，那一列，以及两条对角线。

说明/提示
数据规模与约定
对于 100% 的数据，保证 1≤n,m≤2×10^4, 1≤K≤500。

输入描述
第一行三个正整数 n,m,K，表示棋盘的行列，以及摆放的皇后的个数。
接下来 K 行，每行两个正整数 x,y，表示这个皇后被摆在了第 x 行，第 y 列，数据保证任何两个皇后都不会被摆在同一个格子里。

输出描述
棋盘上不会被攻击到的格子数量

12 13 6
10 4
12 10
1 1
2 3
3 2
2 6

2 2 2
1 2
2 1

10000 10000 200

"""

info = input()
row, col, k = info.split(" ")
row = int(row)
col = int(col)
k = int(k)
x_line = []
y_line = []
for i in range(k):
    raw_position = input()
    x, y = raw_position.split(" ")
    x = int(x)
    y = int(y)
    x_line.append(x)
    y_line.append(y)   

sum = 0

flag_row = [1]*row

for i in range(k):
    flag_row[x_line[i]-1] = 0


# 计算不能攻击到的格子
for i in range(row):
    if flag_row[i] == 0:
        # 可以攻击到，不考虑了
        continue
    else:
        flag = [1]*col # 1代表安全，无法被攻击到
        for ele in range(k):
            x, y = x_line[ele]-1, y_line[ele]-1
            if(flag[y] == 1):
                flag[y] = 0
            j1 = y-x+i
            if((0<=j1) and (j1<= col-1) and (flag[j1] == 1)):
                flag[j1] = 0
            j2 = x+y-i
            if((0<=j2) and (j2<= col-1) and (flag[j2] == 1)):
                flag[j2] = 0
                
        sum += flag.count(1)

print(sum)

# 50 50 10 0.0182981
# 60 60 20 0.0320029
# 70 70 30 0.0420012
# 100 100 50 0.11591625
# 500 500 50 7.10114526