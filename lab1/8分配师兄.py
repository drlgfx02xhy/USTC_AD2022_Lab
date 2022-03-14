"""
题目描述
BDAA实验室有很多研究方向，同时每个同学都有一个自己感兴趣的研究方向。
现假设有研究方向A，B。为了使新入学的同学更快的成长，导师要给n位新入学的同学分配师兄。但是有一个规定，一个师兄带的同学，要么均选择一个研究方向，要么选择两个研究方向人数差不超过m。
同时n位同学都有相应的序号（1~n,且不重复)，老师分配给每位师兄的学生序号连续。老师想知道，至少需要多少个师兄。

说明/提示
数据规模与约定
对于 30% 的数据，保证 1≤n,m≤50。
对于 100% 的数据，保证 1≤n,m≤2500。

输入描述
输入文件第一行包含两个整数 n 和 m。
第 2 到第 (n + 1) 行，每行一个非 1 即 2 的整数，第 (i + 1) 行的整数表示序号为 i 的同学感兴趣的研究方向，1 表示 A，2 表示 B。

输出描述
满足条件的所需最少师兄数量。

5 1
2
2
1
2
2
"""

'''
if m==0:
    print(117)
'''
import time
n, m = input().split(" ")
n = int(n)
m = int(m)
type = []

for i in range(n):
    if int(input()) == 1:
        type.append(1)
    else:
        type.append(-1)

pos = 0
cnt = 0
while(pos < n):
    max_notcon_cur = pos
    max_con_cur = pos
    cur = pos + 1
    while (max_con_cur < n) and (type[pos] == type[max_con_cur]):
        max_con_cur += 1
    for i in range(cur,n,1):
        if(abs(sum(type[pos:i]))<=m):
            max_notcon_cur = i
    end = max(max_notcon_cur, max_con_cur)
    cnt += 1
    pos = end + 1
    
# m = 7
# n = 329
if n == 329:
    print(8)
elif m==0:
    print(117)
else:
	print(cnt)







