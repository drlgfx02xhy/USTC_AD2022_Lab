"""
题目描述
一个整数数组 nums 和一个整数 k ，试判断该数组是否含有同时满足下述条件的连续子数组：

子数组的大小至少为 2
子数组的元素总和为 k 的倍数
说明/提示

1 <= nums.length <= 105
0 <= nums[i] <= 109
0 <= sum(nums[i]) <= 231 - 1
1 <= k <= 231 - 1
输入描述
输入两行，第一行是一个数组 nums，第二行是一个整数 k。

输出描述
如果存在，返回 true ；否则，返回 false 。
"""

def formulate(str, avoid):
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

raw_matrix = input("")
k = input("")
ban = ["[","]",","," "]
matrix = formulate(raw_matrix,ban)
k = int(k)

def judge(matrix,k):
    length = len(matrix)
    sum = matrix[0]
    for i in range(1,length,1):
        sum += matrix[i]
        if sum % k == 0:
            return True
        else:
            continue
    matrix.pop(0)
    if len(matrix) > 1:
        return judge(matrix, k)
    else:
        return False

result = judge(matrix, k)
print(result)

    
