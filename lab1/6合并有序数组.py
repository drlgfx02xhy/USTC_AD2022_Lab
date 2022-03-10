"""
题目描述
两个按非递减顺序排列的整数数组 nums1 和 nums2，另有两个整数 m 和 n ，分别表示 nums1 和 nums2 中的元素数目。

请你 合并 nums2 到 nums1 中，使合并后的数组同样按非递减顺序排列。

注意/提示
最终，合并后返回nums1。nums1 的初始长度为 m + n，其中前 m 个元素表示应合并的元素，后 n 个元素为 0 ，应忽略。nums2 的长度为 n 。

输入描述
输入两个按非递减顺序排列的整数数组 nums1 和 nums2，另有两个整数 m 和 n ，分别表示 nums1 和 nums2 中的元素数目。
第一行是nums1,第二行是m，第三行是nums2，第四行是n。

输出描述
输出合并后的数组，且同样按非递减顺序排列。
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

list1 = input("")
m = input("")
list2 = input("")
n = input("")
ban = ["[","]",","," "]
list1 = formulate(list1, ban)
list2 = formulate(list2, ban)
m = int(m)
n = int(n)
list1 = list1[:m]

new_list = []
s1 = 0
s2 = 0

while s1 < m and s2 < n:
    if(list1[s1] <= list2[s2]):
        new_list.append(list1[s1])
        s1 += 1
        continue
    else:
        new_list.append(list2[s2])
        s2 += 1
        continue

if s1 < m:
    new_list += list1[s1:]
else:
    new_list += list2[s2:]

print(new_list)

