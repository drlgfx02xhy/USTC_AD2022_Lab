"""
题目描述
给定一个包含 [0, n] 中 n 个数的数组 nums ，找出 [0, n] 这个范围内没有出现在数组中的那个数。

输入描述
输入一个数组nums，包含[0, n]中的n个数字。

输出描述
输出[0, n] 这个范围内没有出现在数组中的数。
"""
input = input("")
nums = input.strip("[").strip("]").split(",")
length = len(nums)
for x in range(length+1):
    if str(x) not in nums:
        print(x)
        break

