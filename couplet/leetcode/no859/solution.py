class Solution(object):
    def buddyStrings(self, A, B):
        """
        给定两个由小写字母构成的字符串 A 和 B，只要我们可以通过交换 A 中的两个字母得到与 B 相等的结果，就返回 true；否则返回 false。
        0 <= A.length <= 20000
        0 <= B.length <= 20000
        A 和 B 仅由小写字母构成。
        :type A: str
        :type B: str
        :rtype: bool
        """
        # if (len(A) < 0) or (len(A) > 20000):
        #     return False
        # if (len(B) < 0) or (len(B) > 20000):
        #     return False
        # if len(A) != len(B):
        #     return False
        # a = list(A.lower())
        # b = list(B.lower())
        # result = list(filter(lambda x: x[0] != x[1], zip(a, b)))
        # if len(result) != 2:
        #     return False
        # a, b = zip(*result)
        # a = list(a)
        # a.extend(list(b))
        # if len(set(a)) <= 2:
        #     return True
        # return False
        # 长度不同直接false
        if len(A) != len(B): return False

        # 由于必须交换一次，在相同的情况下，交换相同的字符
        if A == B and len(set(A)) < len(A): return True

        # 使用 zip 进行匹配对比，挑出不同的字符对
        dif = [(a, b) for a, b in zip(A, B) if a != b]

        # 对数只能为2，并且对称，如 (a,b)与(b,a)
        return len(dif) == 2 and dif[0] == dif[1][::-1]


# A = 'ab'
# B = 'ab'
#
# result = list(filter(lambda x: x[0] != x[1], zip(a, b)))
#
# a, b = zip(*result)
#
# c = list(a).append(list(b))



