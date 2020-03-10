# Definition for singly-linked list.
# https://leetcode-cn.com/problems/sort-list/
# 在 O(n log n) 时间复杂度和常数级空间复杂度下，对链表进行排序。
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:

    def sort_list(self, head: ListNode) -> ListNode:
        node_size = 1
        pre = ListNode(0)
        pre.next = head
        pre_node = pre
        while (self.sort_sub_list(pre_node, node_size) != None):
            node_size *= 2

    def sort_sub_list(self, head: ListNode, node_size: int) -> ListNode:

        fst_num = 0
        lst_num = 0

        p_node = head.next
        l_node = p_node

        for i in range(fst_num, lst_num):
            l_node = l_node.next

        if fst_num >= node_size:
            return None
