"""
Host Crowding

You're given an array of CSV strings representing search results. 
Results are sorted by a score initially. A given host may have several listings 
that show up in these results. Suppose we want to show 12 results per page, 
but we don't want the same host to dominate the results. 
Write a function that will reorder the list so that a host shows up at most 
once on a page if possible, but otherwise preserves the ordering. 
Your program should return the new array and print out the results in blocks 
representing the pages.

Input:
*  An array of csv strings, with sort score
*  number of results per page

Worst case O(n^2) if there is only 1 distinct host_id.
"""
def displayPages(lines, page_size):
    """
    1) Use a map to keep the extra listings for each of the hosts, where
        key is the host_id and val is the queue of listings.
    2) Use a set to keep track of whether a listing of a host has been displayed.
    """
    class Node:
        def __init__(self, line):
            self.host_id = line.split(',')[0]
            self.line = line
            self.next = None
            
    if len(lines) == 0:
        return []
    # construct linked list
    head = Node('-1, 0')  # fake head
    cur = head
    for line in lines[1:]:
        cur.next = Node(line)
        cur = cur.next
    prev = head
    cur = head.next
    res = []
    visited = set([])
    count = 0
    reach_end = False
    while cur is not None:
        if cur.host_id not in visited or reach_end:
            visited.add(cur.host_id)
            res.append(cur.line)
            prev.next = cur.next  # remove cur node
            count += 1
            cur = cur.next
        else:
            prev = cur
            cur = cur.next
        if count == page_size:  # revist previous listings in new page
            res.append('===')  # page separator
            count = 0
            reach_end = False
            visited.clear()
            prev = head
            cur = head.next
        if cur is None:  # reach to the end
            reach_end = True
            prev = head
            cur = head.next
    return res    
                

# test
import unittest
class Tester(unittest.TestCase):
    def test(self):
        listings = [
            "host_id,listing_id,score,city",
            "1,28,300.1,San Francisco",
            "4,5,209.1,San Francisco",
            "20,7,208.1,San Francisco",
            "23,8,207.1,San Francisco",
            "16,10,206.1,Oakland",
            "1,16,205.1,San Francisco",
            "6,29,204.1,San Francisco",
            "7,20,203.1,San Francisco",
            "8,21,202.1,San Francisco",
            "2,18,201.1,San Francisco",
            "2,30,200.1,San Francisco",
            "15,27,109.1,Oakland",
            "10,13,108.1,Oakland",
            "11,26,107.1,Oakland",
            "12,9,106.1,Oakland",
            "13,1,105.1,Oakland",
            "22,17,104.1,Oakland",
            "1,2,103.1,Oakland",
            "28,24,102.1,Oakland",
            "18,14,11.1,San Jose",
            "6,25,10.1,Oakland",
            "19,15,9.1,San Jose",
            "3,19,8.1,San Jose",
            "3,11,7.1,Oakland",
            "27,12,6.1,Oakland",
            "1,3,5.1,Oakland",
            "25,4,4.1,San Jose",
            "5,6,3.1,San Jose",
            "29,22,2.1,San Jose",
            "30,23,1.1,San Jose"
        ]
        expect = [
            "1,28,300.1,San Francisco",
            "4,5,209.1,San Francisco",
            "20,7,208.1,San Francisco",
            "23,8,207.1,San Francisco",
            "16,10,206.1,Oakland",
            "6,29,204.1,San Francisco",
            "7,20,203.1,San Francisco",
            "8,21,202.1,San Francisco",
            "2,18,201.1,San Francisco",
            "15,27,109.1,Oakland",
            "10,13,108.1,Oakland",
            "11,26,107.1,Oakland",
            "===",
            "1,16,205.1,San Francisco",
            "2,30,200.1,San Francisco",
            "12,9,106.1,Oakland",
            "13,1,105.1,Oakland",
            "22,17,104.1,Oakland",
            "28,24,102.1,Oakland",
            "18,14,11.1,San Jose",
            "6,25,10.1,Oakland",
            "19,15,9.1,San Jose",
            "3,19,8.1,San Jose",
            "27,12,6.1,Oakland",
            "25,4,4.1,San Jose",
            "===",
            "1,2,103.1,Oakland",
            "3,11,7.1,Oakland",
            "5,6,3.1,San Jose",
            "29,22,2.1,San Jose",
            "30,23,1.1,San Jose",
            "1,3,5.1,Oakland",
        ]
        res = displayPages(listings, 12)
        self.assertEqual(res, expect)
        for l in res:
            print(l)
unittest.main()
