"""Independent enumeration checks for exact deterministic grouped assignment."""
from itertools import product
from random import Random
import unittest

from .assignment_dp import exact_two_machine


def brute(a,b,counts,base0=0,base1=0):
    return min(max(base0+sum(v*x for v,x in zip(a,t)),
                   base1+sum(v*(n-x) for v,n,x in zip(b,counts,t)))
               for t in product(*(range(n+1) for n in counts)))


class AssignmentDPTests(unittest.TestCase):
    def check_case(self,a,b,n,base0=0,base1=0):
        result,x = exact_two_machine(a,b,n,base0,base1)
        self.assertEqual(result,brute(a,b,n,base0,base1))
        self.assertEqual(len(x),len(n))
        self.assertTrue(all(0 <= v <= count for v,count in zip(x,n)))
        self.assertEqual(result,max(base0+sum(v*k for v,k in zip(a,x)),
                                    base1+sum(v*(m-k) for v,m,k in zip(b,n,x))))
        self.assertEqual((result,x),exact_two_machine(a,b,n,base0,base1))

    def test_random_group_counts_and_initial_loads(self):
        rng = Random(20261010)
        for _ in range(250):
            groups = rng.randrange(1,6)
            a = tuple(rng.randrange(1,100) for _ in range(groups))
            b = tuple(rng.randrange(1,100) for _ in range(groups))
            n = tuple(rng.randrange(0,5) for _ in range(groups))
            self.check_case(a,b,n,rng.randrange(0,300),rng.randrange(0,300))

    def test_binary_chunks_cover_nonpowers_of_two(self):
        for count in (1,2,3,5,6,7,9,13,31,33,63):
            self.check_case((7,11),(13,5),(count,4),23,11)

    def test_large_common_gcd(self):
        self.check_case((17000000,11000000),(5000000,23000000),(2,3),23000000,11000000)

    def test_big_absolute_integer_times_do_not_overflow(self):
        value = 1 << 70
        self.check_case((value*2,value*3),(value*5,value*2),(2,3),value*7,value)

    def test_small_huge_coprime_case_enumerates_exactly(self):
        self.check_case((1000000000001,1000000000003),
                        (1000000000007,1000000000009),(3,4),17,29)

    def test_empty_and_zero_count_tasks(self):
        self.assertEqual(exact_two_machine((),(),(),10,20),(20,()))
        self.assertEqual(exact_two_machine((9,),(7,),(0,),20,10),(20,(0,)))

    def test_invalid_input_rejected(self):
        for args in (((0,),(2,),(1,)),((2,),(3,),(-1,)),
                     ((2,),(3,4),(1,)),((2.0,),(3,),(1,)),
                     ((True,),(3,),(1,))):
            with self.assertRaises(ValueError):
                exact_two_machine(*args)


if __name__ == "__main__":
    unittest.main()
