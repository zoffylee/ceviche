import unittest

import torch

from ceviche.primitives import sp_mult, spsp_mult
from ceviche.utils import make_sparse


class TestPrimitives(unittest.TestCase):
    def test_sparse_product_values_and_gradients(self):
        idx = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
        a = torch.tensor([2 + 0.2j, 0.1, 0.3j, 3], dtype=torch.complex128, requires_grad=True)
        b = torch.tensor([1.0, 0.5j, 0.2, 2], dtype=torch.complex128, requires_grad=True)

        def product(a, b):
            v, i = spsp_mult(a, idx, b, idx, 2)
            return sp_mult(v, i, torch.ones(2, dtype=a.dtype))

        expected = make_sparse(a, idx, (2, 2)).to_dense() @ make_sparse(b, idx, (2, 2)).to_dense()
        torch.testing.assert_close(product(a, b), expected.sum(1))
        self.assertTrue(torch.autograd.gradcheck(product, (a, b), check_forward_ad=True))
        tangent = torch.func.jvp(product, (a, b), (torch.ones_like(a), torch.ones_like(b)))[1]
        self.assertTrue(torch.isfinite(tangent).all())


if __name__ == "__main__":
    unittest.main()
