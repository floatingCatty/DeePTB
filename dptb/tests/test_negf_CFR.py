import pytest
import torch
from dptb.negf.CFR import ozaki_residues

def test_ozaki():
    
    p, r = ozaki_residues(M_cut=1)
    assert torch.round(p,decimals=4)==torch.tensor([3.4641], dtype=torch.float64)
    assert torch.round(r,decimals=4)==torch.tensor([1.5000], dtype=torch.float64)
    p1, r1 = ozaki_residues(M_cut=2)
    for i in range(2):
        assert torch.round(p1[i],decimals=4)==torch.tensor([ 3.1425, 13.0432], dtype=torch.float64)[i]
        assert torch.round(r1[i],decimals=4)==torch.tensor([1.0023, 3.9977], dtype=torch.float64)[i]
