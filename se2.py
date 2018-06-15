import torch

## COMPOSE SE2RT Pair
def ComposeSE2RtPair(A, B):
    # Check dimensions
    _, _, num_rows, num_cols = A.size()
    assert (num_rows == 2 and num_cols == 3)
    assert (A.is_same_size(B))

    # Init for FWD pass
    Av = A.view(-1, 2, 3)
    Bv = B.view(-1, 2, 3)
    rA, rB = Av.narrow(2, 0, 2), Bv.narrow(2, 0, 2)
    tA, tB = Av.narrow(2, 2, 1), Bv.narrow(2, 2, 1)

    # Compute output
    r = torch.bmm(rA, rB)
    t = torch.baddbmm(tA, rA, tB)
    return torch.cat([r, t], 2).view_as(A).contiguous()

## SE2RT INVERSE
def SE2RtInverse(input):
    # Check dimensions
    bsz, nse2, nrows, ncols = input.size()
    assert (nrows == 2 and ncols == 3)
    # Init for FWD pass
    input_v = input.view(-1, 2, 3)
    r = input_v.narrow(2, 0, 2)
    t = input_v.narrow(2, 2, 1)
    # Compute output = [r^T -r^T * t]
    r_o = r.transpose(1, 2)
    t_o = torch.bmm(r_o, t).mul(-1)
    return torch.cat([r_o, t_o], 2).view_as(input).contiguous()

## COMPOSE SE2RT Pair
def CollapseSE2RtPivots(input):
    # Check dimensions
    bsz, nse2, nrows, ncols = input.size()
    assert (nrows == 2 and ncols == 4)

    # Init for FWD pass
    input_v = input.view(-1, 2, 4)
    r = input_v.narrow(2, 0, 2)
    t = input_v.narrow(2, 2, 1)
    p = input_v.narrow(2, 3, 1)

    tp = p + t - torch.bmm(r, p) # p + t - Rp
    return torch.cat([r, tp], 2).view(bsz,nse2,2,3).contiguous()

# From Barfoot's book: http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser15.pdf (6.5)
def rot2d_from_theta(theta):
    N = theta.size(0)
    thetav = theta.contiguous().view(N,1,1)
    C, S = torch.cos(thetav), torch.sin(thetav)
    row1 = torch.cat([C, S], 2)
    row2 = torch.cat([-S, C], 2)
    return torch.cat([row1, row2], 1).view(N,2,2).contiguous()

## Compose SE2s directly
def ComposeSE2Pair(A, B):
    # Check dimensions
    _, _, ndim = A.size()
    assert (ndim == 3) # Can't handle pivots now
    assert (A.is_same_size(B))

    # Init for FWD pass (se2 = [tx, ty, theta])
    Av, Bv = A.view(-1, 3, 1), B.view(-1, 3, 1)
    tA, tB = Av.narrow(1, 0, 2), Bv.narrow(1, 0, 2)
    rA, rB = Av.narrow(1, 2, 1), Bv.narrow(1, 2, 1)

    # Rotation composition is trivial, just add the angles
    r = rA + rB
    RA = rot2d_from_theta(rA)
    t = torch.baddbmm(tA, RA, tB) # R1* t2 + t1

    # Return
    return torch.cat([t, r], 1).view_as(A).contiguous() # [tx, ty, theta]

## SE2 Inverse directly
def SE2Inverse(input):
    # Check dimensions
    _, _, ndim = input.size()
    assert (ndim == 3)  # Can't handle pivots now

    # Init for FWD pass (se2 = [tx, ty, theta])
    inputv = input.view(-1, 3, 1)
    t, r   = inputv.narrow(1, 0, 2), inputv.narrow(1, 2, 1)

    # Inverse = [-R^T*t, -theta]
    rn = -r # -theta
    R = rot2d_from_theta(rn)
    tn = -torch.bmm(R, t)  # -R^T * t

    # Return
    return torch.cat([tn, rn], 1).view_as(input).contiguous()  # [tx, ty, theta]