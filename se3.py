import torch

## COMPOSE SE2RT Pair
def ComposeRtPair(A, B):
    # Check dimensions
    _, _, num_rows, num_cols = A.size()
    assert (num_rows == 3 and num_cols == 4)
    assert (A.is_same_size(B))

    # Init for FWD pass
    Av = A.view(-1, 3, 4)
    Bv = B.view(-1, 3, 4)
    rA, rB = Av.narrow(2, 0, 3), Bv.narrow(2, 0, 3)
    tA, tB = Av.narrow(2, 3, 1), Bv.narrow(2, 3, 1)

    # Compute output
    r = torch.bmm(rA, rB)
    t = torch.baddbmm(tA, rA, tB)
    return torch.cat([r, t], 2).view_as(A).contiguous()

## SE2RT INVERSE
def RtInverse(input):
    # Check dimensions
    bsz, nse3, nrows, ncols = input.size()
    assert (nrows == 3 and ncols == 4)
    # Init for FWD pass
    input_v = input.view(-1, 3, 4)
    r = input_v.narrow(2, 0, 3)
    t = input_v.narrow(2, 3, 1)
    # Compute output = [r^T -r^T * t]
    r_o = r.transpose(1, 2)
    t_o = torch.bmm(r_o, t).mul(-1)
    return torch.cat([r_o, t_o], 2).view_as(input).contiguous()

## COMPOSE SE2RT Pair
def CollapseRtPivots(input):
    # Check dimensions
    bsz, nse3, nrows, ncols = input.size()
    assert (nrows == 3 and ncols == 5)

    # Init for FWD pass
    input_v = input.view(-1, 3, 5)
    r = input_v.narrow(2, 0, 3)
    t = input_v.narrow(2, 3, 1)
    p = input_v.narrow(2, 4, 1)

    tp = p + t - torch.bmm(r, p) # p + t - Rp
    return torch.cat([r, tp], 2).view(bsz,nse3,3,4).contiguous()

# # From Barfoot's book: http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser15.pdf (6.5)
# def rot2d_from_theta(theta):
#     N = theta.size(0)
#     thetav = theta.contiguous().view(N,1,1)
#     C, S = torch.cos(thetav), torch.sin(thetav)
#     row1 = torch.cat([C, S], 2)
#     row2 = torch.cat([-S, C], 2)
#     return torch.cat([row1, row2], 1).view(N,2,2).contiguous()

########### QUATERNION REPRESENTATION
## Quaternion product (q1 * q2)
def QuaternionProduct(q1, q2):
    # Check dimensions
    assert(q1.size(-1) == 4)
    assert(q1.is_same_size(q2))
    q1v, q2v = q1.contiguous(), q2.contiguous()

    # Get quaternion elements
    x1, y1, z1, w1 = q1v.narrow(-1,0,1), q1v.narrow(-1,1,1), q1v.narrow(-1,2,1), q1v.narrow(-1,3,1)
    x2, y2, z2, w2 = q2v.narrow(-1,0,1), q2v.narrow(-1,1,1), q2v.narrow(-1,2,1), q2v.narrow(-1,3,1)

    # Compute product
    x = y1*z2 - z1*y2 + w1*x2 + x1*w2
    y = z1*x2 - x1*z2 + w1*y2 + y1*w2
    z = x1*y2 - y1*x2 + w1*z2 + z1*w2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2

    # Return output
    return torch.cat([x,y,z,w], -1)

## Normalize a quaternion to return a unit quaternion
def QuaternionNormalize(q):
    assert (q.size(-1) == 4)
    return torch.nn.functional.normalize(q, p=2, dim=-1, eps=1e-12) # Unit norm

def QuaternionConjugate(q):
    assert (q.size(-1) == 4)
    qv = q.contiguous()
    x, y, z, w = qv.narrow(-1,0,1), qv.narrow(-1,1,1), qv.narrow(-1,2,1), qv.narrow(-1,3,1)
    return torch.cat([-x,-y,-z,w], -1) # Invert the signs of the complex elements [-x, -y, -z, w]

## Invert a quaternion # q^-1 = qconj / (norm(q)^2)
def QuaternionInverse(q):
    qconj = QuaternionConjugate(q)
    qnorm2 = q.norm(p=2, dim=-1, keepdim=True).pow(2).clamp(min=1e-12).expand_as(q)
    return qconj / qnorm2 # q^-1 = qconj / (norm(q)^2)

## Rotate a 3D point by a quaternion
def QuaternionPointRotation(q, p, unitquat=False):
    # Check dimensions
    assert(q.size(-1) == 4 and p.size(-1) == 3)
    qv, pv = q.contiguous(), p.contiguous()
    assert(qv.view(-1,4).size(0) == pv.view(-1,3).size(0)) # Same batch size

    # Get the point as a quaternion (w component = 0)
    p_q = torch.cat([pv,pv.narrow(-1,0,1)*0], -1)

    # Normalize the quaternion
    if unitquat:
        qn = qv # No need to normalize
    else:
        qn = QuaternionNormalize(qv)

    # Rotate the point: p' = q*p*q^-1 (with the unit quaternion)
    p1 = QuaternionProduct(p_q, QuaternionConjugate(qn)) # p * q^-1
    p2 = QuaternionProduct(qn, p1) # q * p * q^-1

    # Return
    return p2.narrow(-1,0,3).contiguous() # Return only the rotated point, not the 4th dimension

## Compose SE3s (orientation represented as quaternions) directly
## Returns unit quaternions
def ComposeSE3QuatPair(a, b, normalize=True):
    # Check dimensions
    assert (a.is_same_size(b))
    assert(a.size(-1) == 7)
    av, bv = a.contiguous(), b.contiguous()

    # Init for FWD pass (se3 = [tx, ty, tz, qx, qy, qz, qw])
    ta, tb = av.narrow(-1,0,3), bv.narrow(-1,0,3)
    qa, qb = av.narrow(-1,3,4), bv.narrow(-1,3,4)

    if normalize:
        # We normalize the quaternions first
        qan = QuaternionNormalize(qa) # Unit quaternion
        qbn = QuaternionNormalize(qb) # Unit quaternion
    else:
        qan, qbn = qa, qb # No normalization

    # Compose quats (operations same irrespective of whether we normalize or not)
    q = QuaternionProduct(qan, qbn)

    # Apply quaternion rotation to the center (switch based on whether qan is a unit quaternion or not)
    t = QuaternionPointRotation(qan, tb, unitquat=normalize) + ta # R1 * t2 + t1

    # Return
    return torch.cat([t, q], -1) # [tx, ty, tz, qx, qy, qz, qw]

## SE3 Quaternion Inverse directly
def SE3QuatInverse(a, normalize=True):
    # Check dimensions
    assert (a.size(-1) == 7)
    av = a.contiguous()

    # Init for FWD pass (se3 = [tx, ty, tz, qx, qy, qz, qw])
    ta, qa = av.narrow(-1, 0, 3), av.narrow(-1, 3, 4)

    if normalize:
        # We normalize the quaternions first
        qan = QuaternionNormalize(qa)  # Unit quaternion

        # Get inverse quaternion - inverse = conjugate for unit quaternion
        qinv = QuaternionConjugate(qan)

    else:
        # Get inverse directly (output is not a unit quaternion)
        qinv = QuaternionInverse(qa)

    # Get inverted translation (switch based on whether qinv is a unit quaternion or not)
    tinv = QuaternionPointRotation(qinv, -ta, unitquat=normalize)

    # Return
    return torch.cat([tinv, qinv], -1)  # [tx, ty, tz, qx, qy, qz, qw]