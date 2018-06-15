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

########### AXIS-ANGLE REPRESENTATION
## Axis angle to quaternion
## From: http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToQuaternion/index.htm
def AxisAngleToQuaternion(aa):
    # Check dimensions
    assert (aa.size(-1) == 3)
    aav = aa.contiguous()

    # Get axis and angle
    s = aav.norm(p=2, dim=-1, keepdim=True) # Angle
    n = aav / s.clamp(min=1e-12).expand_as(aav) # Axis

    # Compute quaternion
    ss, cs = torch.sin(s/2), torch.cos(s/2)
    xyz = n * ss.expand_as(n) # scale axis by sin(th/2)

    # Return
    return torch.cat([xyz, cs], -1)

## Axis angle product (a1 * a2)
def AxisAngleProduct(a1, a2):
    # Check dimensions
    assert (a1.size(-1) == 3)
    assert (a1.is_same_size(a2))
    a1v, a2v = a1.contiguous(), a2.contiguous()

    # Get axis and angle values
    s1, s2 = a1v.norm(p=2, dim=-1, keepdim=True), \
             a2v.norm(p=2, dim=-1, keepdim=True) # Angle 1 & 2
    n1, n2 = a1v / s1.clamp(min=1e-12).expand_as(a1v), \
             a2v / s2.clamp(min=1e-12).expand_as(a2v) # Axis 1 & 2

    # Compute cos/sin
    cs1, cs2, ss1, ss2 = torch.cos(s1/2), torch.cos(s2/2), \
                         torch.sin(s1/2), torch.sin(s2/2)

    # Compute the composition
    # From: https://math.stackexchange.com/questions/382760/composition-of-two-axis-angle-rotations
    cos_s   = (cs1 * cs2) - (ss1 * ss2 * (n1 * n2).sum(dim=-1, keepdim=True))
    sin_s_n = (ss1 * cs2).expand_as(n1) * n1 + \
              (cs1 * ss2).expand_as(n2) * n2 + \
              (ss1 * ss2).expand_as(n1) * n1.cross(n2, dim=-1) # n1 x n2

    # Compute the resulting axis and angle
    sin_s   = sin_s_n.norm(p=2, dim=-1, keepdim=True) # sin(\gamma/2)
    n       = sin_s_n / sin_s.clamp(min=1e-12).expand_as(sin_s_n) # Get the axis
    s       = 2 * torch.atan2(sin_s, cos_s) # atan2 gives \gamma/2

    return s.expand_as(n)*n # return angle * axis

## Rotate a 3D point by an axis-angle transform
## From: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
def AxisAnglePointRotation(aa, pt):
    # Check dimensions
    assert (aa.size(-1) == 3 and pt.size(-1) == 3)
    assert (aa.is_same_size(pt))
    aav, ptv = aa.contiguous(), pt.contiguous()

    # Compute the axis and the angle
    s = aav.norm(p=2, dim=-1, keepdim=True)  # Angle
    n = aav / s.clamp(min=1e-12).expand_as(aav)  # Axis

    # Compute the transformed point according to Rodriguez's formula
    ss, cs = torch.sin(s), torch.cos(s)
    ptrot = ptv * cs.expand_as(ptv) + \
            n.cross(ptv, dim=-1) * ss.expand_as(ptv) + \
            n * ((n * ptv).sum(-1, keepdim=True) * (1. - cs)).expand_as(n) # Rodriguez's formula

    return ptrot

## Axis Angle inverse (simplest thing ever)
def AxisAngleInverse(aa):
    # Check dimensions
    assert (aa.size(-1) == 3)
    return -aa # Just put a minus sign!

## Compose two SE3-AA transforms: [tx, ty, tz, aax, aay, aaz]
def ComposeSE3AxisAnglePair(a, b):
    # Check dimensions
    assert (a.is_same_size(b))
    assert (a.size(-1) == 6)
    av, bv = a.contiguous(), b.contiguous()

    # Init for FWD pass (se3 = [tx, ty, tz, aax, aay, aaz])
    ta, tb   = av.narrow(-1, 0, 3), bv.narrow(-1, 0, 3)
    aaa, aab = av.narrow(-1, 3, 3), bv.narrow(-1, 3, 3)

    # Compose axis-angle transforms
    aa = AxisAngleProduct(aaa, aab)

    # Apply axis-angle rotation to the center
    t = AxisAnglePointRotation(aaa, tb) + ta  # R1 * t2 + t1

    # Return
    return torch.cat([t, aa], -1)  # [tx, ty, tz, aax, aay, aaz]

## Invert SE3-AA transform: [tx, ty, tz, aax, aay, aaz]
def SE3AxisAngleInverse(a):
    # Check dimensions
    assert (a.size(-1) == 6)
    av = a.contiguous()

    # Init for FWD pass (se3 = [tx, ty, tz, aax, aay, aaz])
    t, aa = av.narrow(-1, 0, 3), av.narrow(-1, 3, 3)

    # Get inverse of axis-angle transform
    aainv = AxisAngleInverse(aa)

    # Get inverted translation
    tinv  = AxisAnglePointRotation(aainv, -t)

    # Return
    return torch.cat([tinv, aainv], -1)  # [tx, ty, tz, aax, aay, aaz]

def SE3AxisAngleToSE3Quat(a):
    # Check dimensions
    assert (a.size(-1) == 6)
    av = a.contiguous()

    # Init for FWD pass (se3 = [tx, ty, tz, aax, aay, aaz])
    t, aa = av.narrow(-1, 0, 3), av.narrow(-1, 3, 3)

    # Convert aa to quaternion
    q = AxisAngleToQuaternion(aa)

    # Return
    return torch.cat([t, q], -1)
