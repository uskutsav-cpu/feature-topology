from fractions import Fraction as Q
from copy import deepcopy
import pytest
from research_ext.exact import (point, Layer, rational, stored_float, intersections,
    split_polygon, forward, graph_from_segments, polygon_certificate,
    verify_polygon_certificate, matmul, identity, left_inverse, certify_box,
    verify_box_certificate)

P = point
SQUARE = [P([-1,-1]), P([1,-1]), P([1,1]), P([-1,1])]
def layer(w, b, relu=True):
    return Layer(tuple(P(r) for r in w), P(b), relu)

@pytest.mark.parametrize('x', [0.1, True, None, [], float('nan')])
def test_implicit_rationals_reject(x):
    with pytest.raises((ValueError, TypeError)):
        rational(x)

def test_exact_stored_float():
    assert stored_float(.1) == Q(3602879701896397,36028797018963968)
    with pytest.raises(ValueError):
        stored_float(float('inf'))

@pytest.mark.parametrize('segments,expected', [
    ((P([0,0]),P([2,2]),P([0,2]),P([2,0])), [(Q(1,2),Q(1,2))]),
    ((P([0]),P([2]),P([1]),P([3])), [(Q(1,2),Q(0)),(Q(1),Q(1,2))]),
    ((P([0]),P([2]),P([3]),P([1])), [(Q(1,2),Q(1)),(Q(1),Q(1,2))]),
    ((P([0,0]),P([0,0]),P([-1,0]),P([1,0])), [(Q(0),Q(1,2))]),
    ((P([0,0,0]),P([1,1,0]),P([0,1,1]),P([1,0,1])), []),
    ((P([0,0]),P([1,0]),P([0,1]),P([1,1])), []),
    ((P([0,0]),P([0,1]),P([0,2]),P([0,3])), []),
    ((P([1,1]),P([1,1]),P([1,1]),P([1,1])), [(Q(0),Q(0))]),
])
def test_intersections(segments,expected):
    assert intersections(*segments)==expected

@pytest.mark.parametrize('layers,beta,injective', [
    ([],1,True),
    ([layer([[1,0]],[0],False)],0,False),
    ([layer([[0,0]],[0],False)],0,False),
    ([layer([[1,0],[0,1]],[0,0],True)],1,False),
    ([layer([[3,0],[0,3]],[2,7],False)],1,True),
    ([layer([[0,-1],[1,0]],[0,0],False)],1,True),
    ([layer([[1,0],[-1,0],[0,1],[0,-1]],[0,0,0,0],True)],1,True),
])
def test_polygon_controls(layers,beta,injective):
    cert=polygon_certificate(SQUARE,layers)
    assert cert['graph']['beta0']==1
    assert cert['graph']['beta1']==beta
    assert cert['injective_on_polygon_subset']==injective
    assert verify_polygon_certificate(cert)
    assert cert['lean_verified'] is False
    if not injective:
        assert cert['collision']['x'] != cert['collision']['y']
        assert forward(P(cert['collision']['x']),layers)==forward(P(cert['collision']['y']),layers)

def test_graph_with_crossing_and_overlap():
    segments=[(P([-1,0]),P([1,0])),(P([0,-1]),P([0,1])),(P([0,0]),P([2,0]))]
    graph=graph_from_segments(segments)
    assert graph['beta0']==1 and graph['beta1']==0
    assert graph['edge_count']==5

def test_disconnected_graph():
    graph=graph_from_segments([(P([0]),P([1])),(P([2]),P([3]))])
    assert graph['beta0']==2 and graph['beta1']==0

def test_exact_small_separation_not_merged():
    epsilon=Q(1,10**20)
    graph=graph_from_segments([(P([0,0]),P([1,0])),(P([0,epsilon]),P([1,epsilon]))])
    assert graph['beta0']==2

def test_certificate_tamper():
    cert=polygon_certificate(SQUARE,[])
    other=deepcopy(cert); other['graph']['beta1']=0
    assert not verify_polygon_certificate(other)
    other=deepcopy(cert); other['lean_verified']=True
    assert not verify_polygon_certificate(other)

def test_split_midpoints_match_forward():
    layers=[layer([[1,0],[-1,0],[0,1],[0,-1]],[0,0,0,0]),
            layer([[1,-1,1,-1],[1,0,-1,0]],[Q(1,3),Q(-1,4)])]
    for piece in split_polygon(SQUARE,layers):
        t=(piece.t0+piece.t1)/2
        a,b=SQUARE[piece.edge],SQUARE[(piece.edge+1)%4]
        x=tuple(u+t*(v-u) for u,v in zip(a,b))
        assert forward(x,layers)==tuple((u+v)/2 for u,v in zip(piece.left,piece.right))

def test_segment_budget_fails_closed():
    with pytest.raises(ValueError):
        polygon_certificate(SQUARE,[],max_segments=2)
    with pytest.raises(ValueError):
        graph_from_segments([(p,p) for p in SQUARE],max_pairs=2)

@pytest.mark.parametrize('matrix', [
    [[1,0],[0,1]], [[1,2],[3,5],[7,11]], [[Q(1,3),0],[0,Q(1,7)]], [[1]],
])
def test_left_inverse(matrix):
    a=tuple(P(row) for row in matrix)
    b=left_inverse(a)
    assert b is not None
    assert matmul(b,a)==identity(len(matrix[0]))

def test_rank_deficient_not_injective_claim():
    assert left_inverse((P([1,2]),P([2,4]))) is None
    cert=certify_box([layer([[1,0]],[0],False)],P([-1,-1]),P([1,1]))
    assert cert['status']=='inconclusive_rank_deficient'

def test_box_certificate_and_tampering():
    layers=[layer([[1,0],[0,2]],[0,0]),layer([[1,1],[1,-1]],[2,4],False)]
    cert=certify_box(layers,P([1,1]),P([2,2]))
    assert cert['status']=='certified_injective_on_box'
    assert verify_box_certificate(cert)
    cert['left_inverse'][0][0]='999'
    assert not verify_box_certificate(cert)

def test_unstable_box_inconclusive():
    cert=certify_box([layer([[1]],[0])],P([-1]),P([1]))
    assert cert['status']=='inconclusive_activation_boundary'

def test_negative_box_rejected():
    with pytest.raises(ValueError):
        certify_box([],P([2]),P([1]))
