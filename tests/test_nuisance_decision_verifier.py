import copy
import json

import pytest

pytest.importorskip('flint')
from flint import ctx
from src.metrics.nuisance_decision_verifier import CenteredCircleClassifier,replay_decision,verify_decision


def classifier(violation=False,tie=False,gamma=.5,theta_numerator=1):
    ctx.prec=128
    hidden=([[0,0,1,0],[0,0,0,1]],[2,2],True)
    head=[[1,0],[0,0],[0,0],[0,0]]
    current=[hidden,(head,[-2 if violation else -1 if tie else 1,0,0,0],False)]
    initial=[hidden,([[0,0]]*4,[0,0,0,0],False)] if violation or tie else [hidden,(head,[0,0,0,0],False)]
    identity=[[int(i==j) for j in range(4)] for i in range(4)]
    return CenteredCircleClassifier(current,initial,identity,gamma,theta_numerator)


def test_complete_centered_classifier_preserves_affine_branch_cancellation():
    network=classifier()
    result=verify_decision(network,maximum_boxes=1000,seconds=60)
    assert result['decision_certificate'] is True and result['leaf_codes']==['']
    assert int(result['scaled_margin_lower_rational']['numerator'])/2**64>1.9
    assert replay_decision(network,json.loads(json.dumps(result)))
    # The current logits vary with phi; the complete centered prediction does not.
    assert result['target_class']==0


def test_wrong_target_at_one_point_is_retained_as_a_replayed_counterexample():
    network=classifier(violation=True)
    result=verify_decision(network,maximum_boxes=1000,seconds=60)
    assert result['decision_certificate'] is False
    assert result['status']=='target_class_violation'
    assert replay_decision(network,result)
    damaged=copy.deepcopy(result);damaged['violation_code']='00'
    with pytest.raises(ValueError,match='counterexample failed'):
        replay_decision(network,damaged)


def test_replay_rejects_overstated_margin_scope_target_and_incomplete_cover():
    network=classifier();result=verify_decision(network)
    damaged=copy.deepcopy(result);damaged['scaled_margin_lower_rational']['numerator']=str(10<<64)
    with pytest.raises(ValueError,match='exceeds'):replay_decision(network,damaged)
    for key,value in [('theta_fraction',[3,8]),('target_class',1),('scope','all inputs')]:
        damaged=copy.deepcopy(result);damaged[key]=value
        with pytest.raises(ValueError,match='scope changed'):replay_decision(network,damaged)
    damaged=copy.deepcopy(result);damaged['leaf_codes']=['0']
    with pytest.raises(ValueError,match='unresolved'):replay_decision(network,damaged)


def test_nonpositive_scaling_is_not_a_classifier_certificate():
    with pytest.raises(ValueError,match='finite and positive'):classifier(gamma=0)


def test_a_tie_at_one_circle_point_prevents_a_strict_correct_class_certificate():
    result=verify_decision(classifier(tie=True),maximum_boxes=1000,maximum_depth=8,seconds=60)
    assert result['decision_certificate'] is False and result['status']=='inconclusive'


def test_saved_state_encoding_matches_both_pytorch_branches_and_the_scaled_head():
    import math
    import numpy as np
    torch=pytest.importorskip('torch')
    from src.models.mlp import MLP,ScaledModel
    from scripts.remote_nuisance_decision_verifier import classifier_from_state

    ctx.prec=128;torch.manual_seed(73)
    config=dict(gamma=.5,dimension=4,width=6,depth=2,centered=True,
                manifold='torus',swap=False,relevance=0.)
    model=ScaledModel(MLP(dimension=4,width=6,depth=2),gamma=.5,centered=True)
    with torch.no_grad():
        model.network.hidden[0].weight.add_(.02)
        model.network.head.weight.add_(.03)
        model.network.head.bias.add_(torch.tensor([.1,.2,.3,.4]))
    state=dict(config=config,model={k:v.detach().clone() for k,v in model.state_dict().items()})
    encoded=classifier_from_state(state,{'config':config},np.eye(4,dtype=np.float32),1)
    model.double()
    for code in ['00','01','10','11']:
        theta=math.pi/4;phi=2*math.pi*(int(code,2)+.5)/2**len(code)
        point=torch.tensor([[math.cos(theta),math.sin(theta),math.cos(phi),math.sin(phi)]],dtype=torch.float64)
        values=model(point).detach().numpy()[0]
        for competitor,bound in encoded.margins(code,point=True).items():
            assert float(bound.mid())==pytest.approx(float(values[0]-values[competitor]),abs=1e-11)
    changed=dict(state,model={**state['model'],'unexpected.buffer':torch.zeros(1)})
    with pytest.raises(ValueError,match='Unsupported or incomplete'):
        classifier_from_state(changed,{'config':config},np.eye(4,dtype=np.float32),1)
    relevant=dict(config,relevance=.5)
    with pytest.raises(ValueError,match='unchanged centered primary torus'):
        classifier_from_state(dict(state,config=relevant),{'config':relevant},np.eye(4,dtype=np.float32),1)


def test_all_four_declared_targets_match_the_actual_primary_label_rule():
    import math
    import numpy as np
    pytest.importorskip('torch')
    from src.data.torus import labels
    for numerator in [1,3,5,7]:
        network=classifier(theta_numerator=numerator)
        latent=np.array([[2*math.pi*numerator/8,phi] for phi in [0,.7,2.4,5.9]])
        assert set(labels(latent).tolist())=={network.target_class}
