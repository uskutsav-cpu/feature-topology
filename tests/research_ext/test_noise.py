import numpy as np
import pytest
from research_ext.noise import noise_curve


def fixture():
    rng=np.random.default_rng(9)
    a=rng.uniform(0,2*np.pi,200);b=rng.uniform(0,2*np.pi,100)
    x=np.column_stack([np.cos(a),np.sin(a)]);y=np.column_stack([np.cos(b),np.sin(b)])
    return x,y,a,b


def test_clean_information_and_noise_fragility():
    result=noise_curve(*fixture(),levels=[0.,.1,10.],repeats=3)
    assert result['rows'][0]['mean_angular_cosine']>.99999
    assert result['rows'][-1]['mean_angular_cosine']<.4
    assert result['information_destruction_proved'] is False


def test_global_scale_and_translation_invariance():
    x,y,a,b=fixture();one=noise_curve(x,y,a,b,repeats=2)
    two=noise_curve(3*x+7,3*y+7,a,b,repeats=2)
    assert np.allclose([r['mean_angular_cosine'] for r in one['rows']],
                       [r['mean_angular_cosine'] for r in two['rows']])


def test_test_data_does_not_set_training_normalization():
    x,y,a,b=fixture();one=noise_curve(x,y,a,b,repeats=1)
    two=noise_curve(x,100*y,a,b,repeats=1)
    assert one['training_rms']==two['training_rms']


def test_constant_representation_not_imputed_as_success():
    x,y,a,b=fixture();result=noise_curve(x*0,y*0,a,b)
    assert result['status']=='constant_training_representation'
    assert not result['information_destruction_proved']


def test_invalid_noise_settings_rejected():
    with pytest.raises(ValueError):noise_curve(*fixture(),levels=[0.,-1.])
    with pytest.raises(ValueError):noise_curve(*fixture(),alpha=0.)
