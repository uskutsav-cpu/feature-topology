from scripts.width_statistics import plot_width_gate


def gate():
    fits=[]
    for width,center,spread,delta in [(64,.5,2.5,-20.),(128,.5,4.,-22.),(256,.52,7.,-18.)]:
        fits.append({'width':width,'status':'ok','delta_aicc_smooth_minus_transition':delta,
                     'transition':{'center_gamma':center,
                                   'width_10_90_log2_gamma':spread}})
    return {'schema':'feature-topology.width-scaling-result.v1','status':'complete',
            'terminology':'crossover','phase_transition_language_allowed':False,
            'fits':fits}


def test_width_gate_figure_is_deterministic(tmp_path):
    plot_width_gate(gate(),tmp_path/'first')
    plot_width_gate(gate(),tmp_path/'second')
    for suffix in ['png','pdf','svg']:
        assert (tmp_path/f'first/finite_width_gate.{suffix}').read_bytes()==(
            tmp_path/f'second/finite_width_gate.{suffix}').read_bytes()
