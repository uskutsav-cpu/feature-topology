import json

from scripts.certification_statistics import summarize_certification


def test_repository_certification_summary_preserves_evidence_levels(tmp_path):
    result=summarize_certification('.',tmp_path)
    assert result['analytic_exact_controls']==7
    assert result['rational_trained_layer_certificates']==40
    assert result['exact_trained_checkpoints']==20
    assert result['numerical_polygon_layer_diagnostics']==80
    assert result['lean_verified'] is True
    scope=json.loads((tmp_path/'certification_scope.json').read_text())
    assert 'not a smooth-continuum proof' in scope['numerical_scope']
