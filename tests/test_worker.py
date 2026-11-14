from spotpilot.worker import demo

def test_real_worker_resume_matches_reference(tmp_path):
    result=demo(tmp_path/'worker')
    assert result['success'] and result['resumed_from_step']==15
    assert result['recomputed_steps']==2 and result['total_steps']==40
