from scripts.annotate_with_openrouter import call_openrouter


def test_call_openrouter_dry_run():
    messages = [{'role': 'user', 'content': 'Improve this bullet point: increased sales by 20%.'}]
    resp, err, meta = call_openrouter(messages, api_keys=['fake_key'], dry_run=True)
    assert err is None
    assert isinstance(resp, dict)
    assert 'choices' in resp
    assert isinstance(meta, dict)
    assert meta.get('est_input_tokens', 0) >= 1
