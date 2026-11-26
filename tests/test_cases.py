

def test_a_repeated_case_id_is_made_unique():
    import json
    from margin_referee import cases
    text = "\n".join(json.dumps({"id": i, "input": "q", "expected": "yes"}) for i in "aab")
    assert [c.id for c in cases.parse(text).cases] == ["a", "a#2", "b"]


def test_a_renamed_id_never_collides_with_a_literal_one():
    import json
    from margin_referee import cases
    text = "\n".join(json.dumps({"id": i, "input": "q", "expected": "yes"}) for i in
                     ["a", "a", "a#2"])
    got = [c.id for c in cases.parse(text).cases]
    assert len(set(got)) == 3 and got[2] == "a#2"
