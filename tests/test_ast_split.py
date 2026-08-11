from code_jepa.ast_split import extract_functions


SRC = '''
def add(a, b):
    """Add two numbers."""
    x = a + b
    y = x * 2
    return y

def too_short(x):
    return x
'''


def test_extract_skips_short_and_keeps_doc():
    recs = extract_functions(SRC, repo="demo", path="a.py", min_body_stmts=3)
    assert len(recs) == 1
    assert recs[0].func_name == "add"
    assert recs[0].docstring.startswith("Add two numbers")
    assert "return y" in recs[0].body_text
