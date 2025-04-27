from adacth import adact_h, generate_g, check_transition, generate_chi, l_x_distance, getprobs
import itertools

def test_hello_world():
    assert 1 + 1 == 2

def test_open():
    with open(r'tests\tmaze25x5x2test.txt', 'r') as file:
        content = file.read()
        print(content)
        assert content is not None

def test_generate_chi():
    l = 5
    g = 17
    chi = generate_chi(l, g)
    assert len(chi) == l
    for i in range(l):
        x = list(itertools.repeat([c for c in range(g)], times = i+1))
        print(x)
        print(itertools.product(*x))
        assert len(chi[i]) == g**(i+1)
        