import sys, os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, BASE_DIR)

def test(number):
    number /= 100
    number_str = f"{number:.2f}".replace('.', '')
    number_list = list(map(int, number_str))
    print(number_list)

test(0)
test(199)
test(200)
