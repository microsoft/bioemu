
from typing import List, Tuple
import fire

def hello(name):
	print(type(name))
	print(type(name[0]))
	for i in name:
		print(i, type(i))

if __name__ == '__main__':
  fire.Fire(hello)