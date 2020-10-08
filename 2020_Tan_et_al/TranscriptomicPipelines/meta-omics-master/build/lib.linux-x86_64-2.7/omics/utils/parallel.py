"""utility module enabling lazy parallelization among cores"""

from multiprocessing import Pool
from multiprocessing import cpu_count

def parallel_jobs(func, args):
	"""return a list of concatinated results processed by multiple cores
	Input:
		func: 	a callable function
		args:	a list of parameters
	Output:
		res:	a list of result"""
	num_cores = cpu_count()
	with Pool(num_cores) as p:
		res = p.starmap(func, args)
	return res
