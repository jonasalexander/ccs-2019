import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from functools import partial

GAMMA = 2.43

def opt_build_powerlaw_joint(tobs):
	"""
		returns an (anonymous) procedure of one argument - ttotal - that
		computes the joint density evaluated on that argument and tobs.

		TODO: construct and return anonymous procedure
	"""

	def joint(tobs, ttotal):
		if ttotal > tobs and tobs > 0:
			return ttotal**(-GAMMA-1)
		else:
			return 0.0

	return partial(joint, tobs=tobs)


def opt_build_lifespan_joint(tobs):
	"""
		TODO: model after build_powerlaw_joint
	"""

	# Gaussian prior (with mean 75 and standard deviation 16)

	def joint(tobs, ttotal):
		if ttotal > tobs and tobs > 0:
			return norm.pdf(ttotal, loc=75, scale=16)*1.0/ttotal
		else:
			return 0.0

	return partial(joint, tobs=tobs)



def opt_compute_posterior(joint, theta_min, theta_max, num_steps):
	"""
		Computes a table representation of the posterior distribution
		with at most num_steps joint density evaluations, covering the
		range from theta_min to theta_max.

		People interested in fancier integrators should feel free to
		modify the signature for this procedure, as well as its callers,
		as appropriate.

		TODO: compute Z along with an unnormalized table

		TODO: normalize joint

	"""

	step_size = float(theta_max-theta_min)/num_steps
	thetavals = [theta_min+i*step_size for i in range(num_steps)]

	# left Riemann sum
	Z = sum([joint(theta)*float(step_size) for theta in thetavals])

	theta_vals.append(theta_max)
	postvals = [joint(theta)/float(Z) for theta in thetavals]

	return thetavals, postvals










