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

	def joint(ttotal, tobs=tobs):
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

	def joint(ttotal, tobs=tobs):
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

	thetavals.append(theta_max)
	postvals = [joint(theta)/float(Z) for theta in thetavals]

	return thetavals, postvals


"""
write this function yourself. 
- generate plots showing posteriors and (predictive) medians for a range of observed lifespans, for both priors. 
- room for interpretation: how many observed lifespans you plot and which
  - up to you as long as you give us a sense that you've sufficiently explored the model and its prediction (as evident by your plots and the conclusions you draw from them)
"""

"""
- generate plots showing posteriors and predictive medians for a range of observed lifespans. Submit these plots
- comment on how the way the prediction changes as the observation approaches the mean expected lifespan compares to your own intuitions.

one plot per lifespan value choice and joint distribution choice (power law or Gaussian) that shows the posterior distribution, with the median of that distribution marked in the plot.

use 75 (the mean of the Gaussian prior in 2e) as the mean expected lifespan
"""

def find_median_index(discr_distr, step_size):
	s = 0
	i = 0
	while s < 0.5:
		s += discr_distr[i]*step_size
		i += 1
	return i

def opt_predictions_plot(integrating_func, build_joint_func, theta_min, theta_max):
	# plots showing the posteriors and predictive medians
	tobs_vals = [10, 50, 100, 200]

	for tobs in tobs_vals:

		joint = build_joint_func(tobs)
		step_size = 0.01
		num_steps = int((theta_max-theta_min)/step_size)
		thetavals, postvals = integrating_func(joint, theta_min, theta_max, num_steps)
		median = thetavals[find_median_index(postvals, step_size)]

		plt.plot(thetavals, postvals, label="posterior")
		plt.axvline(median, label="predictive median: {0}".format(median), color="r")

		plt.title("Posterior and Predictive Median")
		plt.ylabel("posterior density")
		plt.xlabel("theta")
		plt.legend()
		plt.show()


# theta_min = 0 and theta_max = 300

opt_predictions_plot(opt_compute_posterior, opt_build_powerlaw_joint, 0, 300)
opt_predictions_plot(opt_compute_posterior, opt_build_lifespan_joint, 0, 300)
