import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import comb

# Part 1b
data_h = [3, 1, 5, 4, 8, 0, 10, 21, 26]
data_t = [2, 4, 0, 6, 2, 10, 16, 5, 0]

def log_lhood_ratio(h, t, prior_odds):
	pdh1 = 1.0/2**(h+t)
	pdh2 = sum([(i**h)*((1.0-i)**t) for i in np.linspace(0, 1, 101)])/100.0
	return np.log(pdh1/pdh2)+np.log(prior_odds)

def rating(post, a, b):
	norm = 1.0/(1.0+np.exp(b-a*post))
	return (1-norm)*6+1 # invert, scale

i = 0
ab_vals = [(2, 0), (1, 0), (5, 0), (2, 1), (5, 1), (1, 1)]
a, b = ab_vals[i]

# Control
prior_odds = 1
post_log_lhood_c = []
ratings_c = []
for i in range(len(data_h)):
	h = data_h[i]
	t = data_t[i]
	l = log_lhood_ratio(h, t, prior_odds)
	post_log_lhood_c.append(l)
	ratings_c.append(rating(l, a, b))

human_ratings_c = [2, 4, 6, 2, 6, 6, 3, 6, 7]
"""
plt.scatter(human_ratings_c, ratings_c)
plt.title("Human vs Model Ratings; Control")
plt.ylabel("model")
plt.ylim(0.5, 7.5)
plt.xlabel("human")
plt.xlim(0.5, 7.5)
plt.legend()
plt.show()
"""
print np.corrcoef(human_ratings_c, ratings_c)[0][1]

# Modified
prior_odds = 0.5
post_log_lhood_m = []
ratings_m = []
for i in range(len(data_h)):
	h = data_h[i]
	t = data_t[i]
	l = log_lhood_ratio(h, t, prior_odds)
	post_log_lhood_m.append(l)
	ratings_m.append(rating(l, a, b))

human_ratings_m = [1, 1, 1, 1, 1, 4, 7, 4, 6]
"""
plt.scatter(human_ratings_m, ratings_m)
plt.title("Human vs Model Ratings; Modified")
plt.ylabel("model")
plt.ylim(0.5, 7.5)
plt.xlabel("human")
plt.xlim(0.5, 7.5)
plt.legend()
plt.show()
"""
print np.corrcoef(human_ratings_m, ratings_m)[0][1]

plt.scatter(human_ratings_c, ratings_c, label="control", color="b")
plt.scatter(human_ratings_m, ratings_m, label="modified", color="r")
plt.title("Human vs Model Ratings")
plt.ylabel("model")
plt.ylim(0.5, 7.5)
plt.xlabel("human")
plt.xlim(0.5, 7.5)
plt.legend()
plt.show()
