import numpy as np
import matplotlib.pyplot as plt

def compute_gaussian_likelihoods(x, means, stddevs):
    coeff = 1/(np.sqrt(2 * np.pi) * stddevs)
    diffs = x[:, None] - means[None, :]
    expargs = (diffs**2) / (2 * stddevs**2)[None, :]
    likelihoods = coeff[None, :]*np.exp(-expargs)
    return likelihoods

def compute_gmm_likelihoods(x, priors, means, stddevs):
    gaussian_likelihoods = compute_gaussian_likelihoods(x, means, stddevs)
    gmm_likelihoods = (priors[None, :] * gaussian_likelihoods).sum(axis=1)
    return gmm_likelihoods

def compute_responsibilities(x, priors, means, stddevs):
    scores = priors[None, :]*compute_gaussian_likelihoods(x, means, stddevs)
    responsibilities = scores/scores.sum(axis=1)[:, None]
    return responsibilities

def sample_gmm(priors, means, stddevs, n):
    z_sample = np.random.randn(n)

    # choose the mode for each sample according to the priors
    idx_modes = np.random.choice(np.arange(len(priors)), size=n, p=priors)
    return stddevs[idx_modes]*z_sample + means[idx_modes]



if __name__ == "__main__":

    np.random.seed(42)

    priors_old = [0.5, 0.5]
    means_old = [-1, +1]
    stddevs_old = [1, 1]

    x = [-5, -3, -1, 2.5, 3.5]

    priors_old = np.asarray(priors_old)
    means_old = np.asarray(means_old)
    stddevs_old = np.asarray(stddevs_old)
    x = np.asarray(x)

    # E step

    responsibilities = compute_responsibilities(x, priors_old, means_old, stddevs_old)
    print("Responsibilities:")
    print(responsibilities)


    # M step

    Nks = responsibilities.sum(axis=0)

    means_new = (responsibilities * x[:, None]).sum(axis=0) / Nks
    stddevs_new = np.sqrt((responsibilities * (x[:, None] - means_new[None, :])**2 ).sum(axis=0) / Nks)
    priors_new = Nks/Nks.sum()

    print("Updated means:")
    print(means_new)

    print("Updated standard deviations:")
    print(stddevs_new)

    print("Updated priors:")
    print(priors_new)

    # Log-likelihood comparison

    likelihood_old = compute_gmm_likelihoods(x, priors_old, means_old, stddevs_old).sum()
    likelihood_new = compute_gmm_likelihoods(x, priors_new, means_new, stddevs_new).sum()
    print("Old log-likelihood:", np.log(likelihood_old))
    print("New log-likelihood:", np.log(likelihood_new))

    # Plots

    xrange = np.arange(-8, 8, 0.1)
    old_likelihoods = compute_gaussian_likelihoods(xrange, means_old, stddevs_old)
    old_gmm_likelihoods = compute_gmm_likelihoods(xrange, priors_old, means_old, stddevs_old)

    plt.figure()
    for i, likelihood in enumerate(old_likelihoods.T):
        plt.plot(xrange, likelihood, label=f'mode {i} of GMM (old)')
    plt.plot(xrange, old_gmm_likelihoods, '-', color='gray', label='Overall GMM (old)')
    plt.plot(x, np.zeros_like(x), 'x', color='black', label='samples')
    plt.xlim([-8, 8])
    plt.legend()
    plt.grid(linestyle=':')
    plt.draw()    

    xrange = np.arange(-8, 8, 0.1)
    new_likelihoods = compute_gaussian_likelihoods(xrange, means_new, stddevs_new)
    new_gmm_likelihoods = compute_gmm_likelihoods(xrange, priors_new, means_new, stddevs_new)
    plt.figure()
    for i, likelihood in enumerate(new_likelihoods.T):
        plt.plot(xrange, likelihood, label=f'mode {i} of GMM (new)')
    plt.plot(xrange, new_gmm_likelihoods, '-', color='gray', label='Overall GMM (new)')
    plt.plot(x, np.zeros_like(x), 'x', color='black', label='samples')
    plt.xlim([-8, 8])
    plt.legend()
    plt.grid(linestyle=':')
    plt.draw()


    x_sample = sample_gmm(priors_new, means_new, stddevs_new, n=1000)

    plt.figure()
    plt.hist(x_sample, bins=20, range=(-8, 8), density=True, color='C0', label='Sampled points')
    plt.plot(xrange, new_gmm_likelihoods, '-', color='black', label='Overall GMM (new)')
    plt.legend()
    plt.grid(linestyle=':')
    plt.xlim([-8, 8])
    plt.draw()

    plt.show()



