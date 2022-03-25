import copy

import numpy as np
import scipy.linalg
import scipy.stats
import torch
from matplotlib import pyplot as plt

# from ray import tune
# from ray.tune.suggest import ConcurrencyLimiter
# from ray.tune.suggest.hyperopt import HyperOptSearch
# import tqdm
from torch import nn
from torch.nn import functional as F

import autonomous_optimizer
import sys
sys.path.append('./SEGY_Wrapper/.')
import SEGY_wrapper as wrap
sys.path.append('./bruges/.') 
from bruges.reflection import reflection as avo
from bruges.filters import wavelets as wav 
from torch.nn.functional import conv1d

class Variable(nn.Module):
    """A wrapper to turn a tensor of parameters into a module for optimization."""

    def __init__(self, data: torch.Tensor):
        """Create Variable holding `data` tensor."""
        super().__init__()
        self.x = nn.Parameter(data)

def avoObj():
    """
    avo objective function
    """
    x0 = np.load('initial.npy')
    x0 = torch.tensor(x0[0,:])
    x0.requires_grad = True
    
    print("x0 :", x0)
    print("x0 shape :", np.shape(x0))
    
    dobs = torch.tensor(np.load('seis.npy'))
    dobs = torch.transpose(dobs,0,1)
    dobs = torch.unsqueeze(dobs,0)
    #dobs.requires_grad = True
    #print("dobs shape :", np.shape(dobs))
    
    opval = np.load('true.npy')
    
    wavelet = wav.ricker(0.06,2e-3,30)
    wavelet = wavelet*100
    wavelet = torch.tensor(wavelet).unsqueeze(dim=0).unsqueeze(dim=0).float()
    
    def avof(var):
        x = var.x
        #x = torch.transpose(x,0,1)
        #print("shape of x :", np.shape(x))
        
        zpall = torch.unsqueeze(x,1)
        tr1 = zpall*0
        reflectivity = zpall[:-1,:]*0
        
        #print("shape of zpall :", np.shape(zpall))
        for i in range(zpall.shape[1]):
            zp = zpall[:,i]
            zp2 = zp[1:]
            zp1 = zp[:-1]
            reflect = (zp1 - zp2)/(zp2 + zp1)
            reflectivity[:,i] = reflect
            
            reflect = reflect.unsqueeze(dim=0).float()
            reflect = torch.unsqueeze(reflect,dim=0)
            
            #print("reflect device :", reflect.get_device())
            #wavelet = wavelet.cuda(reflect.get_device())
            synth = conv1d(reflect, wavelet, padding=int(wavelet.shape[-1] / 2))
            
        #print("shape of synth :", synth)
        return F.mse_loss(dobs,synth)
        # tr1 = zpall*0
        # reflectivity = zpall[:-1,:]*0
        
        # for i in range(zpall.shape[1]):
        #     zp = zpall[:,i]
        #     zp2 = zp[1:]
        #     zp1 = zp[:-1]
        #     reflect = (zp1 - zp2)/(zp2 + zp1)
        #     reflectivity[:,i] = reflect
        #     wavelet = wav.ricker(0.06, 2e-3, 30)
        #     wavelet = wavelet*100
        #     wavelet = torch.tensor(wavelet).unsqueeze(dim=0).unsqueeze(dim=0).float()
        #     reflect = torch.tensor(reflect).unsqueeze(dim=0).float()
        #     reflect = torch.unsqueeze(reflect,dim=0)
        #     synth = conv1d(reflect, wavelet, padding=int(wavelet.shape[-1] / 2))
        
            
    iv = avof(Variable(x0))
    print("iv :", iv)
    
    return {
        "model0": Variable(x0),
        "obj_function": avof,
        "iv": iv,
        "dobs" : dobs,
        "optimal_val": opval,
     }

def convex_quadratic():
    """
    Generate a symmetric positive semidefinite matrix A with eigenvalues
    uniformly in [1, 30].

    """
    num_vars = 2

    # First generate an orthogonal matrix (of eigenvectors)
    eig_vecs = torch.tensor(
        scipy.stats.ortho_group.rvs(dim=(num_vars)), dtype=torch.float
    )
    # Now generate eigenvalues
    eig_vals = torch.rand(num_vars) * 29 + 1

    A = eig_vecs @ torch.diag(eig_vals) @ eig_vecs.T
    b = torch.normal(0, 1 / np.sqrt(num_vars), size=(num_vars,))

    x0 = torch.normal(0, 0.5 / np.sqrt(num_vars), size=(num_vars,))

    def quadratic(var):
        x = var.x
        return 0.5 * x.T @ A @ x + b.T @ x

    optimal_x = scipy.linalg.solve(A.numpy(), -b.numpy(), assume_a="pos")
    optimal_val = quadratic(Variable(torch.tensor(optimal_x))).item()

    return {
        "model0": Variable(x0),
        "obj_function": quadratic,
        "optimal_x": optimal_x,
        "optimal_val": optimal_val,
        "A": A.numpy(),
        "b": b.numpy(),
    }


def rosenbrock():
    num_vars = 2

    # Initialization strategy: x_i = -2 if i is even, x_i = +2 if i is odd
    x0 = torch.tensor([-1.5 if i % 2 == 0 else 1.5 for i in range(num_vars)])
    
    print("x0 :", x0)
    print("shape of x0:", np.shape(x0))

    def rosen(var):
        x = var.x
        return torch.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)

    # Optimum at all x_i = 1, giving f(x) = 0
    optimal_x = np.ones(num_vars)
    optimal_val = 0

    return {
        "model0": Variable(x0),
        "obj_function": rosen,
        "optimal_x": optimal_x,
        "optimal_val": optimal_val,
    }


def logistic_regression():
    num_vars = 3

    g0 = torch.distributions.multivariate_normal.MultivariateNormal(
        loc=torch.randn(num_vars),
        scale_tril=torch.tril(torch.randn((num_vars, num_vars))),
    )
    g1 = torch.distributions.multivariate_normal.MultivariateNormal(
        loc=torch.randn(num_vars),
        scale_tril=torch.tril(torch.randn((num_vars, num_vars))),
    )

    x = torch.cat([g0.sample((50,)), g1.sample((50,))])
    y = torch.cat([torch.zeros((50,)), torch.ones((50,))])
    perm = torch.randperm(len(x))
    x = x[perm]
    y = y[perm]

    model0 = nn.Sequential(nn.Linear(num_vars, 1), nn.Sigmoid())

    def obj_function(model):
        y_hat = model(x).view(-1)
        weight_norm = model[0].weight.norm()
        return F.binary_cross_entropy(y_hat, y) + 5e-4 / 2 * weight_norm

    return {"model0": model0, "obj_function": obj_function, "data": (x, y)}


def robust_linear_regression():
    num_vars = 3

    # Create four gaussian distributions with random mean and covariance.
    # For all points drawn from the same gaussian, their labels are
    # generated by projecting them along the same random vector, adding
    # the same random bias, and perturbing them with i.i.d. gaussian noise.
    x = []
    y = []
    for _ in range(4):
        gaussian = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=torch.randn(num_vars),
            scale_tril=torch.tril(torch.randn((num_vars, num_vars))),
        )
        new_points = gaussian.sample((25,))

        # y_i = true_vector `dot` x_i + true_bias + noise
        true_vector = torch.randn(num_vars)
        true_bias = torch.randn(1)
        new_labels = new_points @ true_vector + true_bias + torch.randn(25)

        x.append(new_points)
        y.append(new_labels)

    x = torch.cat(x)
    y = torch.cat(y)

    # Shuffle the dataset
    perm = torch.randperm(len(x))
    x = x[perm]
    y = y[perm]

    model0 = nn.Linear(num_vars, 1)

    def geman_mcclure(model):
        y_hat = model(x).view(-1)
        squared_errors = (y - y_hat) ** 2
        return (squared_errors / (1 + squared_errors)).mean()

    return {"model0": model0, "obj_function": geman_mcclure}


def mlp():
    num_vars = 2

    # Create four gaussian distributions with random mean and covariance
    gaussians = [
        torch.distributions.multivariate_normal.MultivariateNormal(
            loc=torch.randn(num_vars),
            scale_tril=torch.tril(torch.randn((num_vars, num_vars))),
        )
        for _ in range(4)
    ]

    # Randomly assign each of the four gaussians a 0-1 label
    # Do again if all four gaussians have the same label (don't want that)
    gaussian_labels = np.zeros((4,))
    while (gaussian_labels == 0).all() or (gaussian_labels == 1).all():
        gaussian_labels = torch.randint(0, 2, size=(4,))

    # Generate a dataset of 100 points with 25 points drawn from each gaussian
    # Label of the datapoint is the same as the label of the gaussian it came from
    x = torch.cat([g.sample((25,)) for g in gaussians])
    y = torch.cat([torch.full((25,), float(label)) for label in gaussian_labels])
    perm = torch.randperm(len(x))
    x = x[perm]
    y = y[perm]

    model0 = nn.Sequential(
        nn.Linear(num_vars, 2), nn.ReLU(), nn.Linear(2, 1), nn.Sigmoid()
    )

    def obj_function(model):
        y_hat = model(x).view(-1)
        weight_norm = model[0].weight.norm() + model[2].weight.norm()
        return F.binary_cross_entropy(y_hat, y) + 5e-4 / 2 * weight_norm

    return {"model0": model0, "obj_function": obj_function, "dataset": (x, y)}


def run_optimizer(make_optimizer, problem, iterations, hyperparams):
    # Initial solution
    model = copy.deepcopy(problem["model0"])
    obj_function = problem["obj_function"]

    # Define optimizer
    optimizer = make_optimizer(model.parameters(), **hyperparams)

    # We will keep track of the objective values and weight trajectories
    # throughout the optimization process.
    values = []
    trajectory = []

    # Passed to optimizer. This setup is required to give the autonomous
    # optimizer access to the objective value and not just its gradients.
    def closure():
        trajectory.append(copy.deepcopy(model))
        optimizer.zero_grad()

        obj_value = obj_function(model)
        obj_value.backward()

        values.append(obj_value.item())
        return obj_value

    # Minimize
    for i in range(iterations):
        optimizer.step(closure)

        # Stop optimizing if we start getting nans as objective values
        if np.isnan(values[-1]) or np.isinf(values[-1]):
            break

    return np.nan_to_num(values, 1e6), trajectory


def accuracy(model, x, y):
    return ((model(x).view(-1) > 0.5) == y).float().mean().item()


def run_all_optimizers(problem, iterations, tune_dict, policy):
    # SGD
    sgd_vals, sgd_traj = run_optimizer(
        torch.optim.SGD, problem, iterations, tune_dict["sgd"]["hyperparams"]
    )
    print(f"SGD best loss: {sgd_vals.min()}")

    # Momentum
    momentum_vals, momentum_traj = run_optimizer(
        torch.optim.SGD, problem, iterations, tune_dict["momentum"]["hyperparams"]
    )
    print(f"Momentum best loss: {momentum_vals.min()}")

    # Adam
    adam_vals, adam_traj = run_optimizer(
        torch.optim.Adam, problem, iterations, tune_dict["adam"]["hyperparams"]
    )
    print(f"Adam best loss: {adam_vals.min()}")

    # LBFGS
    lbfgs_vals, lbfgs_traj = run_optimizer(
        torch.optim.LBFGS, problem, iterations, tune_dict["lbfgs"]["hyperparams"]
    )
    print(f"LBFGS best loss: {lbfgs_vals.min()}")

    # Autonomous optimizer
    ao_vals, ao_traj = run_optimizer(
        autonomous_optimizer.AutonomousOptimizer,
        problem,
        iterations,
        {"policy": policy},
    )
    print(f"Autonomous Optimizer best loss: {ao_vals.min()}")

    return {
        "sgd": (sgd_vals, sgd_traj),
        "momentum": (momentum_vals, momentum_traj),
        "adam": (adam_vals, adam_traj),
        "lbfgs": (lbfgs_vals, lbfgs_traj),
        "ao": (ao_vals, ao_traj),
    }


def plot_trajectories(trajectories, problem, get_weights, set_weights):
    """Plot optimization trajectories on top of a contour plot.

    Parameters:
        trajectories (List(nn.Module))
        problem (dict)
        get_weights (Callable[[], Tuple[float, float]])
        set_weights (Callable[[float, float], None])

    """
    data = {}
    for name, traj in trajectories.items():
        data[name] = np.array([get_weights(model) for model in traj])

    xmin = min(np.array(d)[:, 0].min() for d in data.values())
    ymin = min(np.array(d)[:, 1].min() for d in data.values())
    xmax = max(np.array(d)[:, 0].max() for d in data.values())
    ymax = max(np.array(d)[:, 1].max() for d in data.values())

    X = np.linspace(xmin - (xmax - xmin) * 0.2, xmax + (xmax - xmin) * 0.2)
    Y = np.linspace(ymin - (ymax - ymin) * 0.2, ymax + (ymax - ymin) * 0.2)

    model = copy.deepcopy(problem["model0"])
    Z = np.empty((len(Y), len(X)))
    for i in range(len(X)):
        for j in range(len(Y)):
            set_weights(model, X[i], Y[j])
            Z[j, i] = problem["obj_function"](model)

    plt.figure(figsize=(10, 6), dpi=500)
    plt.contourf(X, Y, Z, 30, cmap="RdGy")
    plt.colorbar()

    for name, traj in data.items():
        plt.plot(traj[:, 0], traj[:, 1], label=name)
        print("name :", name)
        print("traj[:,0] :", traj[:,0])
        print("traj[:,1] :", traj[:,1])
        
    
    plt.title("Convex Quadratic Trajectory Plot")
    plt.plot(*get_weights(problem["model0"]), "go")
    plt.legend()

    plt.plot()
    plt.show()

def get_trajectories(trajectories, problem, get_weights, set_weights):
    
    data = {}
    for name, traj in trajectories.items():
        data[name] = np.array([get_weights(model) for model in traj])
        
    for name, traj in data.items():
        print("shape of traj :", np.shape(traj))
        
    

'''def tune_algos(
    dataset,
    algo_iters,
    tune_iters,
    hyperparam_space,
    algos=["sgd", "momentum" "adam", "lbfgs"],
):
    """Tune hyperparameters with Bayesian optimization."""

    def make_experiment(make_optimizer):
        def experiment(hyperparams):
            best_obj_vals = []
            for problem in dataset:
                vals, traj = run_optimizer(
                    make_optimizer, problem, algo_iters, hyperparams
                )
                best_obj_vals.append(vals.min())

            tune.report(objective_value=np.mean(best_obj_vals))

        return experiment

    results = {}
    for algo in tqdm.tqdm(algos):

        if algo == "sgd":

            sgd_analysis = tune.run(
                make_experiment(torch.optim.SGD),
                config={"lr": hyperparam_space["lr"]},
                metric="objective_value",
                mode="min",
                search_alg=ConcurrencyLimiter(HyperOptSearch(), max_concurrent=3),
                num_samples=tune_iters,
                verbose=0,
            )
            sgd_hyperparams = sgd_analysis.get_best_config(
                metric="objective_value", mode="min"
            )

            results["sgd"] = {"analysis": sgd_analysis, "hyperparams": sgd_hyperparams}

        if algo == "momentum":

            momentum_analysis = tune.run(
                make_experiment(torch.optim.SGD),
                config={
                    "nesterov": True,
                    "lr": hyperparam_space["lr"],
                    "momentum": hyperparam_space["momentum"],
                },
                metric="objective_value",
                mode="min",
                search_alg=ConcurrencyLimiter(HyperOptSearch(), max_concurrent=3),
                num_samples=tune_iters,
                verbose=0,
            )
            momentum_hyperparams = momentum_analysis.get_best_config(
                metric="objective_value", mode="min"
            )

            results["momentum"] = {
                "analysis": momentum_analysis,
                "hyperparams": momentum_hyperparams,
            }

        if algo == "adam":

            adam_analysis = tune.run(
                make_experiment(torch.optim.Adam),
                config={"lr": hyperparam_space["lr"]},
                metric="objective_value",
                mode="min",
                search_alg=ConcurrencyLimiter(HyperOptSearch(), max_concurrent=3),
                num_samples=tune_iters,
                verbose=0,
            )
            adam_hyperparams = adam_analysis.get_best_config(
                metric="objective_value", mode="min"
            )

            results["adam"] = {
                "analysis": adam_analysis,
                "hyperparams": adam_hyperparams,
            }

        if algo == "lbfgs":

            lbfgs_analysis = tune.run(
                make_experiment(torch.optim.LBFGS),
                config={"lr": hyperparam_space["lr"], "max_iter": 1},
                metric="objective_value",
                mode="min",
                search_alg=ConcurrencyLimiter(HyperOptSearch(), max_concurrent=3),
                num_samples=tune_iters,
                verbose=0,
            )
            lbfgs_hyperparams = lbfgs_analysis.get_best_config(
                metric="objective_value", mode="min"
            )

            results["lbfgs"] = {
                "analysis": lbfgs_analysis,
                "hyperparams": lbfgs_hyperparams,
            }

    return results'''
