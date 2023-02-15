import torch
from torch import nn
from evotorch.decorators import pass_info
from evotorch.neuroevolution import GymNE
from evotorch.algorithms import PGPE
from evotorch.logging import StdOutLogger
from evotorch.algorithms import SNES
import custom_envs

# The decorator `@pass_info` used below tells the problem class `GymNE`
# to pass information regarding the gym environment via keyword arguments
# such as `obs_length` and `act_length`.
@pass_info
class CustomPolicy(torch.nn.Module):
    def __init__(self, obs_length: int, act_length: int, **kwargs):
        super().__init__()
        self.lin1 = torch.nn.Linear(obs_length, 32)
        self.act = torch.nn.Tanh()
        self.lin2 = torch.nn.Linear(32, act_length)

    def forward(self, data):
        return self.lin2(self.act(self.lin1(data)))

problem = GymNE(
    env="fiction_env/QCAEnv-v8",  # Name of the environment
    network=CustomPolicy,  # Linear policy that we defined earlier
    num_actors=0,  # Use 4 available CPUs. Note that you can modify this value, or use 'max' to exploit all available CPUs
    observation_normalization=True,  # Observation normalization was not used in Lunar Lander experiments
)

radius_init = 4.5  # (approximate) radius of initial hypersphere that we will sample from
max_speed = radius_init / 15.  # Rule-of-thumb from the paper
center_learning_rate = max_speed / 2.

searcher = PGPE(
    problem,
    popsize=2000,  # For now we use a static population size
    radius_init=radius_init,  # The searcher can be initialised directely with an initial radius, rather than stdev
    center_learning_rate=center_learning_rate,
    stdev_learning_rate=0.1,  # stdev learning rate of 0.1 was used across all experiments
    optimizer="clipup",  # Using the ClipUp optimiser
    optimizer_config={
        'max_speed': max_speed,  # with the defined max speed
        'momentum': 0.9,  # and momentum fixed to 0.9
    }
)
searcher = SNES(problem, stdev_init=5)

if __name__ == "__main__":
    StdOutLogger(searcher)
    searcher.run(5000)
    center_solution = searcher.status["center"]  # Get mu
    policy_net = problem.to_policy(center_solution)  # Instantiate a policy from mu
