from Code import enviroment
from Code.enviroment import MicrogridEnv

import os
import time
import math
import ptan
import gym
import argparse
from tensorboardX import SummaryWriter

import numpy as np



if __name__ == "__main__":
	env = MicrogridEnv()

	s = env.reset()
	print(env.action_space.sample())
