from Code import enviroment
from Code.enviroment import MicrogridEnv
import collections
import ptan
import pandas as pd

from typing import List

from tensorboardX import SummaryWriter

MAX_STEPS = 40

SCHOOL_MAX_LOAD = 6_012.0
HOUSE_MAX_LOAD =  5_678.0
MOSQUE_MAX_LOAD = 4_324.0
HEALTH_CENTER_MAX_LOAD = 5_800.0 
WATER_PUMP_MAX_LOAD = 770.0

UM_BADER_LOAD_PARAMETERS = [70, 1, 2, 1, 2]
HAMZA_ELSHEIKH_LOAD_PARAMETERS = [50, 1, 1, 0, 1]
TANNAH_LOAD_PARAMETERS = [45, 0, 1, 0, 1]


if __name__ == "__main__":



	usage_trends_df = pd.read_csv("data/usage_trends.csv")
	max_H = max(usage_trends_df["House"]) * HOUSE_MAX_LOAD * TANNAH_LOAD_PARAMETERS[0]
	max_S = max(usage_trends_df["School"]) * SCHOOL_MAX_LOAD  * TANNAH_LOAD_PARAMETERS[1]
	max_M = max(usage_trends_df["Mosque"]) * MOSQUE_MAX_LOAD  * TANNAH_LOAD_PARAMETERS[2]
	max_HS = max(usage_trends_df["Health_center"]) * HEALTH_CENTER_MAX_LOAD * TANNAH_LOAD_PARAMETERS[3]
	max_W = max(usage_trends_df["Water_pump"]) * WATER_PUMP_MAX_LOAD * TANNAH_LOAD_PARAMETERS[4]

	maxall = max_W + max_HS + max_S + max_M + max_H
	print(maxall)







	env = MicrogridEnv()

	state = env.reset()

	rewards = []
	steps = 0
	while True:
		action = env.action_space.sample()
		#print(action)
		state, reward, terminal, _ = env.step(action)
		print(reward)
		rewards.append(reward)
		steps +=1
		print (steps)
		if terminal:
			break

		if steps > MAX_STEPS:
			break
	print("Total rewards:", sum(rewards))





#	agent = DullAgent(agent = test_env.action_space.sample())


#	exp_source = ptan.experience.ExperienceSource(env = test_env, agent = agent, steps_count = 2)

#	for idx, exp in enumerate(exp_source):
#		if idx > 2:
#			break
#		print(exp)


#	agent = Agent(test_env)

'''
	writer = SummaryWriter(comment = "-Q-iterations")
	iter_no = 0
	best_reward = 0.0
	while True:
		iter_no += 1
		agent.play_n_random_steps(100)
		agent.value_iteration()
		reward = 0.0

		for _ in (TEST_EPISODES):
			reward += agent.play_episode(test_env)
		reward /= TEST_EPISODES
		writer.add_scalar("reward", reward, iter_no)

		if reward > best_reward:
			print("Best reward updated %.3f , %.3f" %(best_reward, reward))
			best_reward = reward

			if reward > 1000:
				print("solved in %d ters" %iter_no)
				break
	writer.close()
'''


























	#print(test_env.observation_space.shape, test_env.observation_space.shape)





	
'''parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    

'''
