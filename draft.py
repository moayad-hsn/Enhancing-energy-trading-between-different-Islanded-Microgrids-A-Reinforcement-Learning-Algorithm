#lets get this party started

#Imports
import pandas as pd
import numpy as np
from matplotlib import pyplot
import gym
from gym import spaces
import random


#constants
SCHOOL_MAX_LOAD = 6_012.0
HOUSE_MAX_LOAD =  5_678.0
MOSQUE_MAX_LOAD = 4_324.0
HEALTH_CENTER_MAX_LOAD = 5_800.0 
WATER_PUMP_MAX_LOAD = 770.0

UM_BADER_LOAD_PARAMETERS = [70, 1, 2, 1, 2]
UM_BADER_MAX_LOAD = 500_000.0
UM_BADER_BATTERY_PARAMETERS = [500_000.0, 0.02, 300_000.0, 0.3]
HAMZA_ELSHEIKH_LOAD_PARAMETERS = [50, 1, 1, 0, 1]
HAMZA_ELSHEIKH_MAX_LOAD =350_000.0 
HAMZA_ELSHEIKH_BATTERY_PARAMETERS = [350_000.0, 0.02, 200_000.0, 0.3]
TANNAH_LOAD_PARAMETERS = [45, 0, 1, 0, 1]
TANNAH_MAX_LOAD = 300_000.0
TANNAH_BATTERY_PARAMETERS = [300_000.0, 0.02, 150_000.0, 0.3]

distances = {"Um_Bader_Tannah": 10, "Um_Bader_Hamza_Elsheikh": 50, "Tannah_Hamza_Elsheikh": 30, "Tannah_Um_Bader": 10, "Hamza_Elsheikh_Um_Bader": 50, "Hamza_Elsheikh_Tannah": 30} 


NETWORK_PRICE = 19 #In cents

MAX_STEPS = 10_000


#Helper Classes

#defining a load, ie. schools, houses, health centers and water pumps
class Load:
	def __init__(self, name, max_load, num_of_units):
		self.name = name #name of the load item to get the usage trend
		self.max_load = max_load #maximum_load_needed_by_load_category
		self.usage_trends_df = pd.read_csv("data/usage_trends.csv")
		self.usage_trends_values = np.array(self.usage_trends_df[name]) #trend_of_percentage_of_usage_during_a_day
		self.num_of_units = num_of_units #number_of_units_of_load_available_in_area

	def _current_single_Load(self, time):
		idx = self.usage_trends_df[self.usage_trends_df["Time"] == time].index.values
		current_load = self.max_load * self.usage_trends_values[idx]
		
		return current_load

	def current_total_load(self, time):
		single_load = self._current_single_Load(time)
		
		return single_load * self.num_of_units


#battery Class, to define the storage of the system
class Battery:
	def __init__(self, max_capacity, discharge_cofficient, remaining_capacity, charge_rate):
		self.max_capacity = max_capacity #full_charge_capacity
		#self.dissipation = dissipation #Dissipation_cofficient
		self.discharge_cofficient = discharge_cofficient #Discharge_coefficient
		self.remaining_capacity = remaining_capacity #Remaining_Capacity
		self.charge_rate = charge_rate #percentage_charged_from_inputed_amount
		#self.charge_loss = charge_loss #Charging_Loss
		#self.max_charge_time = max_charge_time #Maximum_Charging_Time
		#self.max_power_output = max_power_output #Max_power_battery_can_output_per_timestep
		#self.charge_step = charge_step #Charging_Step

#charges the battery with given amount and returns leftover amount to be returned if amount + currnet capacity > max capacity
	def charge(self, amount):
		empty = self.max_capacity - self.remaining_capacity
		if empty <=0:
			return amount
		else:
			self.remaining_capacity+= amount
			leftover = self.remaining_capacity - self.max_capacity
			self.remaining_capacity = min(self.max_capacity, self.remaining_capacity)
			
			return max(leftover,0) 

#takes energy from the battery providing the needed amount and returns amount provided form the battery

	def supply(self, amount):
		remaining = self.remaining_capacity
		#print(amount)
		self.remaining_capacity -= amount
		self.remaining_capacity = max(self.remaining_capacity,0)
		
		return min(amount, remaining)
	
#dissipate the battery with the factor, might not need it probably
'''
	def dissipate(self):
		self.remaining_capacity = self.remaining_capacity * math.exp(- self.dissipation)

	def SOC(self):
		self._SOC = self.remaining_capacity/self.max_capacity
		return self._SOC

'''


class Generation:
	def __init__(self, name, maxCapacity = None):
		self.solar_df = pd.read_csv("data/Solar/" + name + "_solar_generation.csv")
		self.wind_df = pd.read_csv("data/wind/" + name + "_wind_generation.csv")
		self.solar_generation = np.array(self.solar_df["value"], dtype = np.float32)
		self.wind_generation = np.array(self.wind_df["value"], dtype = np.float32)
		for i in range(len(self.wind_generation)):
			if self.wind_generation[i] <0:
				self.wind_generation[i] = 0
		self.generation = self.solar_generation + self.wind_generation
		self.max_generation = max(self.generation)

#given current time, give the total generation of the solar and wind units		
	def current_generation(self, time):
		idx = self.solar_df[self.solar_df["Time"] == time].index.values

		return self.generation[idx]

#do this if were doing and expectation of wind/solar power generation, Am I doing it? well lets see
#	def expected_generation(self, time):
#		pass


#	def fewHourExpected(self, time, timeStep1, timeStep2, tume ):
#		return self.power[]




class Microgrid:
	def __init__(self, name, load_parameters, battery_parameters):
		self.name = name #name of the microgrid used for data loading
		self.load_parameters = load_parameters #np array of the parameters to create the load of the microgrid that is the number of schools, houses, mosques, health centers and water pumps
		self.battery_parameters = battery_parameters #np array of the parameters to create the battery of the microgrid
		#self.distance_to_center = distance_to_center # distance to a central point between the microgrids, used for loss calculations
		self.battery = self._create_battery(battery_parameters)
		self.houses, self.schools, self.mosques, self.health_centers, self.water_pumps= self._create_loads(load_parameters)
		self.generation = Generation(name)
		self.unit_price = 10


#creats a battery given its battery parameters ie max cap, dis coeff, initial rem cap and it's charge rate
	def _create_battery(self, battery_parameters):
		max_capacity = battery_parameters[0]
		discharge_cofficient  = battery_parameters[1]
		remaining_capacity = battery_parameters[2]
		charge_rate  = battery_parameters[3]
		battery = Battery(max_capacity, discharge_cofficient, remaining_capacity, charge_rate)
		
		return battery

#creates the loads of the MG, using the number of units for each load and its name for data reasons
	def _create_loads(self, load_parameters):
		num_houses, num_schools, num_mosques, num_health_centers, num_water_pumps = load_parameters
		houses_load = Load("House", HOUSE_MAX_LOAD, num_houses)
		schools_load = Load("School", SCHOOL_MAX_LOAD, num_schools)
		mosques_load = Load("Mosque", MOSQUE_MAX_LOAD, num_mosques)
		health_centers_load = Load("Health_center", HEALTH_CENTER_MAX_LOAD, num_health_centers)
		water_pumps_load = Load("Water_pump", WATER_PUMP_MAX_LOAD, num_water_pumps)

		return houses_load, schools_load, mosques_load, health_centers_load, water_pumps_load

#returns the total load by all MG load units
	def total_load(self, time):
		houses_load = self.houses.current_total_load(time)
		schools_load = self.schools.current_total_load(time)
		mosques_load = self.mosques.current_total_load(time)
		health_centers_load = self.health_centers.current_total_load(time)
		water_pumps_load = self.water_pumps.current_total_load(time)
		total_load = houses_load + schools_load + mosques_load + health_centers_load + water_pumps_load
		
		return total_load

#current status of the MG containing it's battery's remaining capacity, it's current power generation and its current total load
	def state (self, time):
		total_generation = self.generation.current_generation(time)
		total_load = self.total_load(time)
		battery_status = self.battery.remaining_capacity

		return total_load, total_generation, battery_status

	def to_trade(self, time):
		load, generation, battery = self.state(time)

		return abs(load - (generation+battery))

	def supply(self, load, time):
		if load >= self.generation.current_generation(time):
			load -= self.generation.current_generation(time)
			if load <= self.battery.remaining_capacity:
				self.battery.remaining_capacity -= load
				load = 0
			else:
				load -= self.battery.remaining_capacity
				self.battery.remaining_capacity = 0
		else:
			load = 0

		return load




class MicrogridEnv (gym.Env):
	def __init__(self):
		self.main_mG = Microgrid("Hamza_Elsheikh", HAMZA_ELSHEIKH_LOAD_PARAMETERS, HAMZA_ELSHEIKH_BATTERY_PARAMETERS)
		self.first_mg = Microgrid("Um_Bader", UM_BADER_LOAD_PARAMETERS, UM_BADER_BATTERY_PARAMETERS)
		self.second_mg= Microgrid("Tannah", TANNAH_LOAD_PARAMETERS, TANNAH_BATTERY_PARAMETERS)

		self.time_step = 0
		self.dates = np.array(pd.read_csv("data/Solar/" + self.main_mG.name + "_solar_generation.csv")["Time"])
		self.start_date = self.dates[self.time_step]
		self.current_price = NETWORK_PRICE



		#print(self.main_mG.unit_price, self.main_mG.battery.max_capacity, NETWORK_PRICE)
		self.action_space = spaces.Box(low=np.array([0,0,0,self.main_mG.unit_price]), high=np.array([3, 2, self.main_mG.battery.max_capacity, NETWORK_PRICE]), dtype = np.float32)
		self.observation_space =  spaces.Box(low =np.array([0.0, 0.0, 0.0, 0.0, 0.0]), high =np.array([self.main_mG.battery.max_capacity, HAMZA_ELSHEIKH_MAX_LOAD, self.main_mG.generation.max_generation, NETWORK_PRICE, MAX_STEPS]), dtype = np.float32)
		


	def _status(self):
		if self.time_step >= len(self.dates):
			self.time_step = 0
		self.current_date = self.dates[self.time_step]
		current_load, current_generation, remaining_capacity = self.main_mG.state(self.current_date)
		time_s = self.time_step
		previous_price = self.current_price
		state = np.array([remaining_capacity, current_load, current_generation, previous_price, time_s])
		return state

	def reset(self):
		self.start_date = self.dates[random.randint(0,len(self.dates))]
		self.main_mG.battery = self.main_mG._create_battery(HAMZA_ELSHEIKH_BATTERY_PARAMETERS)
		self.current_price = NETWORK_PRICE
		self.energy_bought = []
		self.energy_sold = []
		self.prices = []

		state = self._status()
		return state

	def _travel_loss(self, target_mg, amount):
		
		src_name = self.main_mG.name
		dist_name = target_mg.name
		final_name = src_name +"_"+ dist_name
		distance = distances[final_name]
		base_res = 1.1 #25mm aluminium
		voltage = 33000#use sub_transmission?
		loss = ((amount**2) * (base_res*distance))/(voltage **2)
		
		return loss



	def step(self, action):
		action_type = action[0]
		target_mg_idx = action[1]
		amount = action[2]
		price = action[3]
		reward = 0
		main_mg = self.main_mG


		if target_mg_idx < 1:
			target_mg = self.first_mg
		else:
			target_mg = self.second_mg
			

		amount += self._travel_loss(target_mg, amount)
		offer = target_mg.to_trade(self.current_date)
		
		if action_type <1:#buy from target MG
			if price >= target_mg.unit_price:
				if offer != 0:
					if offer >= amount:
						target_mg.battery.supply(amount)
						main_mg.battery.charge(amount)
						rem_amount = 0
						reward -= rem_amount / amount
						reward += (price - main_mg.unit_price)/main_mg.unit_price
						self.energy_bought.append(amount - rem_amount)
					else:
						target_mg.battery.supply(offer)
						main_mg.battery.charge(offer)
						rem_amount = amount - offer
						reward -= rem_amount / amount
						reward += (price - main_mg.unit_price)/main_mg.unit_price
					self.energy_bought.append(amount - rem_amount)
			else:
				reward -= 1
			self.prices.append(price)
			main_mg.supply(main_mg.total_load(self.current_date), self.current_date)


		elif action_type < 2:
			if price >= main_mg.unit_price and price <= NETWORK_PRICE:
				if offer != 0:

					if offer >= amount:
						main_mg.battery.supply(amount)
						target_mg.battery.charge(amount)
						rem_amount = 0
						reward -= rem_amount / amount
						reward += (price - main_mg.unit_price)/main_mg.unit_price
						self.energy_sold.append(amount - rem_amount)
					else:
						main_mg.battery.supply(offer)
						target_mg.battery.charge(offer)
						rem_amount = amount - offer
						reward -= rem_amount / amount
						reward += (price - main_mg.unit_price)/main_mg.unit_price
						self.energy_sold.append(amount - rem_amount)
			else:
				reward -= 1
			self.prices.append(price)
			main_mg.supply(main_mg.total_load(self.current_date), self.current_date)
			


		else:
			main_mg.supply(main_mg.total_load(self.current_date), self.current_date)
		self.time_step +=1
		state = self._status()
		if self.time_step >= MAX_STEPS:
			is_done = True
		else:
			is_done = False
		return state, reward, is_done, {}


	def render(self):
		pass





'''
class Grid:
#this is for a single MG, we here set the extra produciton or the needed power to acheive stability to the entire system.
	def __init__():
		self.distanceFromMG = distanceFromMG
		self.transmisionLoss = transmisionLoss
		self.additionalProducion = additionalProducion
'''


if __name__ == "__main__":
	env = MicrogridEnv()
	env.seed(1)
	rewards =[]
	state=env.reset()
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
	''
	'''parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    

    # Testing the environment
    # Initialize the environment
    #env = MicrogridEnv()
    #env.seed(1)
    # Save the rewards in a list
    #rewards = []
    # reset the environment to the initial state
    #state = env.reset()
    # Call render to prepare the visualization
    #env.render()
    # Interact with the environment (here we choose random actions) until the terminal state is reached
    while True:
        # Pick an action from the action space (here we pick an index between 0 and 80)
        action = env.action_space.sample()
        # Using the index we get the actual action that we will send to the environment
        print(ACTIONS[action])
        # Perform a step in the environment given the chosen action
        state, reward, terminal, _ = env.step(action)
        env.render()
        print(reward)
        rewards.append(reward)
        if terminal:
            break
    print("Total Reward:",sum(rewards))

    # Plot the TCL SoCs 
    states = np.array(rewards)
    pyplot.plot(rewards)
    pyplot.title("rewards")
    pyplot.xlabel("Time")
    pyplot.ylabel("rewards")
    pyplot.show()
'''
