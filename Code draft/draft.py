#lets get this party started

#Imports

#constants
SCHOOL_MAX_LOAD = 0
HOUSE_MAX_LOAD =  0
MOSQUE_MAX_LOAD = 0
HEALTH_CENTER_MAX_LOAD = 0 
WATER_PUMP_MAX_LOAD = 0

KABKABEA_LOAD_PARAMETERS = 
KABKABYA_BATTERY_PARAMETERS = 
KABKABYA_DISTANCE_TO_CENTER = 
ALFASHIR_LOAD_PARAMETERS = 
ALFASHIR_BATTERY_PARAMETERS = 
ALFASHIR_DISTANCE_TO_CENTER = 
NYALA_LOAD_PARAMETERS = 
NYALA_BATTERY_PARAMETERS = 
NYALA_DISTANCE_TO_CENTER = 

distances = {"Alfashir_Nyala": 10, "Alfashir_Kabkabya": 50, "Nyala_Kabkabya": 30, "Nyala_Alfashir": 10, "Kabkabya_Alfashir": 50, "Kabkabya_Nyala": 30} 


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
		idx = self.usage_trend_df[self.usage_trend_df["Time"] == time].index.values
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
			self.remaining_capacity+= self.charge_rate*amount
			leftover = self.remaining_capacity - self.max_capacity
			self.remaining_capacity = min(self.max_capacity, self.remaining_capacity)
			
			return max(leftover,0) 

#takes energy from the battery providing the needed amount and returns amount provided form the battery

	def supply(self, amount):
		remaining = self.remaining_capacity
		self.remaining_capacity -= amount*self.discharge_cofficient
		self.remaining_capacity = max(self.remaining_capacity,0)
		
		return min(amount, remaining)
	
#dissipate the battery with the factor, might not need it probably

	def dissipate(self):
		self.remaining_capacity = self.remaining_capacity * math.exp(- self.dissipation)
	
	@property
	def SOC(self):
		self._SOC = self.remaining_capacity/self.max_capacity
		
		return self._SOC




class Generation:
	def __init__(self, name, maxCapacity = None):
		self.solar_df = pd.read_csv("data/" + name + "_sloar_generation.csv")
		self.wind_df = pd.read_csv("data/" + name + "_wind_generation.csv")
		self.sloar_generation = np.array(self.solar_df["value"])
		self.wind_generation = np.array(self.wind_df["value"])
		self.generation = self.sloar_generation + self.wind_generation
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
	def __init__(self, name, load_parameters, battery_parameters, distance_to_center):
		self.name = name #name of the microgrid used for data loading
		self.load_parameters = load_parameters #np array of the parameters to create the load of the microgrid that is the number of schools, houses, mosques, health centers and water pumps
		self.battery_parameters = battery_parameters #np array of the parameters to create the battery of the microgrid
		self.distance_to_center = distance_to_center # distance to a central point between the microgrids, used for loss calculations
		self.battery = self._create_battery(battery_parameters)
		self.houses, self.schools, self.mosques, self.health_centers, self.water_pumps= self._create_loads(load_parameters)
		self.generation = Generation(name)
		self.unit_price = 10


#creats a battery given its battery parameters ie max cap, dis coeff, initial rem cap and it's charge rate
	def _create_battery(self, battery_parameters):
		max_capacity, discharge_cofficient, remaining_capacity, charge_rate = battery_parameters
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

#current status of the MG containing it's battery's remaining capacity, it's current poewr generation and its current total load
	def state (self, time):
		total_generation = self.generation.current_generation(time)
		total_load = self.total_load(time)
		battery_status = self.battery.remaining_capacity

		return total_load, total_generation, battery_status

	def to_trade(self, time):
		load, generation, battery = self.state(time)

		return abs(load - (generation+battery))




class MicrogridEnv (gym.Env):
	def __init__(self):
		self.main_mG = Microgrid("kabkabea", KABKABEA_LOAD_PARAMETERS, KABKABYA_BATTERY_PARAMETERS, KABKABYA_DISTANCE_TO_CENTER)
		self.first_mg = Microgrid("Alfashir", ALFASHIR_LOAD_PARAMETERS, ALFASHIR_BATTERY_PARAMETERS, ALFASHIR_DISTANCE_TO_CENTER)
		self.second_mg= Microgrid("Neyala", NYALA_LOAD_PARAMETERS, NYALA_BATTERY_PARAMETERS, NYALA_DISTANCE_TO_CENTER)

		self.time_step = 0
		self.dates = np.array(pd.read_csv("data/" + main_mG.name + "_sloar_generation.csv")["Time"], dtype = np.float32)
		self.start_date = dates[self.time_step]
		self.current_price = NETWORK_PRICE


		self.action_space = spaces.Box(low=np.array([0,0,0,self.main_mG.unit_price]), high=np.array([3, 2, self.main_mG.battery.max_battery_capacity, NETWORK_PRICE]), dtype = np.float32, shape = (1,4))
		self.observation_space =  spaces.Box(low = 0, high = self.max_power_generation, dtype = np.float32, shape = (1,5))
		


	def _status(self):
		self.current_date = self.dates[self.time_step]
		current_load, current_generation, remaining_capacity = self.main_mG.state(current_date)
		time_s = self.time_step
		previous_price = self.current_price
		state = np.array([remaining_capacity, current_load, current_generation, previous_price, time_s])
		return state

	def reset(self):
		self.start_date = self.dates[random.randint(0,len(self.dates))]
		self.main_mG.battery = self.main_mg._create_battery(KABKABYA_BATTERY_PARAMETERS)
		self.current_price = NETWORK_PRICE
		self.energy_bought = []
		self.energy_sold = []
		self.prices = []

		state = self._status()
		return state

	def _travel_loss(self, target_mg, amount):
		'''
		src_name = self.main_mG.name
		dist_name = target_mg.name
		final_name = src_name + dist_name
		distance = distances[final_name]
		base_res = 1.1 #25mm aluminium
		voltage = 66000#use sub_transmission?
		loss = ((amount**2) * (base_res*distance))/(voltage **2)
		'''
		return loss



	def step(slelf, action):
		action_tyep = action[0]
		target_mg_idx = action[1]
		amount = action[2]
		price = action[3]
		reward = 0
		main_mg = self.main_mG


		if targer_mg_index < 1:
			target_mg = self.first_mg
		else:
			target_mg = self.second_mg
			

		amount += self._travel_loss(target_mg, amount)
		offer = target_mg.to_trade(self.current_date)
		
		if action_type <1:#buy from target MG
			if price >= target_mg.unit_price:
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
					reward += (price - main_mg.unit_priceni)/main_mg.unit_price
					self.energy_bought.append(amount - rem_amount)
			else:
				reward -= 1
			self.prices.append(price)


		elif action_type < 2:
			if price >= main_mg.unit_price and price <= NETWORK_PRICE:
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
					reward += (price - main_mg.unit_price)/main_mgi.unit_price
					self.energy_sold.append(amount - rem_amount)
			else:
				reward -= 1
			self.prices.append(price)
			


		else:
		self.time_step +=1
		state = self._status()
		if self.time_step >= MAX_STEPS:
			is_done = True
		else:
			is_done = False
		return state, reward, is_done, _


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
	