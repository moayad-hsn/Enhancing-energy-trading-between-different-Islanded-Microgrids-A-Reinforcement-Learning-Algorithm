some pseudo code for the trading process


note that for an MG the:

MG.best_buy_price = unit_price

MG.best_sell_price = return_price

MG.lowest_sell_price = unit_price

MG.highest_buy_price = network_price


given:
MG1, MG2, Amount

offered = MG2.exess_energy
needed = Amount


if needed > offered:
	price = price_bargain(offered, MG1, MG2)
	MG2.exess_energy = 0
	rem_needed = needed-offered
	reward -= rem_needed/needed
	needed = rem_needed
	reward += price/MG1.optimal_price
	
else 
	price = price_bargain(needed, MG1, MG2)
	offered = offered - needed
	MG2.exess_energy = offered
	rem_needed = 0
	reward-= rem_needed / needed
	needed = 0
	reward+= price/MG1.optimal_price


def price_bargain(offered, Seller, Buyer):
	buyer_offer = Buyer.unit_price? what price is better to do it 
	buyer_limit = Buyer.network_price
	seller_offer = Seller.network_price
	seller_limit = Seller.unit_price
	time_step = 0

do i need all this shit?
no
lets look at this other way of doing it

offer = mg2.cal_offer()
if price >= mg2.gen_price:
	if mg2.offer >= amount:
		mg2.battery.discharge(amount)
		mg1.battery.charge(amount)
		rem_amount = 0
		reward -= rem_needed / amount
		reward += (price - mg1.gen_price)/mg1.gen_price
	else:
		mg2.battery.discharge(offer)
		mg1.battery.charge(offer)
		rem_amount = amount - offer
		reward -= rem_amount / amount
		reward += (price - mg1.gen_price)/mg1.gen_price
 





 