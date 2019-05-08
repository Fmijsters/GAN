
class Testnet():
	"""
	A three hidden-layer generative neural network
	"""
	def __init__(self):
		X = 1

		# Layer 1 Variables
		D_W1 = 2
		D_B1 = 3

		# Layer 2 Variables
		D_W2 = 4
		D_B2 = 5

		# Layer 3 Variables
		D_W3 = 6
		D_B3 = 7

		# Out Layer Variables
		D_W4 = 8
		D_B4 = 9
		# Store Variables in list
		self.var_list = [D_W1, D_B1, D_W2, D_B2, D_W3, D_B3, D_W4, D_B4]


var = Testnet()

print(var)
print(var.var_list)
