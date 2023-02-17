import numpy as np
import sys
import matplotlib.pyplot as plt


sys.path.append("C:\\Users\\bergd\\Desktop\\github\\PHSX815_Week5") #For windows
sys.path.append('/mnt/c/Users/bergd/Desktop/github/PHSX815_Week5') #For windows
from python.Random import Random 

#global variables
bin_width = 0.
Xmin = 0.
Xmax = 1.
random = Random()

# Inverted quadratic function (-3x^2 + 3). Inverted to improve the efficiency of the function near zero
# Note: multiply by bin width to match histogram
def Quad(x): 
	return (-3 * x*x) + 3 # (1./np.sqrt(2.*np.arccos(-1)))*np.exp(-x*x/2.) #Definitionn normal func

def PlotQuad(x, bin_width):
	return bin_width*Quad(x)/2 # The width of our bin * value of Quad at x

# Uniform (flat) distribution scaled to Quadratic max
# Note: multiply by bin width to match histogram
def Flat(x):
	return 3 #Returns the maximum value of the Quad function in the range of observation
	
def PlotFlat(x, bin_width):
	return bin_width*Flat(x)/2 # The width of our bin * value the flat func

# Get a random X value according to a flat distribution
def SampleFlat():
	return Xmin + (Xmax-Xmin)*random.rand() #gives some value in the domain [xmin, xmax] 


#main function
if __name__ == "__main__":


	# default number of samples
	Nsample = 100

	doLog = False
	doExpo = False

	# read the user-provided seed from the command line (if there)
	#figure out if you have to have -- flags before - flags or not
	if '-Nsample' in sys.argv:
		p = sys.argv.index('-Nsample')
		Nsample = int(sys.argv[p + 1])
	if '-range' in sys.argv:
		p = sys.argv.index('-range')
		Xmax = float(sys.argv[p + 1])
		Xmin = -float(sys.argv[p + 1])
#	if '--log' in sys.argv:
#		p = sys.argv.index('--log')
#		doLog = bool(sys.argv[p])
#	if '--expo' in sys.argv:
#		p = sys.argv.index('--expo')
#		doExpo = bool(sys.argv[p])
	if '-h' in sys.argv or '--help' in sys.argv:
		print ("Usage: %s [-Nsample] number [-range] Xmax [--log] [--expo] " % sys.argv[0])
		print
		sys.exit(1)  


	data = []
	Ntrial = 0.
	i = 0.

	while i < Nsample:
		Ntrial += 1

		# Get our random sample
		X = SampleFlat()
		R = Quad(X) / Flat(X)

		# Get our random number
		rand = random.rand()

		#See if need to accept or reject sample point
		if(rand > R): #Reject: random number is larger than function
			continue
		else:
			data.append(X)
			i += 1
		#Repeat whole process until we have the number of samples we said

	#Getting the efficiency of the process	
	if Ntrial > 0:
			print("Efficiency was", float(Nsample) / float(Ntrial))

#normalize data for probability distribution
	weights = np.ones_like(data) / len(data)
	#print(weights)

	n = plt.hist(data, alpha = 0.3, label = "samples from f(x)", bins = 100, weights = weights)
	plt.ylabel("Probability / bin")
	plt.xlabel("x")
	
	bin_width = n[1][1] - n[1][0]
	hist_max = max(n[0])
	print(n)
	#plt.show()

	if not doLog:
		plt.ylim(min(bin_width * Quad(Xmax), 1. / float(Nsample + 1)),
		1.5 * max(hist_max,bin_width*Quad(0)))
	else:
		plt.ylim(min(bin_width * Quad(Xmax), 1. / float(Nsample + 1)),
		80 * max(hist_max,bin_width * Quad(0)))
		plt.yscale("log")


	# Plotting the solid line plots
	x = np.arange(Xmin, Xmax, 0.001)
	y_norm = list(map(PlotQuad, x, np.ones_like(x) * bin_width))
	plt.plot(x, y_norm, color = 'green', label = 'target f(x)')

	y_flat = list(map(PlotFlat, x, np.ones_like(x) * bin_width))
	plt.plot(x, y_flat, color = 'red', label = 'proposal g(x)')
	plt.title("Density estimation with Monte Carlo")

	plt.legend()
	plt.show()
	plt.savefig("RandomQuad.pdf")