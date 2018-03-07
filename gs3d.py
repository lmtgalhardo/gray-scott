import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as anim
from numpy import zeros
from random import random
from random import randint
from math import floor,ceil
import matplotlib
from matplotlib.pyplot import *
from matplotlib.animation import FuncAnimation
from numpy import array
from matplotlib import colors
import sys
from pylab import *

class Diffuser(object):

	def __init__(self,
	           nX = 25,
	           nY = 25,
	           nZ = 25,
	           diffusion = 1,
	           f = .05,
	           k = .01,
	           da = .1,
	           db = .1,
	           ):
		self.nX = nX
		self.nY = nY
		self.nZ = nZ
		self.blockA = np.ones((nX, nY, nZ)) 
		self.blockB = zeros((nX, nY, nZ)) 
		self.diffusion = diffusion
		self.f = f
		self.k = k
		self.da = da
		self.db = db

	#sets a block to a value 
	def setBlock(self,block,val):
		x, y, z = block[0], block[1], block[2] 
		self.blockB[x,y,z] = val

	#returns particle B's grid of concentrations
	def getGrid(self):
		return self.blockB

	def simulateTimeStep(self,nIters=1):
		for step in xrange(nIters):
			self.blockB = self.update()

	def update(self):
		raise NotImplementedError("Only instantiate Stochastic or Laplacian Diffusers")

	# Returns the correct x index after moving step blocks
	# to the right (to the left for negative step)
	def nextX(self,x,step):
		return (x+step)%self.nX
	# Returns the correct y index after moving step blocks
	# forward (back for negative step)
	def nextY(self,y,step):
		return (y+step)%self.nY

	# Returns the correct z index after moving step blocks
	# to up (down for negative step)
	def nextZ(self,z,step):
		return (z+step)%self.nZ

# convolves 3X3 cells where center cell is passed in block 
# with the Laplacian kernel and returns value
def calcLaplacian(self, block, coords):
	a = 0.1/8 #corner
	b = 0.7/6 #mid
	c = 0.2/12 #cross

	x = coords[0]
	y = coords[1]
	z = coords[2]

	xf = self.nextX(x, 1)
	xb = self.nextX(x,-1)
	yf = self.nextY(y, 1)
	yb = self.nextY(y,-1)
	zf = self.nextZ(z, 1)
	zb = self.nextZ(z,-1)

	#center
	L = -1.0*block[x,y,z]
	#corners
	L += a*(block[xf,yf,zf] + 
		    block[xf,yb,zf] + 
		    block[xb,yf,zf] +
		    block[xb,yb,zf] +
		    block[xf,yf,zb] + 
		    block[xf,yb,zb] +
		    block[xb,yf,zb] +
		    block[xb,yb,zb]) 

	#mid
	L += b*(block[x, y, zf] + 
		    block[x, y, zb] + 
		    block[x, yf, z] +
		    block[x, yb, z] +
		    block[xf, y, z] + 
		    block[xb, y, z]) 

	#cross
	L += c*(block[xf, yf, z] + 
		    block[xf, yb, z] + 
		    block[xb, yf, z] +
		    block[xb, yb, z] +
		    block[xf, y, zf] + 
		    block[xf, y, zb] +
		    block[xb, y, zf] +
		    block[xb, y, zb] +
		    block[x, yf, zf] + 
		    block[x, yf, zb] +
		    block[x, yb, zf] +
		    block[x, yb, zb])

	return L

class GrayScottDiffuser(Diffuser):

	def update(self):
		newBlockA = zeros((self.nX, self.nY, self.nZ)) 
		newBlockB = zeros((self.nX, self.nY, self.nZ))

	    # Next we will iterate through each element of self.blockA and 
	    # self.blockB to compute the new concentrations of A and B in each cell
		for i in range(0, self.nX):
			for j in range(0, self.nY):
				for h in range(0, self.nZ):
					#for (i,j,h), val in enumerate(self.blockA): 

					#calculate Laplacian:
					La = calcLaplacian(self, self.blockA, (i,j,h))
					Lb = calcLaplacian(self, self.blockB, (i,j,h))

					A = self.blockA[i,j,h]
					B = self.blockB[i,j,h]
					f = self.f
					k = self.k
					da = self.da
					db = self.db

					#calculate new concnetrations of particles A and B
					newBlockA[i,j,h] = A + (da*La - A*B**2 + f*(1-A))*0.5 
					newBlockB[i,j,h] = B + (db*Lb + A*B**2 - (k+f)*B)*0.5

		#update
		self.blockA = newBlockA
		self.blockB = newBlockB

		return self.blockB

# returns color associated with passed in density value
def getColor(val):
	cmap = cm.get_cmap('cool', 1000) #define colormap 
	i = int(val * 1000) #index within colormap

	t = "09"
	if (i > 500): t = "3B" #transparency value

	rgb = cmap(i)[:3]
	rgba  = str(matplotlib.colors.rgb2hex(rgb)) + t
	return rgba

#inserts space between voxels so no voxels touch
def explode(data):
	size = np.array(data.shape)*2
	data_e = np.zeros(size - 1, dtype=data.dtype)
	data_e[::2, ::2, ::2] = data
	return data_e


def frame(f,c, fig, length):
	x, y, z = np.indices((length, length, length)) # prepare some coordinates
	voxels = (x == -1) & (y == -1)  & (z == -1) #initialize voxels
	colors = np.empty(voxels.shape, dtype=object) #intialize colors

	grid = c.getGrid()

	#iterate through grid to determine if a cell is filled and with what color
	#from colormap
	for i in range(length):
		for j in range(length):
			for h in range(length):
				if(grid[i,j,h] != 0):
					dot  = (x == i) & (y == j)  & (z == h) 
					if (int(grid[i,j,h] * 1000) > 0):
						voxels = voxels|dot
						colors[dot] = getColor(grid[i,j,h])
							# else: print grid[(i,j,h)], maxval
					#if grid[i,j,h] > 0:
					#	voxels = voxels|dot
					#	colors[dot] = '#11FFFF2D'


	#explode
	voxels_2 = explode(voxels)
	fcolors_2 = explode(colors)

	# Shrink the gaps between voxels
	x, y, z = np.indices(np.array(voxels_2.shape) + 1).astype(float) // 2
	x[0::2, :, :] += 0.05
	y[:, 0::2, :] += 0.05
	z[:, :, 0::2] += 0.05
	x[1::2, :, :] += 0.95
	y[:, 1::2, :] += 0.95
	z[:, :, 1::2] += 0.95

	# and plot everything
	plt.clf()
	ax = fig.gca(projection='3d')
	ax.voxels(x,y,z, voxels_2, facecolors=fcolors_2, edgecolor='#00000000')
	ax.set_xlabel("Iteration: " + str(f))

	c.simulateTimeStep(25)


def main():
	f = .039 # the feed rate of A 
	k = .058 # removal/kill rate of B
	dA = 1.0 # the diffusion rates of A
	dB = 0.5 # the diffusion rates of B
	length = 50

	c = GrayScottDiffuser(nX = length, 
						  nY = length, 
						  nZ = length, 
						  diffusion = 1, 
						  f = f, 
						  k = k, 
						  da = dA, 
						  db = dB,
						  )

	x = length/2
	y = length/2
	z = length/2

	#seed the grid 
	c.setBlock((x, y, z), 1)
	c.setBlock((x+1, y, z), 1)
	c.setBlock((x, y+1, z), 1)
	c.setBlock((x, y, z+1), 1)
	c.setBlock((x-1, y, z), 1)
	c.setBlock((x, y-1, z), 1)
	c.setBlock((x, y, z-1), 1)

	fig = plt.figure()
	a = anim.FuncAnimation(fig, frame, fargs = (c,fig,length), frames=1000, repeat=False)
	plt.show()

# Boiler plate invokes the main function
if __name__ == "__main__":
  main()
