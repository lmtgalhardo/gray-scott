from numpy import zeros
from random import random
from random import randint
from math import floor,ceil
import numpy as np
from matplotlib.pyplot import *
from matplotlib.animation import FuncAnimation
from numpy import array
from matplotlib import colors
import sys


class Diffuser(object):

  def __init__(self,
               nX = 25,
               nY = 25,
               diffusion = 1,
               f = .05,
               k = .01,
               da = .1,
               db = .1,
               ):
    self.nX = nX
    self.nY = nY
    self.blockA = np.ones((nX, nY)) 
    self.blockB = zeros((nX, nY))
    self.diffusion = diffusion
    self.f = f
    self.k = k
    self.da = da
    self.db = db

  #sets the value of a block
  def setBlock(self,block,val):
    x, y = block[0], block[1]
    self.blockB[x,y] = val

  #returns the data set of particle B
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
  # up (down for negative step)
  def nextY(self,y,step):
    return (y+step)%self.nY

def calcLaplacian(self, D, block, coord):
  x, y = coord[0], coord[1]
  L = 0
  for i in range(-1,2):
    for j in range(-1,2):
      L += D[i+1][j+1]*block[self.nextX(x,j),self.nextY(y,-i)] 
  return L


class GrayScottDiffuser(Diffuser):

  def update(self):
    D = [[.05, .2, .05], [.2, -1, .2], [.05, .2, .05]]
    newBlockA = zeros((self.nX, self.nY))
    newBlockB = zeros((self.nX, self.nY))

    # Next we will iterate through each element of self.blocks 
    # to compute the new concentrations of A and B in each cell
    for i in range(self.nX):
      for j in range(self.nY): 
      

        #calculate Laplacian:
        La = calcLaplacian(self, D, self.blockA, (i,j))
        Lb = calcLaplacian(self, D, self.blockB, (i,j))


        A = self.blockA[i,j]
        B = self.blockB[i,j]
        f = self.f
        k = self.k
        da = self.da
        db = self.db

        #calculate new A and B concentrations
        newBlockA[i,j] = A + (da*La - A*B**2 + f*(1-A))*0.5 
        newBlockB[i,j] = B + (db*Lb + A*B**2 - (k+f)*B)*0.5
   
    self.blockA = newBlockA
    self.blockB = newBlockB
    return self.blockB


def main():

  iters = 1000
  cm = "cool"
  length = 100

  f = .037 # the feed rate of A 
  k = .06 # removal/kill rate of B
  dA = 1.0 # the diffusion rates of A
  dB = 0.5 # the diffusion rates of B
  c = GrayScottDiffuser(nX = length,
                        nY = length,
                        diffusion = 1,
                        f = f,
                        k = k,
                        da = dA,
                        db = dB,
                        )

  #seed the grid
  c.setBlock((length/2,length/2),1)

  for l in range(iters):
    clf()

    grid = c.getGrid()
    pcolormesh(array([i for i in xrange(length+1)]),array([j for j in xrange(length+1)]),grid,cmap = cm, shading = 'flat',)
    colorbar()
    clim(0,1) 
    xlabel("Iteration %d"%l)
    pause(0.1)

    c.simulateTimeStep(25)


# Boiler plate invokes the main function
if __name__ == "__main__":
  main()
