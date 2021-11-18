# Licensed under ISC
from itertools import product

import numpy as np

from scipy.stats import wrapcauchy
from tools import *

def lerp(a, b, x):
    return (a + x * (b - a))

def smootheststep(t):
    """Smooth curve with a zero derivative at 0 and 1, making it useful for
    interpolating.
    """
    return t**7*(1716 -9009*t + 20020*t**2 - 24024 * t**3 + 16380*t**4 - 6006 *t**5 + 924*t**6)


class PerlinNoiseFactory(object):
    """Callable that produces Perlin noise for an arbitrary point in an
    arbitrary number of dimensions.  The underlying grid is aligned with the
    integers.

    There is no limit to the coordinates used; new gradients are generated on
    the fly as necessary.
    """

    def __init__(self, dimension, octaves=1, tile=(), unbias=True,gradients=None):
        """Create a new Perlin noise factory in the given number of dimensions,
        which should be an integer and at least 1.

        More octaves create a foggier and more-detailed noise pattern.  More
        than 4 octaves is rather excessive.

        ``tile`` can be used to make a seamlessly tiling pattern.  For example:

            pnf = PerlinNoiseFactory(2, tile=(0, 3))

        This will produce noise that tiles every 3 units vertically, but never
        tiles horizontally.

        If ``unbias`` is true, the smoothstep function will be applied to the
        output before returning it, to counteract some of Perlin noise's
        significant bias towards the center of its output range.
        """
        self.dimension = dimension
        self.octaves = octaves
        self.tile = tile + (0,) * dimension
        self.unbias = unbias

        # For n dimensions, the range of Perlin noise is ±sqrt(n)/2; multiply
        # by this to scale to ±1
        self.scale_factor = 2 * dimension ** -0.5
        
        self.gradient = gradients or {}

    def _generate_gradient(self):
        # Generate a random unit vector at each grid point -- this is the
        # "gradient" vector, in that the grid tile slopes towards it

        # 1 dimension is special, since the only unit vector is trivial;
        # instead, use a slope between -1 and 1
        if self.dimension == 1:
            return (np.random.uniform(-1, 1),)

        # Generate a random point on the surface of the unit n-hypersphere;
        # this is the same as a random unit vector in n dimensions.  Thanks
        # to: http://mathworld.wolfram.com/SpherePointPicking.html
        # Pick n normal random variables with stddev 1

        random_point = np.random.normal(0, 1,size=self.dimension) 
        # Then scale the result to a unit vector
        scale = np.sum(random_point**2)**-0.5# 
        return random_point*scale# tuple(coord * scale for coord in random_point)
    # @profile
    def get_or_gen(self,point):
        g=self.gradient.get(point,None)
        if g is None:
            g=self._generate_gradient()
            self.gradient[point] = g

        return g
    # @profile
    def get_plain_noise(self, point):
        """Get plain noise for a single point, without taking into account
        either octaves or tiling.
        """
        
        point=np.array(point)
        if len(np.shape(point))==1:
            point=point.reshape((1,-1))
        assert np.shape(point)[-1]==self.dimension
    
        
        # Build a list of the (min, max) bounds in each dimension
        
        # point=np.array(point)
        grid_coords =np.zeros((len(point),self.dimension,2))
        grid_coords[:,:,:] = np.floor(point)[:,:,np.newaxis]
        grid_coords[:,:,1] += 1


        # Compute the dot product of each gradient vector and the point's
        # distance from the corresponding grid point.  This gives you each
        # gradient's "influence" on the chosen point.

        grid_points = np.zeros((len(point),2**self.dimension,self.dimension))

        
        gradients=np.zeros((len(point),2**self.dimension,self.dimension))
        for p, coords in enumerate(grid_coords):
            
            prod_coords = list(product(*coords))
            grid_points[p]=prod_coords
            for i,grid_point in enumerate(prod_coords):
                gradients[p][i] = self.get_or_gen(grid_point)
                
        dots = np.sum(gradients*(point[:,np.newaxis,:]-grid_points),axis=2)
        assert len(dots)==len(grid_points)
    

        # Interpolate all those dot products together.  The interpolation is
        # done with smoothstep to smooth out the slope as you pass from one
        # grid cell into the next.
        # Due to the way product() works, dot products are ordered such that
        # the last dimension alternates: (..., min), (..., max), etc.  So we
        # can interpolate adjacent pairs to "collapse" that last dimension.  Then
        # the results will alternate in their second-to-last dimension, and so
        # forth, until we only have a single value left.

        dim = self.dimension
      
        smoothsteps = smootheststep(point-grid_coords[:,:,0])
    
        while (np.shape(dots)[1]>1):
            dim -= 1
            # print(dim)
            s = smoothsteps[:,dim]
            

            dots = lerp(dots[:,0::2],dots[:,1::2],s[:,np.newaxis])
            
        assert np.shape(dots)[1]==1
    
        
        return dots[:,0] * self.scale_factor



    def parallel_get(self,points):

        
        if len(np.array(points).shape)>2:
            assert len(points)==1
            points=points[0]
        
        ret = np.zeros(len(points))
        for o in range(self.octaves):
            o2 = 1 << o


            new_points = np.array(points) * o2
            tile = np.array(self.tile)
            new_points[:,tile!=0] %= tile[tile!=0]*o2
            

            ret += self.get_plain_noise(new_points)/o2

        # Need to scale n back down since adding all those extra octaves has
        # probably expanded it beyond ±1
        # 1 octave: ±1
        # 2 octaves: ±1½
        # 3 octaves: ±1¾
        ret /= 2 - 2 ** (1 - self.octaves)
        
        if self.unbias:
            # The output of the plain Perlin noise algorithm has a fairly
            # strong bias towards the center due to the central limit theorem
            # -- in fact the top and bottom 1/8 virtually never happen.  That's
            # a quarter of our entire output range!  If only we had a function
            # in [0..1] that could introduce a bias towards the endpoints...
            r = (ret + 1) / 2
            # Doing it this many times is a completely made-up heuristic.
            for _ in range(int(self.octaves / 2 + 0.5)):
                r = smootheststep(r)
            ret = r * 2 - 1
        
        return ret
       

    def __call__(self, *point):
        """Get the value of this Perlin noise function at the given point.  The
        number of values given should match the number of dimensions.
        """
        if np.shape(point) != (self.dimension,):
            return self.parallel_get(point)
            

        ret = 0
        for o in range(self.octaves):
            o2 = 1 << o
            new_point = []
            for i, coord in enumerate(point):
                coord *= o2
                if self.tile[i]:
                    coord %= self.tile[i] * o2
                new_point.append(coord)

            nn = self.get_plain_noise(new_point)
            ret += nn / o2

        # Need to scale n back down since adding all those extra octaves has
        # probably expanded it beyond ±1
        # 1 octave: ±1
        # 2 octaves: ±1½
        # 3 octaves: ±1¾
        ret /= 2 - 2 ** (1 - self.octaves)

        if self.unbias:
            # The output of the plain Perlin noise algorithm has a fairly
            # strong bias towards the center due to the central limit theorem
            # -- in fact the top and bottom 1/8 virtually never happen.  That's
            # a quarter of our entire output range!  If only we had a function
            # in [0..1] that could introduce a bias towards the endpoints...
            r = (ret + 1) / 2
            # Doing it this many times is a completely made-up heuristic.
            for _ in range(int(self.octaves / 2 + 0.5)):
                r = smootheststep(r)
            ret = r * 2 - 1

        return ret
