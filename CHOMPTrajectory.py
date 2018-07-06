import math as math
import numpy as np
from CHOMPworld import *

# Defines a point on the trajectory by the uav's location & orientation
# theta = 0 along +y, theta in radians [-pi, pi] +pi/2 is -x, -pi/2 is +x
class CHOMPWaypoint:
    def __init__ (self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
        
    def vector(self):
        vector = np.array([[self.x, self.y, self.theta]])
        return vector.T

# Represents a trajectory of a uav through a world with obstacles
# numWaypoints: the number of waypoints between start and end
# start: CHOMPWaypoint where the uav starts
# end: CHOMPWaypoint where the uav ends
# speed: how fast the robot should move
# uav: UAV2D model of the uav
# world: World2D which provides obstacle information
class CHOMPTrajectory:

    def __init__ (self, numWaypoints, start, end, speed, uav, world):
        self.start = start
        self.end = end
        # start with straight line guess
        delx = (end.x - start.x)/(numWaypoints + 1.)
        dely = (end.y - start.y)/(numWaypoints + 1.)
        deltheta = (end.theta - start.theta)/(numWaypoints + 1.)
        self.waypoints = []
        vector = start.vector()
        for i in range(1, numWaypoints + 1):
            wp = CHOMPWaypoint(start.x + i*delx,
                                        start.y + i*dely,
                                        start.theta + i*deltheta)
            self.waypoints.append(wp)
            vector = np.vstack((vector, wp.vector()))
            
        vector = np.vstack((vector, end.vector()))
        self.numWaypoints = numWaypoints
        #approx total time est
        time = math.sqrt((start.x - end.x)**2 + (start.y - end.y)**2)/speed
        self.delt = time/(numWaypoints + 1.)
        self.uav = uav
        self.world = world
        self.waypointsvector = vector[3:-3]
        
        K = np.zeros(((self.numWaypoints+1)*3,self.numWaypoints*3))
        for i in range(self.numWaypoints*3):
            K[i][i] = 1
            K[i+3][i] = -1
        A = np.dot(K.T,K)

        b = np.zeros(((self.numWaypoints)*3,1))
        b[0] = -self.start.x
        b[1] = -self.start.y
        b[2] = -self.start.theta
        b[self.numWaypoints*3-1] = -self.end.theta
        b[self.numWaypoints*3-2] = -self.end.y
        b[self.numWaypoints*3-3] = -self.end.x

        self.b = b
        self.A = A
    
    def vector(self):
        vec = self.start.vector()
        vec = np.vstack(( vec, self.waypointsvector ))
        return np.vstack((vec, self.end.vector()))

    # The partial objective cost based on smoothness of the trajectory
    # math followed from CHOMP paper
    def calcSmoothness(self):
        Fsmooth = 0

        if self.numWaypoints == 0:
            return .5*((self.start.x - self.end.x)**2 +
                          (self.start.y - self.end.y)**2 +
                          (self.start.theta - self.end.theta)**2)/self.delt**2

        wone = self.waypoints[0]
        Fsmooth = ((self.start.x - wone.x)**2 +
                   (self.start.y - wone.y)**2 +
                   (self.start.theta - wone.theta)**2)/self.delt**2
        
        for i in range(1, self.numWaypoints):
            wi = self.waypoints[i]
            wiprev = self.waypoints[i-1]
            Fsmooth += ((wi.x - wiprev.x)**2 +
                        (wi.y - wiprev.y)**2 +
                        (wi.theta - wiprev.theta)**2)/self.delt**2

        wlast = self.waypoints[-1]
        Fsmooth += ((self.end.x - wlast.x)**2 +
                    (self.end.y - wlast.y)**2 +
                    (self.end.theta - wlast.theta)**2)/self.delt**2

        return Fsmooth*.5/(self.numWaypoints + 1)

    # The partial objective cost gradient based on smoothness
    # math followed from CHOMP paper
    def calcSmoothnessGrad(self):
        
        Atraj2 = np.dot(self.A,self.waypointsvector)

        return Atraj2 + self.b

    # helper function to calculate piecewise cost of obstacle collisions
    # math followed from CHOMP paper
    # wstart: CHOMPWaypoint at start of trajectory piece
    # wend: CHOMPWaypoint at end of trajectory piece
    def calcObstacleAtWaypoints(self, wstart, wend):
        bodyStart = cornersUAVinWorld(self.uav,
                                     wstart.x,
                                     wstart.y,
                                     wstart.theta)
        bodyEnd = cornersUAVinWorld(self.uav,
                                   wend.x,
                                   wend.y,
                                   wend.theta)
        obscost = 0
        
        for i in range(len(bodyStart)):
            cavg = .5*(self.world.distanceCostToClosestObstacle(bodyStart[i]) +
                       self.world.distanceCostToClosestObstacle(bodyEnd[i]))
            obscost += cavg*math.sqrt((bodyEnd[i].x - bodyStart[i].x)**2 +
                                   (bodyEnd[i].y - bodyStart[i].y)**2)

        return obscost

    # The partial object cost based on obstacle collisions in the trajectory
    # math followed from CHOMP paper
    def calcObstacle(self):
        Fobs = 0

        if self.numWaypoints == 0:
            return self.calcObstacleAtWaypoints(self.start, self.end)
 
        Fobs = self.calcObstacleAtWaypoints(self.start, self.waypoints[0])

        for i in range(1, self.numWaypoints):
            Fobs += self.calcObstacleAtWaypoints(self.waypoints[i-1], self.waypoints[i])

        Fobs += self.calcObstacleAtWaypoints(self.waypoints[-1], self.end)

        return Fobs
    
    # Helper function to calculate the partial Obstacle gradient
    # takes an index i of a waypoint to calculate the obstacle gradient for
    def calcObsGradAtWaypoint(self, index):
        wp = self.waypoints[index]
        wprev = wp
        wnext = wp
        if index == 0:
            wprev = self.start
        else:
            wprev = self.waypoints[index - 1]
        
        if index == self.numWaypoints -1:
            wnext = self.end
        else:
            wnext = self.waypoints[index + 1]
        
        bodyPrev = cornersUAVinWorld(self.uav,
                                     wprev.x,
                                     wprev.y,
                                     wprev.theta)
        body = cornersUAVinWorld(self.uav,
                                 wp.x,
                                 wp.y,
                                 wp.theta)
        bodyNext = cornersUAVinWorld(self.uav,
                                     wnext.x,
                                     wnext.y,
                                     wnext.theta)
        
        # compute x', x'' and the normal of x' for each body point at this waypoint 
        xvelmag = []
        xvelnorm = []
        xaccel = []
        for i in range(len(bodyPrev)):
            xv = np.array([[(body[i].x - bodyPrev[i].x)/self.delt,
                                   (body[i].y - bodyPrev[i].y)/self.delt]]).T

            xvn = np.array([[(bodyNext[i].x - body[i].x)/self.delt,
                                   (bodyNext[i].y - body[i].y)/self.delt]]).T
            xvm = math.sqrt(xv[0]**2 + xv[1]**2)
            xvnorm = np.array([xv[0]/xvm, xv[1]/xvm])
            xacc = np.array([(xvn[0] - xv[0])/self.delt,
                              (xvn[1] - xv[1])/self.delt])
            xvelmag.append(xvm)
            xvelnorm.append(xvnorm)
            xaccel.append(xacc)
    
        # compute the jacobian describing the transform of the robot state to world coordinates
        J = []
        corners = cornersUAVinWorld(self.uav, 0.0, 0.0, 0.0)
        for i in range(len(body)):
            j = np.zeros((2,3))
            j[0][0] = 1
            j[1][1] = 1
            j[0][2] = -1*corners[i].x*math.sin(wp.theta) - corners[i].y*math.cos(wp.theta)
            j[1][2] = corners[i].x*math.cos(wp.theta) - corners[i].y*math.sin(wp.theta)
            J.append(j)
        
        # compute the gradient around the waypoint of the distance cost using finite differencing
        gradC = []
        C = []
        for i in range(len(body)):
            bx = Point2D(body[i].x + 0.1, body[i].y)
            by = Point2D(body[i].x, body[i].y + 0.1)
            c = self.world.distanceCostToClosestObstacle(body[i])
            
            cgradx = (self.world.distanceCostToClosestObstacle(bx) - c )/0.1
            cgrady = (self.world.distanceCostToClosestObstacle(by) - c )/0.1

            gc = np.array([[cgradx, cgrady]]).T
            gradC.append(gc)
            C.append(c)
        
        I = np.eye(2)
        
        # Now that all partial terms of the gradient are ready, compute the gradient. 
        # see CHOMP paper for the math being followed here
        gradobsw = np.zeros((3,1))
        for i in range(len(body)):
            xxt = np.dot(xvelnorm[i],xvelnorm[i].T)
            ixxt = I - xxt
            k = np.dot(ixxt, xaccel[i])/xvelmag[i]**2
            ck = C[i] * k
            ixxtgradc = np.dot(ixxt,gradC[i])
            xixxtgradc = xvelmag[i]*ixxtgradc
            xixxtgradcsubck = xixxtgradc - ck
            gradobsi = np.dot(J[i].T,xixxtgradcsubck)
            gradobsw += gradobsi
        return gradobsw
    
    # Calculate the gradient of the cost based on the obstacle objective
    def calcObsGrad(self):
        obsGrad = self.calcObsGradAtWaypoint(0)
        
        for i in range(1, self.numWaypoints):
            gradi = self.calcObsGradAtWaypoint(i)
            obsGrad = np.vstack((obsGrad, gradi))
        
        return obsGrad
    
    # run gradient descent iteratively on this CHOMPTrajectory
    # eta: representing the measure of step size. (1/eta) 
    # iterations: number of iterations to run. 
    #             For visualizations, 1 will give a new frame per iteration
    def runCHOMP(self, eta, iterations):
        for i in range(iterations):
            og = self.calcObsGrad()
            sg = self.calcSmoothnessGrad()
            grad = self.calcObsGrad() + .01*self.calcSmoothnessGrad()
            ainv = np.linalg.inv(self.A)
            gradU = np.dot(ainv, grad)
            trajnext = self.waypointsvector - gradU/eta
            self.waypointsvector = trajnext
            for j in range(self.numWaypoints):
                self.waypoints[j].x = trajnext[3*j][0]
                self.waypoints[j].y = trajnext[3*j + 1][0]
                self.waypoints[j].theta = trajnext[3*j + 2][0]
            
        return self.vector() 
                
        
        
        
        
        

        
        
        

