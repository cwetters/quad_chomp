import math as math

# A rectangle approximation
class UAV2D:

    def __init__(self, width, length):
        self.w = width
        self.l = length

# A circle centered at x,y with radius r        
class Obstacle:
    
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r

class Point2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
# xmax, xmin, ymax, ymin for use in visualization- the world assumes no boundaries
# obstacles: a list of Obstacles 
class World2D:
    
    def __init__(self, xmax, xmin, ymax, ymin, obstacles):
        self.xmax = xmax
        self.xmin = xmin
        self.ymax = ymax
        self.ymin = ymin
        self.obstacles = obstacles
        self.epsilon = abs(xmax - xmin)*0.1 #some reasonable "far enough away"

    # Calculates the distance Cost (as described in the CHOMP paper) for a point
    # in the world.
    # point: Point2D in world
    def distanceCostToClosestObstacle(self, point):
        if len(self.obstacles) == 0:
            return 0
        cost = 0
        for obstacle in self.obstacles:
            c = math.sqrt((point.x-obstacle.x)**2 + (point.y-obstacle.y)**2) - obstacle.r
            if c < 0:
                c = -c + .5*self.epsilon
            elif c < self.epsilon:
                c = .5/self.epsilon*(c - self.epsilon)**2
            else:
                c = 0
            if c > cost:
                cost = c
        return cost

# For a given uav orientation, calculates the world coordinates of the
# exterior corners of the uav
# theta = 0 along +y, theta in radians [-pi, pi] +pi/2 is -x, -pi/2 is +x
def cornersUAVinWorld(uav, x, y, theta):
    corners = []
    corners.append(Point2D(-uav.w/2., uav.l/2.))
    corners.append(Point2D(uav.w/2., uav.l/2.))
    corners.append(Point2D(-uav.w/2., -uav.l/2.))
    corners.append(Point2D(uav.w/2., -uav.l/2.))
    rotated = []
    
    while theta < 0:
        theta += 2*math.pi

    for corner in corners:
        rotated.append(Point2D(x + corner.x*math.cos(theta) - corner.y*math.sin(theta),
                            y + corner.x*math.sin(theta) + corner.y*math.cos(theta)))

    return rotated              


