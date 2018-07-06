import os
import imp
import sys
import timeout_decorator
import unittest
import math
import numpy as np
from gradescope_utils.autograder_utils.decorators import weight
from gradescope_utils.autograder_utils.json_test_runner import JSONTestRunner

from CHOMPworld import *
from CHOMPTrajectory import *



class TestSet_CHOMPWorld(unittest.TestCase):
    def setUp(self):
        pass

    @weight(1)
    @timeout_decorator.timeout(1.0)
    def test_point2d(self):
        """Test Point2D"""

        point = Point2D(2.0,1.0)
        self.assertTrue(point.x == 2.0, "# point x")
        self.assertTrue(point.y == 1.0, "# point y")
    
    @weight(1)
    @timeout_decorator.timeout(1.0)    
    def test_obstacle(self):
        """Test Obstacle"""

        obs = Obstacle(2.0,1.0,0.5)
        self.assertTrue(obs.x == 2.0, "# obstacle x")
        self.assertTrue(obs.y == 1.0, "# obstacle y")
        self.assertTrue(obs.r == 0.5, "# obstacle r")

    @weight(1)
    @timeout_decorator.timeout(1.0)    
    def test_world2d(self):
        """Test World2D"""

        obs = Obstacle(2.0,1.0,0.5)
        world = World2D(10.0, -10.0, 10.0, -10.0, [obs])
        self.assertTrue(world.xmax == 10.0, "# world xmax")
        self.assertTrue(world.xmin == -10.0, "# world xmin")
        self.assertTrue(world.ymax == 10.0, "# world ymax")
        self.assertTrue(world.ymin == -10.0, "# world ymin")
        self.assertTrue(world.epsilon == 2.0, "# world epsilon")
        self.assertTrue(len(world.obstacles) == 1, "# world obstacles")
        self.assertTrue(world.obstacles[0] == obs, "# world obstacle")

    @weight(1)
    @timeout_decorator.timeout(1.0)
    def test_world2d_discost(self):
        """Test world2d distance cost"""

        point = Point2D(2.0,1.0)
        near = Point2D(1.1,1.1)
        far = Point2D(-10.0, -10.0)
        
        obs = Obstacle(2.0,1.0,0.5)
        world = World2D(10.0, -10.0, 10.0, -10.0, [obs])
        
        self.assertTrue(world.distanceCostToClosestObstacle(point) == 1.5, 
                        "# distance cost for middistance point")
        self.assertTrue(round(world.distanceCostToClosestObstacle(near) - 0.635576857733, 10) == 0, 
                        "# distance cost for near point")
        self.assertTrue(world.distanceCostToClosestObstacle(far) == 0, 
                        "# distance cost for far point")
        
    @weight(1)
    @timeout_decorator.timeout(1.0)
    def test_uav2d(self):
        """Test UAV2D"""

        uav = UAV2D(0.5,0.3)
        self.assertTrue(uav.w == 0.5, "# uav w")
        self.assertTrue(uav.l == 0.3, "# uav l")
   
    @weight(1)
    @timeout_decorator.timeout(1.0)
    def test_uav2d_corners(self):
        """Test corner transforms"""
        
        uav = UAV2D(0.5,0.3)
        corners =  cornersUAVinWorld(uav, 0.0, 0.0, math.pi)
        self.assertTrue(len(corners) == 4, "# number of corners")

        self.assertTrue(round(corners[3].x - -0.25, 10) == 0, "# uav corner no rotation 3x")
        self.assertTrue(round(corners[3].y - 0.15, 10) == 0, "# uav corner no rotation 3y") 
        self.assertTrue(round(corners[2].x - 0.25, 10) == 0, "# uav corner no rotation 2x")
        self.assertTrue(round(corners[2].y - 0.15, 10) == 0, "# uav corner no rotation 2y")
        self.assertTrue(round(corners[1].x - -0.25, 10) == 0, "# uav corner no rotation 1x")
        self.assertTrue(round(corners[1].y - -0.15, 10) == 0, "# uav corner no rotation 1y") 
        self.assertTrue(round(corners[0].x - 0.25, 10) == 0, "# uav corner no rotation 0x")
        self.assertTrue(round(corners[0].y - -0.15, 10) == 0, "# uav corner no rotation 0y")
        
        corners =  cornersUAVinWorld(uav, 1.0, 1.0, -math.pi/2)
        
        self.assertTrue(round(corners[0].x - 1.15, 10) == 0, "# uav corner rotation 3x")
        self.assertTrue(round(corners[0].y - 1.25, 10) == 0, "# uav corner rotation 3y") 
        self.assertTrue(round(corners[1].x - 1.15, 10) == 0, "# uav corner rotation 2x")
        self.assertTrue(round(corners[1].y - 0.75, 10) == 0, "# uav corner rotation 2y")
        self.assertTrue(round(corners[2].x - 0.85, 10) == 0, "# uav corner rotation 1x")
        self.assertTrue(round(corners[2].y - 1.25, 10) == 0, "# uav corner rotation 1y") 
        self.assertTrue(round(corners[3].x - 0.85, 10) == 0, "# uav corner rotation 0x")
        self.assertTrue(round(corners[3].y - 0.75, 10) == 0, "# uav corner rotation 0y")
        
class TestSet_CHOMPTrajectory(unittest.TestCase):
    def setUp(self):
        pass

    @weight(1)
    @timeout_decorator.timeout(1.0)    
    def test_waypoint(self):
        """Test CHOMPWaypoint"""
        from CHOMPTrajectory import CHOMPWaypoint

        wp = CHOMPWaypoint(2.0,1.0,0.0)
        self.assertTrue(wp.x == 2.0, "# waypoint x")
        self.assertTrue(wp.y == 1.0, "# waypoint y")
        self.assertTrue(wp.theta == 0.0, "# waypoint theta")
        self.assertTrue((wp.vector() == np.array([[2.0, 1.0, 0.0]]).T).all, "# waypoint vector")
    
    @weight(1)
    @timeout_decorator.timeout(1.0)    
    def test_trajectory_init(self):
        """Test CHOMPTrajectory"""

        obs1 = Obstacle(6.0, 5.0, 1.0)
        obs2 = Obstacle (3.0, 7.0, 1.0)
        obstacles = [obs1, obs2]
        uav = UAV2D(0.5, 0.5)
        world = World2D(10.0, -10.0, 10.0, -10.0, obstacles)
        start = CHOMPWaypoint(1.0,1.0,0.0)
        end = CHOMPWaypoint(10.0, 10.0, 0.0)

        traj = CHOMPTrajectory(8, start, end, 1.0, uav, world)
        
        self.assertTrue(traj.numWaypoints == 8, "# trajectory number of waypoints")
        self.assertTrue((traj.start.vector() == start.vector()).all, "# trajectory start")
        self.assertTrue((traj.end.vector() == end.vector()).all, "# trajectory end")
        wpvec = np.array([[ 2.,  2.,  0.,  3.,  3.,  0.,  4.,  4.,  0.,  5.,  5.,  0.,  6.,  6.,  0.,  7.,  7.,  0., 8.,  8.,  0.,  9.,  9.,  0.]]).T
        self.assertTrue((traj.waypointsvector == wpvec).all, "# waypoint vector of linear first guess")   

    @weight(1)
    @timeout_decorator.timeout(1.0)    
    def test_trajectory_smooth_cost(self):
        """Test Smoothness cost on trajectory"""

        obs1 = Obstacle(6.0, 5.0, 1.0)
        obs2 = Obstacle (3.0, 7.0, 1.0)
        obstacles = [obs1, obs2]
        uav = UAV2D(0.5, 0.5)
        world = World2D(10.0, -10.0, 10.0, -10.0, obstacles)
        wp = CHOMPWaypoint(1.0,1.0,0.0)
        end = CHOMPWaypoint(10.0, 10.0, 0.0)

        traj = CHOMPTrajectory(8, wp, end, 1.0, uav, world)
        
        self.assertTrue(round(traj.calcSmoothness() - 0.5,10) == 0, "# smoothness cost of straight trajectory")
        smoothGrad = np.zeros((24,1))
        self.assertTrue((traj.calcSmoothnessGrad() == smoothGrad).all, "# smoothness cost grad of straight trajectory")
        
    @weight(1)
    @timeout_decorator.timeout(1.0)    
    def test_trajectory_no_obs_cost(self):
        """Test Obstacles cost on trajectory with no obstacles"""

        uav = UAV2D(0.5, 0.5)
        world = World2D(10.0, -10.0, 10.0, -10.0, [])
        start = CHOMPWaypoint(1.0,1.0,0.0)
        end = CHOMPWaypoint(10.0, 10.0, 0.0)

        traj = CHOMPTrajectory(8, start, end, 1.0, uav, world)
        
        self.assertTrue(traj.calcObstacle() == 0.0, "# obstacle cost of no obs trajectory")
        obsGrad = np.zeros((24,1))
        self.assertTrue((traj.calcObsGrad() == obsGrad).all, 
                        "# obstacle cost grad of no obs trajectory")         
    
    @weight(1)
    @timeout_decorator.timeout(1.0)    
    def test_trajectory_obs_cost_(self):
        """Test Obstacles cost on trajectory with obstacles"""

        obs1 = Obstacle(6.0, 5.0, 1.0)
        obs2 = Obstacle (3.0, 7.0, 1.0)
        obstacles = [obs1, obs2]
        uav = UAV2D(0.5, 0.5)
        world = World2D(10.0, -10.0, 10.0, -10.0, obstacles)
        start = CHOMPWaypoint(1.0,1.0,0.0)
        end = CHOMPWaypoint(10.0, 10.0, 0.0)

        traj = CHOMPTrajectory(8, start, end, 1.0, uav, world)
        
        self.assertTrue(round(traj.calcObstacle()- 12.818109216, 10) == 0, "# obstacle cost of obs trajectory")

    @weight(1)
    @timeout_decorator.timeout(1.0)    
    def test_trajectory_run_chomp_(self):
        """Test Obstacles cost on trajectory with obstacles"""

        obs1 = Obstacle(6.0, 5.0, 1.0)
        obs2 = Obstacle (3.0, 7.0, 1.0)
        obstacles = [obs1, obs2]
        uav = UAV2D(0.5, 0.5)
        world = World2D(10.0, -10.0, 10.0, -10.0, obstacles)
        start = CHOMPWaypoint(1.0,1.0,0.0)
        end = CHOMPWaypoint(10.0, 10.0, 0.0)

        traj = CHOMPTrajectory(8, start, end, 1.0, uav, world)
        chompresult = traj.runCHOMP(10,100)
        self.assertTrue(round(traj.calcSmoothness() - 0.914200301709, 10) == 0, "# smoothness cost after CHOMP")
        self.assertTrue(round(traj.calcObstacle() - 3.06512905863, 10) == 0, "# obstacle cost after CHOMP")
        
def pretty_format_json_results(test_output_file):
    import json
    import textwrap

    output_str = ""

    try:
        with open(test_output_file, "r") as f:
            results = json.loads(f.read())
 
        total_score_possible = 0.0

        if "tests" in results.keys():
            for test in results["tests"]:
                output_str += "Test %s: " % (test["name"])
                output_str += "%2.2f/%2.2f.\n" % (test["score"], test["max_score"])
                total_score_possible += test["max_score"]
                if "output" in test.keys():
                    output_str += "  * %s\n" % (
                        textwrap.fill(test["output"], 70,
                                      subsequent_indent = "  * "))
                output_str += "\n"

            output_str += "TOTAL TESTS PASSED: %2.2f/%2.2f\n" % (results["score"], total_score_possible)

        else:
            output_str += "TOTAL TESTS PASSED: %2.2f\n" % (results["score"])
            if "output" in results.keys():
                output_str += "  * %s\n" % (
                        textwrap.fill(results["output"], 70,
                                      subsequent_indent = "  * "))
                output_str += "\n"
    
    except IOError:
        output_str += "No such file %s" % test_output_file
    except Exception as e:
        output_str += "Other exception while printing results file: ", e

    return output_str

def global_fail_with_error_message(msg, test_output_file):
    import json

    results = {"score": 0.0,
               "output": msg}

    with open(test_output_file, 'w') as f:
        f.write(json.dumps(results,
                           indent=4,
                           sort_keys=True,
                           separators=(',', ': '),
                           ensure_ascii=True))

def run_tests(test_output_file = "test_results.json"):
    try:
        # Check for existence of the expected files
        expected_files = [
            "CHOMPworld.py",
            "CHOMPTrajectory.py"
        ]
        for file in expected_files:
            if not os.path.isfile(file):
                raise ValueError("Couldn't find an expected file: %s" % file)

        do_testing = True

    except Exception as e:
        import traceback
        global_fail_with_error_message("Somehow failed trying to import the files needed for testing " + traceback.format_exc(1), test_output_file)
        do_testing = False

    if do_testing:
        test_cases = [TestSet_CHOMPWorld,
                      TestSet_CHOMPTrajectory]

        suite = unittest.TestSuite()
        for test_class in test_cases:
            tests = unittest.defaultTestLoader.loadTestsFromTestCase(test_class)
            suite.addTests(tests)

        with open(test_output_file, "w") as f:
            JSONTestRunner(stream=f).run(suite)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Please invoke with one argument: the result json to write."
        print "(This test file assumes it's in the same directory as the code to be tested."
        exit(1)

    run_tests(test_output_file=sys.argv[1])
