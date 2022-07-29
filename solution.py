# Importing Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2.aruco as aruco
import requests


video = cv2.VideoCapture(1)  # Capturing the arena from the overhead camera


n_detection      =    True
crop_1_bool      =    False
crop_2_bool      =    False
round_1_bool     =    False
round_2_bool     =    False
round_3_bool     =    False
match            =    False
adj              =    False
path_1_bool      =    False
path_2_bool      =    False
path_3_bool      =    False
death_count      =    0
crop_bool_count  =    0
next_start       =    44
node_count       =    0
weapon_count     =    0
weapon           =    'nothing'
b_cells_c        =    [ (315, 315), (245, 595), (315, 595), (385, 595) ] # arena
# b_cells_c      =    [ (540, 180), (540, 300), (540, 420) ]     # r_1 arena
b_cells          =    [ 40, 35, 44, 53 ]


def shape_detection(c):
    """ Returns the shape of the arena grid.  """
    
    rect = cv2.minAreaRect(c)
    w = rect[1][0]
    h = rect[1][1]
    area = cv2.contourArea(c)
    peri = 2 * (w + h)
    ratio = peri ** 2 / area
    if ratio > 18:
        return 'circle'
    else:
        return 'square'


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """

    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

............>>> angle_between((1, 0, 0), (0, 1, 0))
............1.5707963267948966
............>>> angle_between((1, 0, 0), (1, 0, 0))
............0.0
............>>> angle_between((1, 0, 0), (-1, 0, 0))
............3.141592653589793
...."""

    v1_unit_vector = unit_vector(v1)
    v2_unit_vector = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_unit_vector, v2_unit_vector),
                     -1.0, 1.0))


def distance(a, b):
    """ Returns the distance between two coordinates  """

    dist = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
    return dist


def reverse(my_list):
    """ Reverses a list of elements  """

    return [ele for ele in reversed(my_list)]


import sys


class Graph:

    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)] for row in
                      range(vertices)]

    # Debug
    def print_solution(self, dist):
        print 'Vertex \tDistance from Source'
        for node in range(self.V):
            print (node, '\t', dist[node])

    # Determines the vertex with minimum distance from the set of unvisited vertices
    def min_distance(self, dist, shortest_path_set):

        min_dist = sys.maxsize

        for v in range(self.V):

            if dist[v] < min_dist and shortest_path_set[v] == False:
                min_dist = dist[v]
                global min_index
                min_index = v

        return min_index

    # Dijkstra's single source shortest path algorithm
    def find_parent(self, a):
        for i in parent:
            if i[1] == a:
                return i[0]

    def dijkstra(self, src, dest):

        global dist
        dist = [sys.maxsize] * self.V
        dist[src] = 0
        shortest_path_set = [False] * self.V
        global parent
        parent = []

        for cout in range(self.V):

            # Minimum distance vertex from the set of unvisited vertices
            # u is always equal to src in the first iteration

            u = self.min_distance(dist, shortest_path_set)

            # Add the minimum distance vertex to the shotest path tree

            shortest_path_set[u] = True

            # Update dist of the adjacent vertices of the newly added minimum distance vertex

            for v in range(self.V):
                if self.graph[u][v] > 0 and shortest_path_set[v] \
                    == False and dist[v] > dist[u] + self.graph[u][v]:
                    dist[v] = dist[u] + self.graph[u][v]
                    parent.append((u, v))

        global path
        path = []
        pqr = True
        end = dest
        path.append(dest)
        while pqr:
            path.append(self.find_parent(end))
            end = self.find_parent(end)
            if end == src:
                pqr = False
        path.reverse()


g = Graph(81)

while 1:
    (_, ooimg) = video.read()
    oimg = ooimg[10:452, 130:583]
    img = cv2.resize(oimg, (630, 630))
    img_1 = img.copy()
    img_2 = img.copy()
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    (corners, ids, _) = aruco.detectMarkers(img, aruco_dict,
            parameters=parameters)

    min_angle = 0.19
    min_dist = 200
    grab_dist = 3000

    if n_detection:

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([20, 37, 165])
        upper_yellow = np.array([35, 255, 255])

        # lower_red = np.array([0,120,70])
        # upper_red = np.array([10,255,255])

        lower_blue = np.array([85, 47, 155])
        higher_blue = np.array([179, 255, 255])

        # lower_white=np.array([0,0,157])
        # higher_white=np.array([179,51,255])

        # lower_yellow=np.array([20,37,153])
        # upper_yellow=np.array([35,255,255])

        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])

        # lower_blue = np.array([88,134,128])
        # higher_blue = np.array([179,166,255])

        lower_white = np.array([0, 0, 149])
        higher_white = np.array([179, 51, 255])
        
        lower_green = np.array([72, 74, 4])
        upper_green = np.array([90, 174, 213])

        # mask1   ->  for red
        # mask2   ->  for yellow
        # mask3   ->  for blue
        # mask4   ->  for white

        kernel = np.ones((5, 5), np.uint8)
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        # mask1=cv2.morphologyEx(mask1,cv2.MORPH_CLOSE,kernel)
        # mask1=cv2.GaussianBlur(mask1,(15,15),0)

        mask2 = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask2 = cv2.GaussianBlur(mask2, (15, 15), 0)
        
        mask3 = cv2.inRange(hsv, lower_blue, higher_blue)
        mask3 = cv2.GaussianBlur(mask3, (15, 15), 0)
        
        mask4 = cv2.inRange(hsv, lower_white, higher_white)
        mask4 = cv2.morphologyEx(mask4, cv2.MORPH_CLOSE, kernel)
        mask4 = cv2.morphologyEx(mask4, cv2.MORPH_CLOSE, kernel)
        mask4 = cv2.GaussianBlur(mask4, (15, 15), 0)
        
        mask5 = cv2.inRange(hsv, lower_green, upper_green)
        mask5 = cv2.morphologyEx(mask5, cv2.MORPH_CLOSE, kernel)
        mask5 = cv2.GaussianBlur(mask5, (15, 15), 0)
        
        lower_red = np.array([157, 53, 21])
        upper_red = np.array([179, 255, 255])
        
        mask6 = cv2.inRange(hsv, lower_red, upper_red)
        
        mask1 = mask1 + mask6
        mask1 = cv2.GaussianBlur(mask1, (15, 15), 0)

        way_points = []
        g_cells = []
        uw_box = []
        w_box = []
        w_weapons = []
        death_eaters = []
        ub_cells = []

        (contours, _) = cv2.findContours(mask2, cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_1, contours, -1, (130, 255, 0), 3)

        cv2.imshow('mask', mask2)

        for c in contours:
            M = cv2.moments(c)
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            approx = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True),
                    True)
            if cv2.contourArea(c) > 150:
                if len(approx) > 3:
                    if shape_detection(c) == 'circle':
                        shape = 'yCircle'
                    else:
                        shape = 'ySquare'
                    way_points.append([cX, cY, shape])

        for i in way_points:
            cv2.circle(img_1, (i[0], i[1]), 5, (0, 0, 0), -1)
            cv2.putText(
                img_1,
                i[2],
                (i[0] - 25, i[1] - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                )

        (contours, _) = cv2.findContours(mask1, cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_1, contours, -1, (130, 255, 0), 3)

        for c in contours:
            M = cv2.moments(c)
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True),
                    True)
            if cv2.contourArea(c) > 40:
                if len(approx) > 3:
                    if shape_detection(c) == 'circle':
                        shape = 'rCircle'
                    else:
                        shape = 'rSquare'
                way_points.append([cX, cY, shape])

        for i in way_points:
            cv2.circle(img_1, (i[0], i[1]), 5, (0, 0, 0), -1)
            cv2.putText(
                img_1,
                i[2],
                (i[0] - 25, i[1] - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                )

        (contours, _) = cv2.findContours(mask3, cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_1, contours, -1, (130, 255, 0), 3)

        for c in contours:
            M = cv2.moments(c)
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            approx = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True),
                    True)
            if cv2.contourArea(c) > 5000:
                shape = 'bSquare'
                way_points.append([cX, cY, shape])

        for i in way_points:
            cv2.circle(img_1, (i[0], i[1]), 5, (0, 0, 0), -1)
            cv2.putText(
                img_1,
                i[2],
                (i[0] - 25, i[1] - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                )

        (contours, _) = cv2.findContours(mask4, cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_1, contours, -1, (130, 255, 0), 3)
        for c in contours:
            M = cv2.moments(c)
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            w_count = 0
            approx = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True),
                    True)

            if cv2.contourArea(c) > 1500:
                shape = 'wSquare'
                print len(b_cells)
                print len(b_cells_c)

                for i in range(len(b_cells_c)):
                    if distance(b_cells_c[i], (cX, cY)) > 2000:
                        w_count += 1
                    else:

                        ub_cells.append(b_cells[i])
                        break
                if w_count >= len(b_cells):
                    way_points.append([cX, cY, shape])

        (contours, _) = cv2.findContours(mask5, cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_1, c, -1, (130, 255, 0), 3)
        for c in contours:
            M = cv2.moments(c)
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            approx = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True),
                    True)
            if cv2.contourArea(c) > 5000:
                g_cells.append((cX, cY))
                print ('green', cv2.contourArea(c))

        for i in way_points:
            cv2.circle(img_1, (i[0], i[1]), 5, (0, 0, 0), -1)
            cv2.putText(
                img_1,
                i[2],
                (i[0] - 25, i[1] - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                )

        # way_points = list(dict.fromkeys(way_points))

        # cv2.line(img, (way_points[2][0], way_points[2][1]), (way_points[3][0], way_points[3][1]), (255, 255, 255), 5)
        # cv2.imshow('image', img)

        way_points.sort(key=lambda x: x[0])
        nodes = []
        for x in range(9):
            nodes1 = sorted(way_points[0 + 9 * x:9 + 9 * x],
                            key=lambda x: x[1])
            nodes = nodes + nodes1
        print way_points
        print 'breaked'
        print nodes
        cv2.imshow('feddd', img_1)

        for i in range(len(nodes)):
            if nodes[i][2] == 'wSquare':
                w_box.append(i)

        for i in w_box:
            for j in g_cells:
                if distance((nodes[i][0], nodes[i][1]), j) < 3000:
                    death_eaters.append(i)
                    print distance((nodes[i][0], nodes[i][1]), j)

        w_weapons = [x for x in w_box if x not in death_eaters]

        b_cells = [x for x in b_cells if x not in ub_cells]

        death_eaters1 = [x for x in death_eaters]
        w_weapons1 = [x for x in w_weapons]

        # b_cells.append(12)  #remove it
        # b_cells.append(5)
        # b_cells.append(15)
        # b_cells.sort()

        # b_cells.reverse()

        n_detection = False
        path_1_bool = True
        adj = True
        print 'b_cells'
        print b_cells
        print 'Weapons!'
        print w_weapons
        print 'death_eaters!'
        print death_eaters
        print ('g_cells', g_cells)

        # break

    # adjecency_matrix

    avoid1 = 'bSquare'
    avoid = 'wSquare'

    if adj == True:
        adjacency_matrix = np.array([[0 for i in range(81)] for j in
                                    range(81)])
        for i in range(7):
            for j in range(7):
                adjacency_matrix[10 + j + 9 * i][10 + j + 9 * i - 1] = \
                    adjacency_matrix[10 + j + 9 * i][10 + j + 9 * i
                        + 1] = adjacency_matrix[10 + j + 9 * i][10 + j
                        + 9 * i - 9] = adjacency_matrix[10 + j + 9
                        * i][10 + j + 9 * i + 9] = 30
                if nodes[10 + j + 9 * i - 1][2] == weapon:

                    adjacency_matrix[10 + j + 9 * i][10 + j + 9 * i
                            - 1] = 5
                if nodes[10 + j + 9 * i + 1][2] == weapon:

                    adjacency_matrix[10 + j + 9 * i][10 + j + 9 * i
                            + 1] = 5
                if nodes[10 + j + 9 * i - 9][2] == weapon:

                    adjacency_matrix[10 + j + 9 * i][10 + j + 9 * i
                            - 9] = 5
                if nodes[10 + j + 9 * i + 9][2] == weapon:

                    adjacency_matrix[10 + j + 9 * i][10 + j + 9 * i
                            + 9] = 5
                if nodes[10 + j + 9 * i - 1][2] == avoid:

                    adjacency_matrix[10 + j + 9 * i][10 + j + 9 * i
                            - 1] = 2000
                if nodes[10 + j + 9 * i + 1][2] == avoid:

                    adjacency_matrix[10 + j + 9 * i][10 + j + 9 * i
                            + 1] = 2000
                if nodes[10 + j + 9 * i - 9][2] == avoid:

                    adjacency_matrix[10 + j + 9 * i][10 + j + 9 * i
                            - 9] = 2000
                if nodes[10 + j + 9 * i + 9][2] == avoid:

                    adjacency_matrix[10 + j + 9 * i][10 + j + 9 * i
                            + 9] = 2000
                if nodes[10 + j + 9 * i - 1][2] == avoid1:

                    adjacency_matrix[10 + j + 9 * i][10 + j + 9 * i
                            - 1] = 500
                if nodes[10 + j + 9 * i + 1][2] == avoid1:

                    adjacency_matrix[10 + j + 9 * i][10 + j + 9 * i
                            + 1] = 500
                if nodes[10 + j + 9 * i - 9][2] == avoid1:

                    adjacency_matrix[10 + j + 9 * i][10 + j + 9 * i
                            - 9] = 500
                if nodes[10 + j + 9 * i + 9][2] == avoid1:

                    adjacency_matrix[10 + j + 9 * i][10 + j + 9 * i
                            + 9] = 500

        for i in range(7):
            adjacency_matrix[i + 1][i + 1 + 1] = adjacency_matrix[i
                    + 1][i + 1 - 1] = adjacency_matrix[i + 1][i + 1
                    + 9] = 30
            if nodes[i + 1 + 1][2] == weapon:
                adjacency_matrix[i + 1][i + 1 + 1] = 5
            if nodes[i + 1 - 1][2] == weapon:
                adjacency_matrix[i + 1][i + 1 - 1] = 5
            if nodes[i + 1 + 9][2] == weapon:
                adjacency_matrix[i + 1][i + 1 + 9] = 5
            if nodes[i + 1 + 1][2] == avoid:
                adjacency_matrix[i + 1][i + 1 + 1] = 2000
            if nodes[i + 1 - 1][2] == avoid:
                adjacency_matrix[i + 1][i + 1 - 1] = 2000
            if nodes[i + 1 + 9][2] == avoid:
                adjacency_matrix[i + 1][i + 1 + 9] = 2000
            if nodes[i + 1 + 1][2] == avoid1:
                adjacency_matrix[i + 1][i + 1 + 1] = 500
            if nodes[i + 1 - 1][2] == avoid1:
                adjacency_matrix[i + 1][i + 1 - 1] = 500
            if nodes[i + 1 + 9][2] == avoid1:
                adjacency_matrix[i + 1][i + 1 + 9] = 500

        for i in range(7):
            adjacency_matrix[9 + 9 * i][9 + 9 * i - 9] = \
                adjacency_matrix[9 + 9 * i][9 + 9 * i + 9] = \
                adjacency_matrix[9 + 9 * i][9 + 9 * i + 1] = 30
            if nodes[9 + 9 * i - 9][2] == weapon:
                adjacency_matrix[9 + 9 * i][9 + 9 * i - 9] = 5
            if nodes[9 + 9 * i + 9][2] == weapon:
                adjacency_matrix[9 + 9 * i][9 + 9 * i + 9] = 5
            if nodes[9 + 9 * i + 1][2] == weapon:
                adjacency_matrix[9 + 9 * i][9 + 9 * i + 1] = 5
            if nodes[9 + 9 * i - 9][2] == avoid:
                adjacency_matrix[9 + 9 * i][9 + 9 * i - 9] = 2000
            if nodes[9 + 9 * i + 9][2] == avoid:
                adjacency_matrix[9 + 9 * i][9 + 9 * i + 9] = 2000
            if nodes[9 + 9 * i + 1][2] == avoid:
                adjacency_matrix[9 + 9 * i][9 + 9 * i + 1] = 2000
            if nodes[9 + 9 * i - 9][2] == avoid1:
                adjacency_matrix[9 + 9 * i][9 + 9 * i - 9] = 500
            if nodes[9 + 9 * i + 9][2] == avoid1:
                adjacency_matrix[9 + 9 * i][9 + 9 * i + 9] = 500
            if nodes[9 + 9 * i + 1][2] == avoid1:
                adjacency_matrix[9 + 9 * i][9 + 9 * i + 1] = 500
        for i in range(7):
            adjacency_matrix[17 + 9 * i][17 + 9 * i - 9] = \
                adjacency_matrix[17 + 9 * i][17 + 9 * i + 9] = \
                adjacency_matrix[17 + 9 * i][17 + 9 * i - 1] = 30
            if nodes[17 + 9 * i - 9][2] == weapon:
                adjacency_matrix[17 + 9 * i][17 + 9 * i - 9] = 5
            if nodes[17 + 9 * i + 9][2] == weapon:
                adjacency_matrix[17 + 9 * i][17 + 9 * i + 9] = 5
            if nodes[17 + 9 * i - 1][2] == weapon:
                adjacency_matrix[17 + 9 * i][17 + 9 * i - 1] = 5
            if nodes[17 + 9 * i - 9][2] == avoid:
                adjacency_matrix[17 + 9 * i][17 + 9 * i - 9] = 2000
            if nodes[17 + 9 * i + 9][2] == avoid:
                adjacency_matrix[17 + 9 * i][17 + 9 * i + 9] = 2000
            if nodes[17 + 9 * i - 1][2] == avoid:
                adjacency_matrix[17 + 9 * i][17 + 9 * i - 1] = 2000
            if nodes[17 + 9 * i - 9][2] == avoid1:
                adjacency_matrix[17 + 9 * i][17 + 9 * i - 9] = 500
            if nodes[17 + 9 * i + 9][2] == avoid1:
                adjacency_matrix[17 + 9 * i][17 + 9 * i + 9] = 500
            if nodes[17 + 9 * i - 1][2] == avoid1:
                adjacency_matrix[17 + 9 * i][17 + 9 * i - 1] = 500
        for i in range(7):
            adjacency_matrix[i + 73][i + 73 + 1] = adjacency_matrix[i
                    + 73][i + 73 - 1] = adjacency_matrix[i + 73][i + 73
                    - 9] = 30
            if nodes[i + 73 + 1][2] == weapon:
                adjacency_matrix[i + 73][i + 73 + 1] = 5
            if nodes[i + 73 - 1][2] == weapon:
                adjacency_matrix[i + 73][i + 73 - 1] = 5
            if nodes[i + 73 - 9][2] == weapon:
                adjacency_matrix[i + 73][i + 73 - 9] = 5
            if nodes[i + 73 + 1][2] == avoid:
                adjacency_matrix[i + 73][i + 73 + 1] = 2000
            if nodes[i + 73 - 1][2] == avoid:
                adjacency_matrix[i + 73][i + 73 - 1] = 2000
            if nodes[i + 73 - 9][2] == avoid:
                adjacency_matrix[i + 73][i + 73 - 9] = 2000
            if nodes[i + 73 + 1][2] == avoid1:
                adjacency_matrix[i + 73][i + 73 + 1] = 500
            if nodes[i + 73 - 1][2] == avoid1:
                adjacency_matrix[i + 73][i + 73 - 1] = 500
            if nodes[i + 73 - 9][2] == avoid1:
                adjacency_matrix[i + 73][i + 73 - 9] = 500

        adjacency_matrix[0][1] = adjacency_matrix[0][9] = 30
        adjacency_matrix[8][7] = adjacency_matrix[8][17] = 30
        adjacency_matrix[72][73] = adjacency_matrix[72][63] = 30
        adjacency_matrix[80][79] = adjacency_matrix[80][71] = 30

        if nodes[79][2] == avoid:
            adjacency_matrix[80][79] = 2000
        if nodes[79][2] == avoid1:
            adjacency_matrix[80][79] = 500
        if nodes[79][2] == weapon:
            adjacency_matrix[80][79] = 5

        if nodes[71][2] == avoid:
            adjacency_matrix[80][71] = 2000
        if nodes[71][2] == avoid1:
            adjacency_matrix[80][71] = 500
        if nodes[71][2] == weapon:
            adjacency_matrix[80][71] = 5

        if nodes[73][2] == avoid:
            adjacency_matrix[72][73] = 2000
        if nodes[73][2] == avoid1:
            adjacency_matrix[72][73] = 500
        if nodes[73][2] == weapon:
            adjacency_matrix[72][73] = 5

        if nodes[63][2] == avoid:
            adjacency_matrix[72][63] = 2000
        if nodes[63][2] == avoid1:
            adjacency_matrix[72][63] = 500
        if nodes[63][2] == weapon:
            adjacency_matrix[72][63] = 5

        if nodes[7][2] == avoid:
            adjacency_matrix[8][7] = 2000
        if nodes[7][2] == avoid1:
            adjacency_matrix[8][7] = 500
        if nodes[7][2] == weapon:
            adjacency_matrix[8][7] = 5

        if nodes[17][2] == avoid:
            adjacency_matrix[8][17] = 2000
        if nodes[17][2] == avoid1:
            adjacency_matrix[8][17] = 500
        if nodes[17][2] == weapon:
            adjacency_matrix[8][17] = 5

        if nodes[1][2] == avoid:
            adjacency_matrix[0][1] = 2000
        if nodes[1][2] == avoid1:
            adjacency_matrix[0][1] = 500
        if nodes[1][2] == weapon:
            adjacency_matrix[0][1] = 5

        if nodes[9][2] == avoid:
            adjacency_matrix[0][9] = 2000
        if nodes[9][2] == avoid1:
            adjacency_matrix[0][9] = 500
        if nodes[9][2] == weapon:
            adjacency_matrix[0][9] = 5

        print adjacency_matrix
        g.graph = adjacency_matrix
        adj = False

    # dijkstra

    if path_1_bool == True and death_count >= len(death_eaters):

        crop_1_bool = True
        path_1_bool = False

        # print(death_count,len(death_eaters))

        requests.get('http://192.168.43.88/s')
        crop_bool_count = 0
        weapon_count = 0

    if path_1_bool == True and death_count < len(death_eaters):

        all_paths = []
        path = []
        rnode_list = []
        node_list = []
        next_node = []

        round_1 = ((next_start, death_eaters[death_count]),
                   (death_eaters[death_count], b_cells[death_count]))
        update = True
        for i in round_1:
            g.dijkstra(i[0], i[1])
            all_paths.append(path)
            if update:
                grabbing_node = len(path) - 1
                update = False

        if dist[b_cells[death_count]] < 2000:
            print dist[19]
            print ('hey there', dist[b_cells[death_count]])
            del b_cells[death_count]
            del death_eaters[death_count]
            for j in all_paths:
                for i in range(len(j) - 1):

                    # cv2.circle(img,(nodes[j[i]][0],nodes[j[i]][1]))

                    cv2.line(img, (nodes[j[i]][0], nodes[j[i]][1]),
                             (nodes[j[i + 1]][0], nodes[j[i + 1]][1]),
                             (0 + 10 * i, 0 + 50 * i, 0 + i), 2)

            for j in all_paths:
                rnode_list = []
                for i in j:
                    rnode_list.append((nodes[i][0], nodes[i][1]))
                node_list.append(rnode_list)

            node_list[0] = node_list[0] + node_list[1]
            del node_list[1]

            # node_list[1]=node_list[1]+node_list[2]
            # del node_list[2]
            # node_list[2]=node_list[2]+node_list[3]
            # del node_list[3]

            print node_list
            next_start = path[-2]
            path_1_bool = False
            round_1_bool = True
            node_count = 0
        else:
            death_count += 1

    if round_1_bool:

        # round_1 = ((22, 24), (24, 22))
        # for i in round_1:
        # ....g.dijkstra(i[0], i[1])
        # ....all_paths.append(path)

        for j in all_paths:
            for i in range(len(j) - 1):
                # cv2.circle(img,(nodes[j[i]][0],nodes[j[i]][1]))

                cv2.line(img, (nodes[j[i]][0], nodes[j[i]][1]),
                         (nodes[j[i + 1]][0], nodes[j[i + 1]][1]), (0
                         + 10 * i, 0 + 50 * i, 0 + i), 2)

        # for j in all_paths:
        # ....rnode_list=[]
        # ....for i in j:
        # ........rnode_list.append((nodes[i][0], nodes[i][1]))
        # ....node_list.append(rnode_list)

        # node_list[0] = node_list[0] + node_list[1]
        # del node_list[1]
        # # node_list[1] = node_list[1] + node_list[2]
        # # del node_list[2]
        # # node_list[2] = node_list[2] + node_list[3]
        # # del node_list[3]
        # print(node_list)

        # imCrop = img [534:534 + 66, 0:66] + img [0:66, 530:530 + 66]
        # cv2.imshow('cropped', imCrop)

        # '''

        # aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        # parameters = aruco.DetectorParameters_create()
        # corners, ids, _ = aruco.detectMarkers(img, aruco_dict, parameters = parameters)
        # centroid = [
        # #   (corners[0][0][0] + corners[0][1][0] + corners[0][2][0] + corners[0][3][0]) / 4,
        # #    (corners[0][0][1] + corners[0][1][1] + corners[0][2][1] + corners[0][3][1]) / 4 ]

        if ids == None:
            print 'marker not detected'
            continue

        # print(corners[0][0])
        # min_angle = 0.2
        # slow_dist = 1000
        # min_dist = 15

        # locomotion
        centroid = ((corners[0][0][0][0] + corners[0][0][1][0]
                    + corners[0][0][2][0] + corners[0][0][3][0]) / 4,
                    (corners[0][0][0][1] + corners[0][0][1][1]
                    + corners[0][0][2][1] + corners[0][0][3][1]) / 4)
        next_node = node_list[0]
        releasing_node = len(next_node) - 1
        node_tbt = next_node[node_count]
        vector_aruco = (corners[0][0][0][0] - corners[0][0][3][0],
                        corners[0][0][0][1] - corners[0][0][3][1])
        vector_node = (node_tbt[0] - centroid[0], node_tbt[1]
                       - centroid[1])
        angle = angle_between(vector_node, vector_aruco)
        if distance(centroid, node_tbt) < min_dist:
            requests.get('http://192.168.43.88/s')
            node_count = node_count + 1
            print 'stop'

        if angle < min_angle and distance(centroid, node_tbt) \
            > min_dist and node_count == grabbing_node \
            and distance(centroid, node_tbt) > grab_dist:
            requests.get('http://192.168.43.88/F')
            print distance(centroid, node_tbt)
            print 'about to grab'

        if angle < min_angle and distance(centroid, node_tbt) \
            > min_dist and node_count == grabbing_node \
            and distance(centroid, node_tbt) <= grab_dist:
            requests.get('http://192.168.43.88/s')
            requests.get('http://192.168.43.88/g')
            node_count = node_count + 1
            print distance(centroid, node_tbt)
            print 'grab'

        if angle < min_angle and distance(centroid, node_tbt) \
            > min_dist and node_count == releasing_node:
            requests.get('http://192.168.43.88/boxrel')
            print 'boxrel'

            round_1_bool = False
            path_1_bool = True
            print (death_count, '''







''')
            node_count = 0

        if angle < min_angle and distance(centroid, node_tbt) \
            > min_dist and node_count != releasing_node and node_count \
            != grabbing_node:
            requests.get('http://192.168.43.88/F')
            print 'forward'

        if angle > min_angle and distance(corners[0][0][0], node_tbt) \
            > distance(corners[0][0][1], node_tbt) \
            and distance(centroid, node_tbt) > min_dist:
            requests.get('http://192.168.43.88/r')
            print 'right'

        if angle > min_angle and distance(corners[0][0][0], node_tbt) \
            <= distance(corners[0][0][1], node_tbt) \
            and distance(centroid, node_tbt) > min_dist:
            requests.get('http://192.168.43.88/l')
            print 'left'
        print distance(centroid, node_tbt)

    # crop_1_bool = True    #remove it!!!!!!!!!!!!!!!!!

    if crop_1_bool == True and crop_bool_count >= len(death_eaters1):

        crop_1_bool = False
        path_2_bool = True

        # remove it!!!!!

    if crop_1_bool == True and crop_bool_count < len(death_eaters1) \
        and death_eaters1[crop_bool_count]:
        shape = 'wSquare'
        print death_count
        print ('Kill', nodes[death_eaters1[crop_bool_count]][1],
               nodes[death_eaters1[crop_bool_count]][0])
        im_reveal_0 = img_2[nodes[death_eaters1[crop_bool_count]][1]
                            - 22:
                            nodes[death_eaters1[crop_bool_count]][1]
                            + 22,
                            nodes[death_eaters1[crop_bool_count]][0]
                            - 22:
                            nodes[death_eaters1[crop_bool_count]][0]
                            + 22]

        cv2.imshow('as', im_reveal_0)
        time.sleep(2)

        # cv2.imshow('sd', im_reveal_0)

        im_reveal_0_hsv = cv2.cvtColor(im_reveal_0, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([20, 37, 165])
        upper_yellow = np.array([35, 255, 255])
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])

        # mask_1  ->  for red
        # mask    ->  for yellow

        mask1 = cv2.inRange(im_reveal_0_hsv, lower_red, upper_red)
        # mask1=cv2.GaussianBlur(mask1,(15,15),0)

        mask2 = cv2.inRange(im_reveal_0_hsv, lower_yellow, upper_yellow)
        mask2 = cv2.GaussianBlur(mask2, (15, 15), 0)
        
        lower_red = np.array([157, 53, 21])
        upper_red = np.array([179, 255, 255])
        
        mask6 = cv2.inRange(im_reveal_0_hsv, lower_red, upper_red)
        
        mask1 = mask1 + mask6
        mask1 = cv2.GaussianBlur(mask1, (15, 15), 0)

        (contours, _) = cv2.findContours(mask2, cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im_reveal_0, contours, -1, (130, 255, 0), 3)

        # cv2.imshow('modified', mask)

        for c in contours:
            M = cv2.moments(c)
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            approx = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True),
                    True)
            if len(approx) > 3:

            # ....shape = 'yCircle'
            # else:
            # ....shape = 'ySquare'

                if shape_detection(c) == 'circle':
                    shape = 'yCircle'
                else:
                    shape = 'ySquare'
            cv2.circle(im_reveal_0, (cX, cY), 5, (0, 0, 0), -1)
            cv2.putText(
                im_reveal_0,
                shape,
                (cX - 25, cY - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                )

        (contours, _) = cv2.findContours(mask1, cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im_reveal_0, contours, -1, (130, 255, 0), 3)

        # cv2.imshow('modified', mask)

        for c in contours:
            M = cv2.moments(c)
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True),
                    True)
            if len(approx) > 3:
                if shape_detection(c) == 'circle':
                    shape = 'rCircle'
                else:
                    shape = 'rSquare'

            cv2.circle(im_reveal_0, (cX, cY), 5, (0, 0, 0), -1)
            cv2.putText(
                im_reveal_0,
                shape,
                (cX - 25, cY - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                )
        cv2.imshow('horcrux', im_reveal_0)
        nodes[death_eaters1[crop_bool_count]][2] = shape
        crop_bool_count += 1
        time.sleep(1)  # remove it....!!!!!

    if path_2_bool == True and weapon_count >= len(w_weapons):
        path_1_bool = True
        death_count = 0
        path_2_bool = False

    if path_2_bool == True and weapon_count < len(w_weapons):
        all_paths = []
        path = []
        rnode_list = []
        node_list = []
        next_node = []
        g.dijkstra(next_start, w_weapons[weapon_count])
        print ('hey there', path)
        non = len(path)
        all_paths.append(path)

        for j in all_paths:
            for i in range(len(j) - 1):
                cv2.line(img, (nodes[j[i]][0], nodes[j[i]][1]),
                         (nodes[j[i + 1]][0], nodes[j[i + 1]][1]), (0
                         + 10 * i, 0 + 50 * i, 0 + i), 2)

        for j in all_paths:
            rnode_list = []
            for i in j:
                rnode_list.append((nodes[i][0], nodes[i][1]))
            node_list.append(rnode_list)
        round_2_bool = True
        path_2_bool = False
        next_start = path[-2]
        print node_list

    # round_2
    # round_2_bool = True

    if round_2_bool == True:

        # # round_2 = ((9, 18))
        # g.dijkstra(79, 2)
        # all_paths.append(path)

        for j in all_paths:
            for i in range(len(j) - 1):
                cv2.line(img, (nodes[j[i]][0], nodes[j[i]][1]),
                         (nodes[j[i + 1]][0], nodes[j[i + 1]][1]), (0
                         + 10 * i, 0 + 50 * i, 0 + i), 2)

        # for j in all_paths:
        # ....rnode_list = []
        # ....for i in j:
        # ........rnode_list.append((nodes[i][0], nodes[i][1]))
        # ....node_list.append(rnode_list)

            # locomotion

        # aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_100)
        # parameters = aruco.DetectorParameters_create()
        # corners, ids, _ = aruco.detectMarkers(img, aruco_dict, parameters = parameters)
        # centroid = [
        # #   (corners[0][0][0] + corners[0][1][0] + corners[0][2][0] + corners[0][3][0]) / 4,
        # #   (corners[0][0][1] + corners[0][1][1] + corners[0][2][1] + corners[0][3][1]) / 4 ]

        if ids == None:
            print 'Marker not detected'
            continue

        # print(corners[0][0])
        # min_angle = 0.2
        # slow_dist = 1000
        # min_dist  = 15

        centroid = ((corners[0][0][0][0] + corners[0][0][1][0]
                    + corners[0][0][2][0] + corners[0][0][3][0]) / 4,
                    (corners[0][0][0][1] + corners[0][0][1][1]
                    + corners[0][0][2][1] + corners[0][0][3][1]) / 4)
        next_node = node_list[0]
        node_tbt = next_node[node_count]
        vector_aruco = (corners[0][0][0][0] - corners[0][0][3][0],
                        corners[0][0][0][1] - corners[0][0][3][1])
        vector_node = (node_tbt[0] - centroid[0], node_tbt[1]
                       - centroid[1])
        angle = angle_between(vector_node, vector_aruco)
        if distance(centroid, node_tbt) < min_dist:
            requests.get('http://192.168.43.88/s')
            node_count = node_count + 1
            print 'stop'

        if angle < min_angle and distance(centroid, node_tbt) \
            > min_dist and node_count == non - 1 and distance(centroid,
                node_tbt) > grab_dist:
            requests.get('http://192.168.43.88/F')

        if angle < min_angle and distance(centroid, node_tbt) \
            > min_dist and node_count == non - 1 and distance(centroid,
                node_tbt) <= grab_dist:
            requests.get('http://192.168.43.88/s')
            requests.get('http://192.168.43.88/g')
            requests.get('http://192.168.43.88/b')
            time.sleep(3)
            requests.get('http://192.168.43.88/s')
            requests.get('http://192.168.43.88/blue')
            print 'revealed!'

            # vary!!!!!!!!!!!!!

            crop_2_bool = True
            node_count = 0
            round_2_bool = False

            # time.sleep(10)

            continue

        if angle < min_angle and distance(centroid, node_tbt) \
            > min_dist and node_count != non - 1:
            requests.get('http://192.168.43.88/F')
            print 'forward'

        if angle > min_angle and distance(corners[0][0][0], node_tbt) \
            > distance(corners[0][0][1], node_tbt) \
            and distance(centroid, node_tbt) > min_dist:
            requests.get('http://192.168.43.88/r')
            print 'right'

        if angle > min_angle and distance(corners[0][0][0], node_tbt) \
            <= distance(corners[0][0][1], node_tbt) \
            and distance(centroid, node_tbt) > min_dist:
            requests.get('http://192.168.43.88/l')
            print 'left'
        print distance(centroid, node_tbt)

    if crop_2_bool == True:
        im_reveal_0 = img_2[nodes[w_weapons[weapon_count]][1] - 22:
                            nodes[w_weapons[weapon_count]][1] + 22,
                            nodes[w_weapons[weapon_count]][0] - 22:
                            nodes[w_weapons[weapon_count]][0] + 22]

        # cv2.imshow('sd',img_1)

        im_reveal_0_hsv = cv2.cvtColor(im_reveal_0, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([20, 37, 165])
        upper_yellow = np.array([35, 255, 255])
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])

        # mask1   ->  for red
        # mask    ->  for yellow

        mask1 = cv2.inRange(im_reveal_0_hsv, lower_red, upper_red)

        # mask1 = cv2.GaussianBlur(mask1, (15, 15), 0)

        mask2 = cv2.inRange(im_reveal_0_hsv, lower_yellow, upper_yellow)
        mask2 = cv2.GaussianBlur(mask2, (15, 15), 0)

        lower_red = np.array([157, 53, 21])
        upper_red = np.array([179, 255, 255])
        
        mask6 = cv2.inRange(im_reveal_0_hsv, lower_red, upper_red)
        
        mask1 = mask1 + mask6
        mask1 = cv2.GaussianBlur(mask1, (15, 15), 0)

        (contours, _) = cv2.findContours(mask2, cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im_reveal_0, contours, -1, (130, 255, 0), 3)

        # cv2.imshow('modified',mask)

        for c in contours:
            M = cv2.moments(c)
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            approx = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True),
                    True)
            if cv2.contourArea(c) > 40:
                if shape_detection(c) == 'circle':
                    weapon = shape = 'yCircle'
                else:
                    weapon = shape = 'ySquare'

            cv2.circle(im_reveal_0, (cX, cY), 5, (0, 0, 0), -1)
            cv2.putText(
                im_reveal_0,
                shape,
                (cX - 25, cY - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                )

        (contours, _) = cv2.findContours(mask1, cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im_reveal_0, contours, -1, (130, 255, 0), 3)

        # cv2.imshow('modified',mask)

        for c in contours:
            M = cv2.moments(c)
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            approx = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True),
                    True)
            if cv2.contourArea(c) > 40:
                if shape_detection(c) == 'circle':
                    weapon = shape = 'rCircle'
                else:
                    weapon = shape = 'rSquare'

            cv2.circle(im_reveal_0, (cX, cY), 5, (0, 0, 0), -1)
            cv2.putText(
                im_reveal_0,
                shape,
                (cX - 25, cY - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                )
        nodes[w_weapons[weapon_count]][2] = shape
        cv2.imshow('weapon', im_reveal_0)

        adj = True
        crop_2_bool = False
        match = True
        continue

    if match == True:
        if nodes[w_weapons[weapon_count]][2] == nodes[27][2]:
            end1 = 27
            path_3_bool = True
            match = False
            continue

        if nodes[w_weapons[weapon_count]][2] == nodes[45][2]:
            end1 = 45
            path_3_bool = True
            match = False
            continue

        if nodes[w_weapons[weapon_count]][2] == nodes[58][2]:
            end1 = 58
            path_3_bool = True
            match = False
            continue

        if nodes[w_weapons[weapon_count]][2] == nodes[22][2]:
            end1 = 22
            path_3_bool = True
            match = False
            continue

    if path_3_bool == True:
        node_list = []
        rnode_list = []
        all_paths = []
        path = []
        next_node = []
        g.dijkstra(w_weapons[weapon_count], end1)
        all_paths.append(path)
        del w_weapons[weapon_count]

        for j in all_paths:
            for i in range(len(j) - 1):
                cv2.line(img, (nodes[j[i]][0], nodes[j[i]][1]),
                         (nodes[j[i + 1]][0], nodes[j[i + 1]][1]), (0
                         + 10 * i, 0 + 50 * i, 0 + i), 2)

        for j in all_paths:
            rnode_list = []
            for i in j:
                rnode_list.append((nodes[i][0], nodes[i][1]))
            node_list.append(rnode_list)
        print node_list
        path_3_bool = False
        round_3_bool = True
        next_start = path[-2]

    if round_3_bool == True:

        # round_2 = ((9, 18))
        # g.dijkstra(2, end1)
        # all_paths.append(path)

        for j in all_paths:
            for i in range(len(j) - 1):
                cv2.line(img, (nodes[j[i]][0], nodes[j[i]][1]),
                         (nodes[j[i + 1]][0], nodes[j[i + 1]][1]), (0
                         + 10 * i, 0 + 50 * i, 0 + i), 2)

        # for j in all_paths:
        # ....rnode_list = []
        # ....for i in j:
        # ........rnode_list.append((nodes[i][0], nodes[i][1]))
        # ....node_list.append(rnode_list)
        # print(node_list)
            # locomotion

        # aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_100)
        # parameters = aruco.DetectorParameters_create()
        # corners, ids, _ = aruco.detectMarkers(img, aruco_dict, parameters = parameters)
        # centroid = [
        # #   (corners[0][0][0] + corners[0][1][0] + corners[0][2][0] + corners[0][3][0]) / 4,
        # #   (corners[0][0][1] + corners[0][1][1] + corners[0][2][1] + corners[0][3][1]) / 4 ]

        if ids == None:
            print 'marker not detected'
            continue

        # print(corners[0][0])
        # min_angle = 0.2
        # slow_dist = 1000
        # min_dist  = 15

        centroid = ((corners[0][0][0][0] + corners[0][0][1][0]
                    + corners[0][0][2][0] + corners[0][0][3][0]) / 4,
                    (corners[0][0][0][1] + corners[0][0][1][1]
                    + corners[0][0][2][1] + corners[0][0][3][1]) / 4)
        next_node = node_list[0]
        non = len(next_node)

        # non ->  no. of nodes

        node_tbt = next_node[node_count]
        vector_aruco = (corners[0][0][0][0] - corners[0][0][3][0],
                        corners[0][0][0][1] - corners[0][0][3][1])
        vector_node = (node_tbt[0] - centroid[0], node_tbt[1]
                       - centroid[1])
        angle = angle_between(vector_node, vector_aruco)
        if distance(centroid, node_tbt) < min_dist:
            requests.get('http://192.168.43.88/s')
            node_count = node_count + 1
            print 'stop'

        if angle < min_angle and distance(centroid, node_tbt) \
            > min_dist and node_count == non - 1:
            requests.get('http://192.168.43.88/boxrel')
            print 'released!'

            # time.sleep(10)    #vary!!!!!!!!!!!!!!!

            round_3_bool = False
            node_count = 0
            death_count = 0
            path_1_bool = True
            requests.get('http://192.168.43.88/blue')
            nodes[end1][2] = 'wSquare'
            adj = True
            if len(death_eaters) == 0 and len(w_weapons) == 0:
                requests.get('http://192.168.43.88/blue')
                requests.get('http://192.168.43.88/blue')
                break

        if angle < min_angle and distance(centroid, node_tbt) \
            > min_dist and node_count != non - 1:
            requests.get('http://192.168.43.88/F')
            print 'forward'

        if angle > min_angle and distance(corners[0][0][0], node_tbt) \
            > distance(corners[0][0][1], node_tbt) \
            and distance(centroid, node_tbt) > min_dist:
            requests.get('http://192.168.43.88/r')
            print 'right'

        if angle > min_angle and distance(corners[0][0][0], node_tbt) \
            <= distance(corners[0][0][1], node_tbt) \
            and distance(centroid, node_tbt) > min_dist:
            requests.get('http://192.168.43.88/l')
            print 'left'
        print distance(centroid, node_tbt)

    cv2.imshow('feed', img)
    k = cv2.waitKey(1)
    if k == ord('q'):
        requests.get('http://192.168.43.88/s')
        cv2.destroyAllWindows()
        video.release()
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
