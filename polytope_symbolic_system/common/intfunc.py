# Author: Joao Moura
# Contact: jpousad@ed.ac.uk
# Date: 15/12/2020
# -------------------------------------------------------------------
# Description:
# 
# Integration functions based on scipy integrate and symbolic casadi
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Import libraries
# -------------------------------------------------------------------
import numpy as np
from scipy import integrate
import casadi as cs

# -------------------------------------------------------------------
# python integration lambda functions (1D and 2D)
# -------------------------------------------------------------------
square_np = lambda sq_side: integrate.dblquad(lambda x, y: np.sqrt((x**2)
    + (y**2)), - sq_side/2, sq_side/2, -sq_side/2, sq_side/2)[0]
quad_np = lambda sq_side: integrate.quad(lambda var: var**2,
    - sq_side/2, sq_side/2)[0]
# -------------------------------------------------------------------
# casadi auxiliary variables
# -------------------------------------------------------------------
# Fixed step Runge-Kutta 4 integrator
M = 4  # RK4 steps per interval
N = 4  # number of control intervals
sLenght = cs.SX.sym('sLenght')
xLenght = cs.SX.sym('xLenght')
yLenght = cs.SX.sym('yLenght')
x = cs.SX.sym('x')
y = cs.SX.sym('y')
DX = xLenght/(N*M)
DY = yLenght/(N*M)
# -------------------------------------------------------------------
# 1D casadi integration of g
# integrand y'=f(x)
# integrate x^2 dx for x=-xL/2..xL/2
# h equals to DX and DY for x and y separately
# cost function
g = cs.Function('g_ext', [x], [DX, (x**2)*DX])
Q = 0  # initialize cost
xx = -xLenght/2  # initialize initial cond
for n in range(N):
    for m in range(M):
        k1, k1_q = g(xx)
        k2, k2_q = g(xx + k1/2)
        k3, k3_q = g(xx + k2/2)
        k4, k4_q = g(xx + k3)
        Q += (k1_q + 2*k2_q + 2*k3_q + k4_q)/6
        xx += (k1 + 2*k2 + 2*k3 + k4)/6
quad_cs = cs.Function('quad_cs', [xLenght], [Q])
# -------------------------------------------------------------------
# 2D casadi integration of g
# integrand g'=f(x, y)
# integrate sqrt(x^2+y^2) dxdy for x=-xL/2..xL/2, y=-yL/2..yL/2
# h equals to DX and DY for x and y separately
g = cs.Function('h_ext', [x, y], [DX, DY, (cs.sqrt((x**2)+(y**2)))*DX*DY])
Q = 0  # initialize cost
yy = -yLenght/2  # initialize initial cond
for ny in range(N):
    for my in range(M):
        xx = -xLenght/2
        for nx in range(N):
            for mx in range(M):
                k1_x, k1_y, k1_q = g(xx, yy)
                k2_x, k2_y, k2_q = g(xx + k1_x/2, yy + k1_y/2)
                k3_x, k3_y, k3_q = g(xx + k2_x/2, yy + k2_y/2)
                k4_x, k4_y, k4_q = g(xx + k3_x, yy + k3_y)
                Q += (k1_q + 2*k2_q + 2*k3_q + k4_q)/6
                xx += (k1_x + 2*k2_x + 2*k3_x + k4_x)/6
        yy += (k1_y + 2*k2_y + 2*k3_y + k4_y)/6
rect_cs = cs.Function('rect_cs', [xLenght, yLenght], [Q])
square_cs = cs.Function('square_cs', [sLenght], [rect_cs(sLenght, sLenght)])

# -------------------------------------------------------------------
# 2D casadi integration of g
# integrand g'=f(x, y)
# integrate sqrt(x^2+y^2) dxdy for (x, y) in a polygon
# h equals to DX and DY for x and y separately
def in_polygon(x, y, pts):
    """
    Check if a point is inside a polygon using CasADi's SX symbols

    :param x: SX symbol, x coordinate of the point
    :param y: SX symbol, y coordinate of the point
    :param pts: SX symbol, vertices of the polygon as an array of shape (n, 2)

    :return inside: SX symbol, 1 if the point is inside the polygon, 0 otherwise
    """
    n = pts.shape[0]
    inside = cs.SX(0)  # Initialize as False (0)

    for i in range(n):
        x1, y1 = pts[i, 0], pts[i, 1]
        x2, y2 = pts[(i + 1) % n, 0], pts[(i + 1) % n, 1]
        
        # Check if the edge crosses the horizontal line at y
        condition1 = cs.if_else(cs.logic_and(y1 <= y, y < y2), 1, 0)
        condition2 = cs.if_else(cs.logic_and(y2 <= y, y < y1), 1, 0)
        edge_crosses = cs.logic_or(condition1, condition2) 
        
        # Compute the x coordinate of the intersection point
        x_intersect = (x2 - x1) * (y - y1) / (y2 - y1) + x1
        
        # Check if x is to the left of the intersection point
        left_of_intersection = cs.if_else(x < x_intersect, 1, 0)
        
        # Update inside state if edge is crossed
        inside = cs.fmod(inside + edge_crosses * left_of_intersection, 2)
    
    return inside

def RungeKutta4_Integrator_Polygon(g, pts, x_len, y_len, return_func_name):
    """
    Runge-Kutta 4 integrator inside a polygon area

    :param g: the integrand function. type: cs.Function
    :param pts: the vertices of the polygon. type: cs.SX, shape: (n, 2)
    :param x_len: the length of the x axis. type: cs.SX
    :param y_len: the length of the y axis. type: cs.SX
    :param return_func_name: the name of the function to return. type: str

    :return integ_func: the integration function. type: cs.Function
    """
    Q = 0  # initialize cost
    yy = -y_len/2  # initialize initial cond
    for ny in range(N):
        for my in range(M):
            xx = -x_len/2
            for nx in range(N):
                for mx in range(M):
                    k1_x, k1_y, k1_q = g(xx, yy)
                    k2_x, k2_y, k2_q = g(xx + k1_x/2, yy + k1_y/2)
                    k3_x, k3_y, k3_q = g(xx + k2_x/2, yy + k2_y/2)
                    k4_x, k4_y, k4_q = g(xx + k3_x, yy + k3_y)
                    Q += (k1_q + 2*k2_q + 2*k3_q + k4_q)/6
                    xx += (k1_x + 2*k2_x + 2*k3_x + k4_x)/6
            yy += (k1_y + 2*k2_y + 2*k3_y + k4_y)/6
    
    integ_func = cs.Function(return_func_name, [pts], [Q])
    return integ_func

def poly_area(pts):
    max_x, min_x, max_y, min_y = cs.mmax(pts[:, 0]), cs.mmin(pts[:, 0]), cs.mmax(pts[:, 1]), cs.mmin(pts[:, 1])
    x_len = max_x - min_x
    y_len = max_y - min_y
    DX = x_len/(N*M)
    DY = y_len/(N*M)
    g = cs.Function('h_ext', [x, y], [DX, DY, in_polygon(x, y, pts)*DX*DY])

    return RungeKutta4_Integrator_Polygon(g, pts, x_len, y_len, 'poly_area')

def poly_cs(pts):
    max_x, min_x, max_y, min_y = cs.mmax(pts[:, 0]), cs.mmin(pts[:, 0]), cs.mmax(pts[:, 1]), cs.mmin(pts[:, 1])
    x_len = max_x - min_x
    y_len = max_y - min_y
    DX = x_len/(N*M)
    DY = y_len/(N*M)
    g = cs.Function('h_ext', [x, y], [DX, DY, in_polygon(x, y, pts)*cs.sqrt((x**2)+(y**2))*DX*DY])

    return RungeKutta4_Integrator_Polygon(g, pts, x_len, y_len, 'poly_cs')
