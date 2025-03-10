import warnings
import numpy as np
# Scipy
try:
    import scipy.linalg as spa
except:
    warnings.warn("You don't have scipy package installed. You may get error while using some feautures.")

# Pydrake
try:
    import pydrake
    # import pydrake.solvers.mathematicalprogram as MP
    # import pydrake.solvers.gurobi as Gurobi_drake
    # import pydrake.solvers.osqp as OSQP_drake
    from pydrake.solvers import MathematicalProgram, Solve
    from pydrake.solvers import GurobiSolver, OsqpSolver
    from pydrake.solvers import SolverOptions
    # use Gurobi solver
    global gurobi_solver,OSQP_solver, license
    gurobi_solver=GurobiSolver()
    license = gurobi_solver.AcquireLicense()
    OSQP_solver=OsqpSolver()
except:
    warnings.warn("You don't have pydrake installed properly. Methods that rely on optimization may fail.")
    

# Pypolycontain
try:
    from pypolycontain.lib.objects import Box,hyperbox,H_polytope,AH_polytope
except:
    warnings.warn("You don't have pypolycontain properly installed. Can not execute 'import pypyplycontain'")


# Polytope Symbolic System
try:
    from polytope_symbolic_system.common.utils import *
except:
    warnings.warn("You don't have polytope_symbolic_system properly installed. Can not execute 'import polytope_symbolic_system")


def to_AH_polytope(P):
    """
    Converts the polytopic object P into an AH-polytope
    """
    #This file is used
    # print(f"P name:{P.__name__}")
    if P.__name__=="AH_polytope":#P.__name__=="AH_polytope":
        return P
    elif P.__name__=="H_polytope":
        n=P.H.shape[1]
        return AH_polytope(np.eye(n),np.zeros((n,1)),P)
    elif P.__name__=="zonotope":
        q=P.G.shape[1]  #shape 0??
        return AH_polytope(P.G,P.x,Box(N=q),color=P.color)
    else:
        raise ValueError("P type not understood:",P.type)

"""
Optimization-based Operations:
"""  
      
def point_membership(Q,x,tol=10**-5,solver="gurobi"):
    if type(Q).__name__=="H_polytope":
        return Q.if_inside(x,tol)
    else:
        Q=to_AH_polytope(Q)
        prog=MathematicalProgram()
        zeta=prog.NewContinuousVariables(Q.P.H.shape[1],1,"zeta")
        prog.AddLinearConstraint(A=Q.P.H,ub=Q.P.h+tol,lb=-np.inf*np.ones((Q.P.h.shape[0],1)),vars=zeta)
        prog.AddLinearEqualityConstraint(Q.T,x-Q.t,zeta)
        if solver=="gurobi":
            result=gurobi_solver.Solve(prog,None,None)
        elif solver=="osqp":
            prog.AddQuadraticCost(np.eye(zeta.shape[0]),np.zeros(zeta.shape),zeta)
            result=OSQP_solver.Solve(prog,None,None)
        else:
            result=Solve(prog)
    return result.is_success()

def point_membership_fuzzy(Q,x,tol=10**-5,solver="gurobi"):
    """
    Fuzzy membership check. If x contains NaN, the entry is unconstrained
    @param Q: Polytope in R^n
    @param x: n*1 numpy array, may contain NaNs
    @param tol:
    @param solver: solver to use
    @return: boolean of whether x is in Q
    """
    Q=to_AH_polytope(Q)
    prog=MathematicalProgram()
    zeta=prog.NewContinuousVariables(Q.P.H.shape[1],1,"zeta")
    prog.AddLinearConstraint(A=Q.P.H,ub=Q.P.h+tol,lb=-np.inf*np.ones((Q.P.h.shape[0],1)),vars=zeta)
    assert(x.shape[1]==1)
    for i, xi in enumerate(x):
        if not np.isnan(xi):
            prog.AddLinearEqualityConstraint(np.atleast_2d(Q.T[i,:]),(x[i]-Q.t[i]).reshape([-1,1]),zeta)
    if solver=="gurobi":
        result=gurobi_solver.Solve(prog,None,None)
    elif solver=="osqp":
        prog.AddQuadraticCost(np.eye(zeta.shape[0]),np.zeros(zeta.shape),zeta)
        result=OSQP_solver.Solve(prog,None,None)
    else:
        result=Solve(prog)
    return result.is_success()

def check_non_empty(Q,tol=10**-5,solver="gurobi"):
    Q=to_AH_polytope(Q)
    prog=MathematicalProgram()
    zeta=prog.NewContinuousVariables(Q.P.H.shape[1],1,"zeta")
    prog.AddLinearConstraint(A=Q.P.H,ub=Q.P.h+tol,lb=-np.inf*np.ones((Q.P.h.shape[0],1)),vars=zeta)
    if solver=="gurobi":
            result=gurobi_solver.Solve(prog,None,None)
    elif solver=="osqp":
        prog.AddQuadraticCost(np.eye(zeta.shape[0]),np.zeros(zeta.shape),zeta)
        result=OSQP_solver.Solve(prog,None,None)
    else:
        result=Solve(prog)
    return result.is_success()

def directed_Hausdorff_distance(Q1,Q2,ball="infinty_norm",solver="gurobi"):
    r"""
    Computes the directed Hausdorff distance of Q_1 and Q_2 (AH_polytopes)
    ***************************************************************************
    The optimization problem is:
                        Minimize    epsilon  
                        such that   Q1 \subset Q2+epsilon(Ball)
                        
    It is zero if and only if Q1 subset Q2. The method is based on 
                
                    Sadraddini&Tedrake, 2019, CDC (available on ArXiv)
                    
    We solve the following problem:
        D*ball+Q1 subset Q2
    We solve the following linear program:
    ..math::
        \min     D
        s.t.    Lambda_1 H_1=H_2 Gamma_1
                Lambda_2 H_1=H_ball Gamma_2
                Lambda_1 h_1<=h_2 + H_2 beta_1
                Lambda_2 h_2<=D h_ball + H_ball beta_2
                x_2 - X_2 beta_1 - beta_2 = x_1
                X_2 Gamma_1 + Gamma_2 = X_1
    ***************************************************************************
    """
    Q1,Q2=to_AH_polytope(Q1),to_AH_polytope(Q2)
    n=Q1.t.shape[0]
    if ball=="infinty_norm":
        HB=np.vstack((np.eye(n),-np.eye(n)))
        hB=np.vstack((np.ones((n,1)),np.ones((n,1))))
    elif ball=="l1":
        HB,hb=make_ball(ball)
    prog=MathematicalProgram()
    # Variables
    D=prog.NewContinuousVariables(1,1,"D")
    Lambda_1=prog.NewContinuousVariables(Q2.P.H.shape[0],Q1.P.H.shape[0],"Lambda_1")
    Lambda_2=prog.NewContinuousVariables(HB.shape[0],Q1.P.H.shape[0],"Lambda2")
    Gamma_1=prog.NewContinuousVariables(Q2.P.H.shape[1],Q1.P.H.shape[1],"Gamma1")
    Gamma_2=prog.NewContinuousVariables(HB.shape[1],Q1.P.H.shape[1],"Gamma1")
    beta_1=prog.NewContinuousVariables(Q2.P.H.shape[1],1,"beta1")
    beta_2=prog.NewContinuousVariables(HB.shape[1],1,"beta1")
    # Constraints
    # Lambda_1 and Lambda_2 positive
    prog.AddBoundingBoxConstraint(0,np.inf,Lambda_1)
    prog.AddBoundingBoxConstraint(0,np.inf,Lambda_2)
    # Lambda_1 H_1
    Lambda_H_Gamma(prog,Lambda_1,Q1.P.H,Q2.P.H,Gamma_1)
    # Lambda_2 H_1
    Lambda_H_Gamma(prog,Lambda_2,Q1.P.H,HB,Gamma_2)
    # Lambda_1 h_1
    Lambda_h_Inequality(prog,Lambda_1,beta_1,Q2.P.H,Q1.P.h,Q2.P.h)
    # Lambda_2 h_1
    Lambda_h_Inequality_D(prog,Lambda_2,beta_2,HB,Q1.P.h,hB,D)
    # X2 beta_1   
    prog.AddLinearEqualityConstraint(-np.hstack((Q2.T,np.eye(n))),Q1.t-Q2.t,np.vstack((beta_1,beta_2)))
    # X2 Gamma_1
    Aeq=np.hstack((Q2.T,np.eye(Q2.T.shape[0])))
    for i in range(Gamma_1.shape[1]):
        beq=Q1.T[:,i]
        var=np.hstack((Gamma_1[:,i],Gamma_2[:,i]))
        prog.AddLinearEqualityConstraint(Aeq,beq,var)
    # Cost
    # Optimize
    if solver=="gurobi":
            prog.AddLinearCost(D[0,0])
            result=gurobi_solver.Solve(prog,None,None)
    elif solver=="osqp":
        prog.AddQuadraticCost(D[0,0]*D[0,0])
        result=OSQP_solver.Solve(prog,None,None)
    else:
        result=Solve(prog)
    if result.is_success():
        return np.asscalar(result.GetSolution(D))

def Hausdorff_distance(Q1,Q2,ball="infinty_norm",solver="gurobi"):
    return max(directed_Hausdorff_distance(Q1,Q2,ball,solver),directed_Hausdorff_distance(Q2,Q1,ball,solver))
    
def distance_polytopes(Q1,Q2,ball="infinity",solver="gurobi"):
    Q1,Q2=to_AH_polytope(Q1),to_AH_polytope(Q2)
    n=Q1.n
    prog=MathematicalProgram()
    zeta1=prog.NewContinuousVariables(Q1.P.H.shape[1],1,"zeta1")
    zeta2=prog.NewContinuousVariables(Q2.P.H.shape[1],1,"zeta2")
    delta=prog.NewContinuousVariables(n,1,"delta")
    prog.AddLinearConstraint(A=Q1.P.H,ub=Q1.P.h,lb=-np.inf*np.ones((Q1.P.h.shape[0],1)),vars=zeta1)
    prog.AddLinearConstraint(A=Q2.P.H,ub=Q2.P.h,lb=-np.inf*np.ones((Q2.P.h.shape[0],1)),vars=zeta2)
    prog.AddLinearEqualityConstraint( np.hstack((Q1.T,-Q2.T,np.eye(n))),Q2.t-Q1.t,np.vstack((zeta1,zeta2,delta)) )
    if ball=="infinity":
        delta_abs=prog.NewContinuousVariables(1,1,"delta_abs")
        prog.AddBoundingBoxConstraint(0,np.inf,delta_abs)
        prog.AddLinearConstraint(np.greater_equal( np.dot(np.ones((n,1)),delta_abs),delta,dtype='object' ))
        prog.AddLinearConstraint(np.greater_equal( np.dot(np.ones((n,1)),delta_abs),-delta,dtype='object' ))
        cost=delta_abs
    elif ball=="l1":
        delta_abs=prog.NewContinuousVariables(n,1,"delta_abs")
        prog.AddBoundingBoxConstraint(0,np.inf,delta_abs)
        prog.AddLinearConstraint(np.greater_equal( delta_abs,delta,dtype='object' ))
        prog.AddLinearConstraint(np.greater_equal( delta_abs,-delta,dtype='object' ))
        cost=np.dot(np.ones((1,n)),delta_abs)
    else:
        raise NotImplementedError
    if solver=="gurobi":
        prog.AddLinearCost(cost[0,0])
        result=gurobi_solver.Solve(prog,None,None)
    elif solver=="osqp":
        prog.AddQuadraticCost(cost[0,0]*cost[0,0])
        result=OSQP_solver.Solve(prog,None,None)
    else:
        prog.AddLinearCost(cost[0,0])
        result=Solve(prog)
    if result.is_success():
        return np.sum(result.GetSolution(delta_abs)),\
            np.dot(Q1.T,result.GetSolution(zeta1).reshape(zeta1.shape[0],1))+Q1.t,\
            np.dot(Q2.T,result.GetSolution(zeta2).reshape(zeta2.shape[0],1))+Q2.t

def _setup_program_distance_point(P,ball="infinity",solver="Gurobi",distance_scaling_array = None):
    """
    Initilize the mathematial program
    Choice of balls:
        infinity: L-infinity norm
        l1: l1 norm (Manhattan Distance)
        l2: l2 norm (Euclidean Distance)
    ------
    AH_polytope: Q={t+Tx | x \in R^p, Hx <= h} \in R^{n \cdot p}
    """
    if P.distance_program is None:
        prog=MathematicalProgram()
        Q=to_AH_polytope(P)
        n=Q.n
        x=np.zeros((n,1))
        # var: x
        P.zeta=prog.NewContinuousVariables(Q.P.H.shape[1],1,"zeta")
        # var: delta
        delta=prog.NewContinuousVariables(n,1,"delta")
        # constr: Hx <= h
        prog.AddLinearConstraint(A=Q.P.H,ub=Q.P.h,lb=-np.inf*np.ones((Q.P.h.shape[0],1)),vars=P.zeta)
        # constr: Tx - delta = -t
        P.distance_constraint=prog.AddLinearEqualityConstraint( np.hstack((Q.T,-np.eye(n))),x-Q.t,np.vstack((P.zeta,delta)) )
        if ball=="infinity":
            # min cost=delta_abs
            # -delta_abs * vector1 <= delta <= delta_abs * vector1
            # vector1 = [1, 1, ..., 1]
            delta_abs=prog.NewContinuousVariables(1,1,"delta_abs")  # delta_abs \in R, slack variable
            prog.AddBoundingBoxConstraint(0,np.inf,delta_abs)
            prog.AddLinearConstraint(np.greater_equal( np.dot(np.ones((n,1)),delta_abs),delta,dtype='object' ))
            prog.AddLinearConstraint(np.greater_equal( np.dot(np.ones((n,1)),delta_abs),-delta,dtype='object' ))
            prog.AddLinearCost(delta_abs[0,0])
        elif ball=="l1":
            # min cost=dot(vector1,delta_abs)
            # -delta_abs <= delta <= delta_abs (applied on each element)
            delta_abs=prog.NewContinuousVariables(n,1,"delta_abs")
            prog.AddBoundingBoxConstraint(0,np.inf,delta_abs)
            prog.AddLinearConstraint(np.greater_equal( delta_abs,delta,dtype='object' ))
            prog.AddLinearConstraint(np.greater_equal( delta_abs,-delta,dtype='object' ))
            cost=np.dot(np.ones((1,n)),delta_abs)
            prog.AddLinearCost(cost[0,0])
        elif ball=="l2":
            # min cost=dot(delta.T,diag(distance_scaling_array),delta)
            if distance_scaling_array is None:
                distance_scaling_array = np.eye(n)
            else:
                assert(distance_scaling_array.shape==(n,))
            prog.AddQuadraticCost(np.diag(np.square(distance_scaling_array)),np.zeros(n),delta)
        else:
            print(("Not a valid choice of norm",str(ball)))
            raise NotImplementedError
        P.distance_program=prog
        return 
    else:
        return
            
        
def distance_point_polytope(P, x, ball="infinity", solver="Gurobi", distance_scaling_array=None):
    """
    Computes the distance of point x from AH-polytope Q
    Solve a underlying LP or QP
    """
    x_vector = np.atleast_2d(x) #in case x is not n*1 vector
    P = to_AH_polytope(P)
    if distance_scaling_array is None:
        distance_scaling_array = np.ones(x.shape[0])
    # solve LP or QP to find the z s.t. {Hz<=h} that minimizes L_norm(x-(t+Tz)), x is the query point
    _setup_program_distance_point(P,ball,solver, distance_scaling_array)
    prog=P.distance_program
    Q=to_AH_polytope(P)
    a=P.distance_constraint.evaluator()
    x_vector=x_vector.reshape(max(x_vector.shape),1)
#    print "sadra",x_vector.shape
    a.UpdateCoefficients(np.hstack((Q.T,-np.eye(Q.n))), x_vector - Q.t)
    if solver=="Gurobi":
        solver_opt = SolverOptions()
        solver_opt.SetOption(gurobi_solver.solver_id(), 'OutputFlag', 0)
        solver_opt.SetOption(gurobi_solver.solver_id(), 'LogToConsole', 0)
        result=gurobi_solver.Solve(prog,None,solver_opt)
    elif solver=="osqp":
        result=OSQP_solver.Solve(prog,None,None)
    else:
        result=Solve(prog)
    if result.is_success():
        # z s.t. Hz <= h
        zeta_num=result.GetSolution(P.zeta).reshape(P.zeta.shape[0],1)
        # the nearest point t+Tz in the polytope
        x_nearest=np.dot(Q.T,zeta_num)+Q.t
        delta=(x_vector - x_nearest).reshape(Q.n)
        if ball=="infinity":
            d=np.linalg.norm(delta,ord=np.inf)
        elif ball=="l1":
            d=np.linalg.norm(delta,ord=1)
        elif ball=="l2":
            d=np.linalg.norm(np.multiply(distance_scaling_array, delta),ord=2)
        else:
            raise NotImplementedError
        return d,x_nearest

def distance_point_polytope_with_multiple_azimuth(polytope, query_point, ball='l2', distance_scaling_array=None, return_modifed_query_point=False):
    """
    Find the closest state in the polytope
    :param polytope: the polytope
    :param query point: the queried point, without psic
    :param return_modifed_query_point: if true, return the modified query point
    :return: Tuple (nearest distance, nearest point in polytope, modifed query point)
    """
    duplicated_states = duplicate_state_with_multiple_azimuth_angle(query_point)
    closest_distance = np.inf
    closest_point = None
    modified_query_point = None
    for i in range(len(duplicated_states)):
        distance, projected_point = distance_point_polytope(polytope, duplicated_states[i], ball=ball, distance_scaling_array=distance_scaling_array)
        if distance < closest_distance:
            closest_distance = distance
            closest_point = projected_point
            modified_query_point = duplicated_states[i]
            # print('new state: ', duplicated_states[i])
    
    if return_modifed_query_point:
        return closest_distance, closest_point, modified_query_point
    else:
        return closest_distance, closest_point

def bounding_box(Q,solver="Gurobi"):
    Q=to_AH_polytope(Q)
    prog=MathematicalProgram()
    zeta=prog.NewContinuousVariables(Q.P.H.shape[1],1,"zeta")
    x=prog.NewContinuousVariables(Q.n,1,"x")
    prog.AddLinearConstraint(A=Q.P.H,ub=Q.P.h,lb=-np.inf*np.ones((Q.P.h.shape[0],1)),vars=zeta)
    prog.AddLinearEqualityConstraint(np.hstack((-Q.T,np.eye(Q.n))),Q.t,np.vstack((zeta,x)))
    lower_corner=np.zeros((Q.n,1))
    upper_corner=np.zeros((Q.n,1))
    c=prog.AddLinearCost(np.dot(np.ones((1,Q.n)),x)[0,0])
    if solver=="Gurobi":
        solver=gurobi_solver
    else:
        raise NotImplementedError
    a=np.zeros((Q.n,1))
    # Lower Corners
    for i in range(Q.n):
        e=c.evaluator()
        a[i,0]=1
        e.UpdateCoefficients(a.reshape(Q.n))
#        print "cost:",e.a(),
        result=solver.Solve(prog,None,None)
        assert result.is_success()
        lower_corner[i,0]=result.GetSolution(x)[i]
        a[i,0]=0
#        print result.GetSolution(x)
    # Upper Corners
    for i in range(Q.n):
        e=c.evaluator()
        a[i,0]=-1
        e.UpdateCoefficients(a.reshape(Q.n))
#        print "cost:",e.a(),
        result=solver.Solve(prog,None,None)
        assert result.is_success()
        upper_corner[i,0]=result.GetSolution(x)[i]
        a[i,0]=0
#    print(lower_corner,upper_corner)
    return hyperbox(corners=(lower_corner,upper_corner))
        
        
def directed_Hausdorff_hyperbox(b1,b2):
    """
    The directed Hausdorff hyperbox 
    min epsilon such that b1 \in b2+epsilon
    """       
    return max(0,np.max(np.hstack((b1.u-b2.u,b2.l-b1.l))))           
    
def distance_hyperbox(b1,b2):
    """
    The distance between boxes
    """
    return max(0,np.max(np.hstack((b1.l-b2.u,b2.l-b1.u))))      
    

def make_ball(n,norm):
    if norm=="l1":
        pass
    elif norm=="infinity":
        pass
    return 
#
def get_nonzero_cost_vectors(cost):
     cost[cost == 0] = 1e-3

def AH_polytope_vertices(P,N=200,epsilon=0.001,solver="Gurobi"):
    """
    Returns N*2 matrix of vertices
    """
    try:
        P.vertices_2D
        if type(P.vertices_2D) == type(None):
            raise Exception
    except:
        Q=to_AH_polytope(P)
        v=np.empty((N,2))
        prog=MathematicalProgram()
        zeta=prog.NewContinuousVariables(Q.P.H.shape[1],1,"zeta")
        prog.AddLinearConstraint(A=Q.P.H,ub=Q.P.h,lb=-np.inf*np.ones((Q.P.h.shape[0],1)),vars=zeta)
        theta=1
        c=np.array([np.cos(theta),np.sin(theta)]).reshape(2,1)
        c_T=np.dot(c.T,Q.T)
        get_nonzero_cost_vectors(c_T)
        # get_nonzero_cost_vectors(c_T)
        a=prog.AddLinearCost(np.dot(c_T,zeta)[0,0])
        if solver=="Gurobi":
            solver=gurobi_solver
        else:
            raise NotImplementedError
        for i in range(N):
            # theta=i*N/2/np.pi+0.01
            theta = (i/N)*(2*np.pi)
            c=np.array([np.cos(theta),np.sin(theta)]).reshape(2,1)
            c_T=np.dot(c.T,Q.T)
            e=a.evaluator()
            cost = c_T.reshape(Q.P.H.shape[1])
            get_nonzero_cost_vectors(cost)
            e.UpdateCoefficients(cost)
            result=solver.Solve(prog,None,None)
            assert result.is_success()
            zeta_n=result.GetSolution(zeta).reshape(zeta.shape)
            v[i,:]=(np.dot(Q.T,zeta_n)+Q.t).reshape(2)
        w=np.empty((4*N,2))
        for i in range(N):
            w[4*i,:]=v[i,:]+np.array([epsilon,epsilon])
            w[4*i+1,:]=v[i,:]+np.array([-epsilon,epsilon])
            w[4*i+2,:]=v[i,:]+np.array([-epsilon,-epsilon])
            w[4*i+3,:]=v[i,:]+np.array([epsilon,-epsilon])
        P.vertices_2D=v,w
        return v,w
    else:
        return P.vertices_2D
    
def convex_hull_of_point_and_polytope(x, Q):
    r"""
    Inputs:
        x: numpy n*1 array
        Q: AH-polytope in R^n
    Returns:
        AH-polytope representing convexhull(x,Q)
    
    .. math::
        \text{conv}(x,Q):=\{y | y= \lambda q + (1-\lambda) x, q \in Q\}.
    """
    # print(f"Q:{Q}, Q.G:{Q.G}, Q.x:{Q.x}")
    Q=to_AH_polytope(Q)
    q=Q.P.H.shape[1]
    # print(f"convex hull: Q.T:{Q.T}, Q.t:{Q.t}, x:{x}, Q.t-x:{Q.t-x}")
    new_T=np.hstack((Q.T,Q.t-x))
    new_t=x
    new_H_1=np.hstack((Q.P.H,-Q.P.h))
    new_H_2=np.zeros((2,q+1))
    new_H_2[0,q],new_H_2[1,q]=1,-1
    new_H=np.vstack((new_H_1,new_H_2))
    new_h=np.zeros((Q.P.h.shape[0]+2,1))
    new_h[Q.P.h.shape[0],0],new_h[Q.P.h.shape[0]+1,0]=1,0
    new_P=H_polytope(new_H,new_h)
    key_vertex = Q.key_vertex
    key_vertex.update({tuple(x.reshape(-1))})

    return AH_polytope(new_T,new_t,new_P,
                       mode_string=Q.mode_string,
                       mode_consistent=Q.mode_consistent,
                       applied_u=Q.applied_u,
                       psic_range=Q.psic_range,
                       key_vertex=key_vertex)

def convex_hull_of_ah_polytopes(P1,P2):
    """
    Inputs:
        P1, P2: AH_polytopes
    Output:
        returns :math:`\text{ConvexHull}(\mathbb{P}_1,\mathbb{P}_2)` as an AH-polytope
    """
    Q1 = to_AH_polytope(P1)
    Q2 = to_AH_polytope(P2)
    T=np.hstack((Q1.T,Q2.T,Q1.t-Q2.t ))
    H_1 = np.hstack((Q1.P.H,np.zeros((Q1.P.H.shape[0],Q2.P.n)), -Q1.P.h ))
    H_2 = np.hstack((np.zeros((Q2.P.H.shape[0],Q1.P.n)), Q2.P.H, Q2.P.h ))
    H_3 = np.zeros((2, Q1.P.n + Q2.P.n + 1))
    H_3[:,-1:] = np.array([1,-1]).reshape(2,1)
    H = np.vstack((H_1,H_2,H_3))
    h = np.vstack((Q1.P.h*0,Q2.P.h,1,0))
    new_P=H_polytope(H=H, h=h)

    # key points
    key_vertex = set()
    key_vertex.update(Q1.key_vertex)
    key_vertex.update(Q2.key_vertex)

    # mode consistency
    assert Q1.mode_consistent == Q2.mode_consistent

    return AH_polytope(T=T,t=Q2.t,P=new_P,
                       mode_string=Q1.mode_string,
                       mode_consistent=Q1.mode_consistent,
                       applied_u=Q1.applied_u,
                       psic_range=Q1.psic_range,
                       key_vertex=key_vertex)


def minkowski_sum(P1,P2):
    r"""
    Inputs: 
        P1, P2: AH_polytopes
    Returns:
        returns the Mkinkowski sum :math:`P_1 \oplus P_2` as an AH-polytope.
        
    **Background**: The Minkowski sum of two sets is defined as:
        
    .. math::
        A \oplus B = \{ a + b \big | a \in A, b \in B\}.
    
    """
    Q1,Q2=to_AH_polytope(P1),to_AH_polytope(P2)
    T=np.hstack((Q1.T,Q2.T))
    t=Q1.t+Q2.t
    H=spa.block_diag(*[Q1.P.H,Q2.P.H])
    h=np.vstack((Q1.P.h,Q2.P.h))
    new_P=H_polytope(H,h)
    return AH_polytope(T,t,new_P)  
    
    
def intersection(P1,P2):
    """
    Inputs: 
        P1, P2: AH_polytopes
    Returns:
        returns :math:`P_1 \wedge P_2` as an AH-polytope
    """
    Q1,Q2=to_AH_polytope(P1),to_AH_polytope(P2)
    T=np.hstack((Q1.T,Q2.T*0))
    t=Q1.t
    H_1=spa.block_diag(*[Q1.P.H,Q2.P.H])
    H_2=np.hstack((Q1.T,-Q2.T))
    H=np.vstack((H_1,H_2,-H_2))
    h=np.vstack((Q1.P.h,Q2.P.h,Q2.t-Q1.t,Q1.t-Q2.t))
    new_P=H_polytope(H,h)
    return AH_polytope(T,t,new_P)

    
"""
Pydrake Mathematical Program Helper: Matrix based Constraints
"""
def AddMatrixInequalityConstraint_classical(mathematical_program,A,X,B):
    raise NotImplementedError    
    
def Lambda_h_Inequality(mathematical_program,Lambda,beta,H,h_1,h_2):
    """
    Adds Lambda H-1 \le h_2 + H beta to the Mathematical Program
    """
    M1=np.kron(np.eye(h_2.shape[0]),h_1.T)
    M=np.hstack((M1,-H))
    v1=Lambda.reshape((Lambda.shape[0]*Lambda.shape[1],1))
    v=np.vstack((v1,beta))
    mathematical_program.AddLinearConstraint(A=M,ub=h_2,lb=-np.inf*np.ones(h_2.shape),vars=v)
    
def Lambda_h_Inequality_D(mathematical_program,Lambda,beta,H,h_1,h_2,D):
    """
    Adds Lambda H-1 \le h_2 D + H beta to the Mathematical Program
    """
    M1=np.kron(np.eye(h_2.shape[0]),h_1.T)
    M=np.hstack((M1,-H,-h_2))
    v1=Lambda.reshape((Lambda.shape[0]*Lambda.shape[1],1))
    v=np.vstack((v1,beta,D))
    mathematical_program.AddLinearConstraint(A=M,ub=np.zeros(h_2.shape),lb=-np.inf*np.ones(h_2.shape),vars=v)
    
def positive_matrix(mathematical_program,Lambda):
    """
    All elements are non-negative 
    """
#    q=Lambda.shape[0]
    mathematical_program.AddBoundingBoxConstraint(0,np.inf,Lambda)
#    [mathematical_program.AddLinearConstraint(A=np.eye(q),vars=Lambda[:,i],
#                                              ub=np.inf*np.ones((q,1)),lb=np.zeros((q,1)))
#                                                for i in range(Lambda.shape[1])]
#    q=Lambda.shape[0]*Lambda.shape[1]
#    mathematical_program.AddLinearConstraint(A=np.eye(q),vars=Lambda.reshape(q),ub=np.inf*np.ones((q)),lb=np.zeros((q)))
        
def Lambda_H_Gamma(mathematical_program,Lambda,H_1,H_2,Gamma):
    """
    Lambda H_1 = H_2 Gamma
    """
#    v_1=Lambda.reshape((Lambda.shape[0]*Lambda.shape[1],1))
#    for j in range(Gamma.shape[1]):
#        M1=np.kron(np.eye(H_2.shape[0]),H_1[:,j].reshape(1,H_1.shape[0]))
#        M2=-H_2
#        M=np.hstack((M1,M2))
#        v=np.vstack((v_1,Gamma[:,j].reshape(Gamma.shape[0],1)))
#        mathematical_program.AddLinearEqualityConstraint(M,np.zeros((M.shape[0],1)),v)
    for i in range(Lambda.shape[0]):
        for j in range(Gamma.shape[1]):
            M=np.hstack((H_1[:,j],-H_2[i,:]))
            v=np.hstack((Lambda[i,:],Gamma[:,j]))
            mathematical_program.AddLinearEqualityConstraint(M.reshape(1,M.shape[0]),np.zeros(1),v)
