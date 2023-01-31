import numpy as np
import scipy.sparse as sps
from scipy.linalg import solve as lin_solve

def FB(x, q, M, l, u):
    n = len(x)
    Zl = np.bitwise_and(l>-np.inf, u==np.inf)
    Zu = np.bitwise_and(l==-np.inf, u<np.inf)
    Zlu = np.bitwise_and(l>-np.inf, u<np.inf)
    Zf = np.bitwise_and(l==-np.inf, u==np.inf)

    a = x
    b = np.matmul(M,x)+q

    a[Zl] = x[Zl]-l[Zl]
    a[Zu] = u[Zu]-x[Zu]
    b[Zu] = -b[Zu]

    if np.any(Zlu):
        nt = Zlu.sum()
        at = u[Zlu]-x[Zlu]
        bt = -b[Zlu]
        st = np.sqrt(at**2+bt**2)
        a[Zlu] = x[Zlu]-l[Zlu]
        b[Zlu] = st-at-bt

    s = np.sqrt(a**2+b**2)
    phi = s-a-b
    phi[Zu] = -phi[Zu]
    phi[Zf] = -phi[Zf]

    psi = 0.5*np.dot(phi,phi)

    if np.any(Zlu):
        M[Zlu, :] = -sps.csr_matrix((at/st-np.ones(nt), (np.arange(nt), np.where(Zlu==True)[0])), shape=(nt,n), dtype=np.float32) - np.matmul(np.diag(bt/st-np.ones(nt)), M[Zlu, :])
    da = a/s-np.ones(n)
    db = b/s-np.ones(n)
    da[Zf] = 0
    db[Zf] = -1
    J = np.diag(da.reshape(-1))+np.matmul(np.diag(db.reshape(-1)),M)

    return psi, phi, J


def LCP(M, q):

    tol = 1e-12
    mu = 1e-3
    mu_step = 5
    mu_min = 1e-5
    max_iter = 10
    b_tol = 1e-6

    n = M.shape[0]
    l = np.zeros(n)
    u = np.inf * np.ones(n)
    x0 = np.minimum(np.maximum(np.ones(n),l),u)

    lu = np.c_[l, u]
    x = x0.copy()

    psi, phi, J = FB(x,q,M,l,u)
    new_x = True

    for iter in range(max_iter):
        if new_x:
            mlu = np.min(np.c_[np.abs(x-l), np.abs(u-x)], axis=1)
            ilu = np.argmin(np.c_[np.abs(x-l), np.abs(u-x)], axis=1)
            bad = np.maximum(np.abs(phi), mlu) < b_tol
            psi = psi - 0.5*np.dot(phi[bad], phi[bad])
            J = J[~bad,:][:,~bad]
            phi = phi[~bad]
            new_x = False
            nx = x.copy()
            nx[bad] = lu.flatten('F')[np.where(bad==True)[0]+ilu[bad]*n]
        H = np.matmul(J.T, J) + mu*np.eye((~bad).sum())
        Jphi = np.matmul(J.T, phi)

        # d = -np.linalg.solve(H,Jphi)
        # Singular matrix
        try:
            d = -lin_solve(H,Jphi)
        except:
            d = -lin_solve(H+(1e-6*np.eye(H.shape[0])),Jphi)

        nx[~bad] = x[~bad] + d
        npsi, nphi, nJ = FB(nx,q,M,l,u)

        r = (psi-npsi)/(-(np.matmul(Jphi.T,d)+0.5*np.matmul(d.T,np.matmul(H,d))))

        if r<0.3:
            mu = max(mu*mu_step,mu_min)
        if r>0:
            x = nx.copy()
            psi = npsi.copy()
            phi = nphi.copy()
            J = nJ
            new_x = True
            if r>0.8:
                mu = mu/mu_step * (mu>mu_min)

        if psi < tol:
            break

    x = np.minimum(np.maximum(x,l),u)
    return x


if __name__ == '__main__':
    import time
    from scipy import io as scio
    nx = scio.loadmat('/home/yongpeng/下载/nx.mat')['nx'].flatten()
    q = scio.loadmat('/home/yongpeng/下载/q.mat')['q'].flatten()
    M = scio.loadmat('/home/yongpeng/下载/M.mat')['M']
    l = np.zeros(4,)
    u = np.inf * np.ones(4)
    psi, phi, J = FB(nx, q, M, l, u)
    print('psi: ', psi)
    print('phi: ', phi)
    print('J: ', J)

    start_time = time.time()
    import pdb; pdb.set_trace()
    x = LCP(M, q)
    print('M: ', M)
    print('q: ', q)
    print('x: ', x)
    print('time cost: ', time.time()-start_time)
