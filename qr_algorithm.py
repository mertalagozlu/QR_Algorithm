import numpy as np
from scipy.linalg import hilbert
import matplotlib.pyplot as pl
#import scipy.linalg

def sign(z):    
    return np.exp(1j*np.angle(z))

def implicit_qr(A): # (a)    
    W = A.copy()    
    m, n = A.shape    
    A = A.astype(np.complex128)
    W = np.zeros((m, n), dtype = np.complex128)    
    for k in range(n):        
        v = A[k:, k].copy()        
        v[0] += sign(v[0])*np.linalg.norm(v)        
        v /= np.linalg.norm(v)        
        W[k:, k] = v        
        A[k:,k:] -= 2.0 * np.outer(v, np.dot(v.conjugate(), A[k:,k:]) )
    return (W, A)

def implicit_apply_Q(W, x):
    m, n = W.shape
    x = x.astype(np.complex128)
    for k in range(n - 1, -1, -1):
        v = W[:, k]
        x -= 2.0 * v * np.dot(v.conjugate(), x)        
    return x


def form_q(W): # (a)    
    Q = W.copy()
    m, n = W.shape
    Q = np.eye(m, dtype = np.complex128)
    for j in range(m):
        Q[:, j] = implicit_apply_Q(W, Q[:, j])        
    return Q

def tridiag(A):
    m = len(A)
    for k in range(0, m-2):        
        x = A[k+1:,k]
        define_e1 = np.eye(m-k-1)
        v = np.sign(x[0]) * np.linalg.norm(x) * define_e1[:,0] + x
        v = v/np.linalg.norm(v)        
        A[k+1:,k:] -= 2* np.outer(v, np.dot(v.conjugate(), A[k+1:,k:]))
        A[0:,k+1:] -= 2* np.outer(np.dot(A[0:,k+1:],v.conjugate()), v.T)
    return A

def QR_alg(T):
    m = len(T)
    t = []
    while abs(T[m-1,m-2]) > 1e-12:
        # Implement Pure QR Algorithm
        # Take Q and R and multiply in reverse until it converges up to 10^-12
        W, R = implicit_qr(T)  
        Q = form_q(W)    
        t.append(abs(T[m-1,m-2])) # Append all |tm,tm-1| --> list := t
        T = R @ Q # At the last step Tnew is upper triangular or diagonal(Hermitian)
    return (T, t)

def wilkinson_shift(T):
    μ = 0
    # Alg. 29.8
    # Pick lower rightmost 2x2 submatrix of A
    m = len(T)
    B = T[m-2:,m-2:] # rightmost 2x2 
    delta = (B[0,0] - B[1,1])/2 
    if delta == 0: # Arbitrary set sign if delta ---> 0     
        sign = 1     
    else:
        sign = np.sign(delta)     
    # Apply wilkinson shift coef
    µ = B[1,1] - sign* B[0,1]*B[1,0]/ ( abs(delta) + np.sqrt(delta**2 + B[0,1]*B[1,0]) )
    return μ

def QR_alg_shifted(T):
    # Alg. 28.2: Practical QR Algorithm
    m = len(T)
    t = []
    while abs(T[m-1,m-2]) > 1e-12:
        # Pick Wilkinson Shift for the initial tridiagonal matrix
       µ = wilkinson_shift(T)  
        #Instead of T, factor out Q(k)R(k) <---- T(k) - µ(k)*Identity at each step until off-diagonal element ---> 0
       W, R = implicit_qr(T - µ*np.identity(m))    
       Q = form_q(W)   
       #Recombine factors in reverse order with wilkinson shift
       t.append(abs(T[m-1,m-2])) # Append all |tm,tm-1| --> list := t
       T = R @ Q + µ*np.identity(m) # At the last step T is upper triangular or diagonal(Hermitian)
    return (T, t)

def QR_alg_driver(A, shift):
    Λ = []
    all_t = []
    T = tridiag(A)
    m = len(T)
    for k in range(m-1,0,-1):
        if shift == False:    
            T, t = QR_alg(T[0:k+1,0:k+1])
            all_t = [*all_t, *t]
            Λ.append(T[k,k])
        elif shift == True:
            T, t = QR_alg_shifted(T[0:k+1,0:k+1])
            all_t = [*all_t, *t]
            Λ.append(T[k,k])     
    Λ.insert(0, T[0,0])
    return (Λ, all_t)

if __name__ == "__main__":
    matrices = {
        "hilbert": hilbert(4),
        "diag(1,2,3,4)+ones": np.diag([1, 2, 3, 4]) + np.ones((4, 4)),
        "diag(5,6,7,8)+ones": np.diag([5, 6, 7, 8]) + np.ones((4, 4)),}
    fig, ax = pl.subplots(len(matrices.keys()), 2, figsize=(10, 10))
    for i, (mat, A) in enumerate(matrices.items()):
        for j, shift in enumerate([True, False]):
            Λ, conv = QR_alg_driver(A, shift)
            ax[i, j].semilogy(range(len(conv)), conv, ".-")
            ax[i, j].set_title(f"A = {mat}, shift = {shift}")
    pl.show()
    
