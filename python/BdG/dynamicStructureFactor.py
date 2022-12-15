import numpy as np
import tqdm
import matplotlib.pylab as plt


def matrix_element_sq(q,psi0,u,v,r):
    f=np.sum( 4*np.pi*(r[1]-r[0])*(u + v )*r*psi0*np.sin(q*r)/q )
    return np.sum(f)

def plot(S,q,omega,mu=None,show_recoil=True):
    """
    S : array with dynamic structure values
    q: transferred momenta axis , 1D array
    omega : energy axis, 1D array
    """
    q_labels=["{:.2f}".format(_q) for _q in q]
    omega_labels=["{:.2f}".format(_omega) for _omega in omega]

    S_data=pd.DataFrame(data=S.transpose(),index=omega_labels,columns=q_labels)
    ax=sns.heatmap(S_data,xticklabels=len(q)//5,yticklabels=len(omega)//5)
    #ax=sns.heatmap(S_data)

    ax.axes.invert_yaxis()
    if show_recoil:
        sns.lineplot(x=q/np.max(q)*len(q), y=0.5*q**2/np.max(omega)*len(omega),color="r" ,label="recoil energy")
    #sns.lineplot(x=q/np.max(q)*len(q), y=0.5*q**2/np.max(omega)*len(omega) )
    Er=q**2/2
    if mu is not None:
            sns.lineplot(x=q/np.max(q)*len(q), y=(Er * np.sqrt(1+2*mu/Er) )/np.max(omega)*len(omega),color="g" ,
                         label=r"$E_r\sqrt{1+2\mu/E_r}$")

    plt.xlabel(r"$q$")
    plt.ylabel(r"$\omega$")




class dynamicStructureFactor:
    def __init__(self,e,vs,r,psi0):
        self.e=e
        self.vs=vs
        self.r=r

    def __call__( self,q ):
        """
        Inputs:
        q: wavevectors
        Outpus:
        S : 2D matrix of shape (len(q),len(e))
        """
        e=self.e
        vs=self.vs
        x=self.r


        S=np.zeros(len(q),(len(e)))
        ks=np.arange(0,len(e))
        for k in tqdm.tqdm(ks):
            u=vs[0:len(x),k]
            v=vs[len(x):,k]
            S[:,k]=[np.abs(matrix_element_sq(_q,psi0,u,v,x))**2 for _q in q]
        return S