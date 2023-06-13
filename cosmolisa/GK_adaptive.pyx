import cython
cimport cython
from libc.stdlib cimport malloc, free
from libc.math cimport pow, fabs, sqrt, exp

import numpy as np
cimport numpy as np

cdef double epsabs = 1e-7    # absolute integration error
cdef double epsrel = 1e-7

ctypedef struct interval_t:
    double left
    double right
    double I
    double err

# nodes and weights for Gauss-Kronrod
cdef double[15][3] gausskronrod = [
    # node               weight Gauss       weight Kronrod
    [ +0.949107912342759, 0.129484966168870, 0.063092092629979 ],
    [ -0.949107912342759, 0.129484966168870, 0.063092092629979 ],
    [ +0.741531185599394, 0.279705391489277, 0.140653259715525 ],
    [ -0.741531185599394, 0.279705391489277, 0.140653259715525 ],
    [ +0.405845151377397, 0.381830050505119, 0.190350578064785 ],
    [ -0.405845151377397, 0.381830050505119, 0.190350578064785 ],
    [  0.000000000000000, 0.417959183673469, 0.209482141084728 ],

    [ +0.991455371120813, 0.000000000000000, 0.022935322010529 ],
    [ -0.991455371120813, 0.000000000000000, 0.022935322010529 ],
    [ +0.864864423359769, 0.000000000000000, 0.104790010322250 ],
    [ -0.864864423359769, 0.000000000000000, 0.104790010322250 ],
    [ +0.586087235467691, 0.000000000000000, 0.169004726639267 ],
    [ -0.586087235467691, 0.000000000000000, 0.169004726639267 ],
    [ +0.207784955007898, 0.000000000000000, 0.204432940075298 ],
    [ -0.207784955007898, 0.000000000000000, 0.204432940075298 ]
]

cdef class GKIntegrator:

#    cdef public int dimension
#    cdef public int current_dimension
#    cdef public double *y
#    cdef public object integrand
#    cdef public object args
#    cdef public double I
#    cdef public double err
#    cdef public np.ndarray a
#    cdef public np.ndarray b

    def __cinit__(self,
                  int limit,
                  int minintervals,
                  double tol):
        
        self.minintervals = minintervals
        self.limit       = limit
        self.tolerance   = tol
        
        if(self.limit < self.minintervals):
            self.limit = self.minintervals

    def integrate(self, object f, object args, np.ndarray a, np.ndarray b):
        return self._integrate(f,args,a,b)

    cdef (double, double) _integrate(self, object f, object args, np.ndarray a, np.ndarray b):
        
        self.dimension = a.shape[0]
        self.a         = a
        self.b         = b
        self.integrand = f
        self.args      = args
        self.y         = np.zeros(self.dimension, dtype=np.double)
        
        I, err = self._gausskronrod_integrate_adaptive(self.dimension)
        
        return I, err

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef (double, double) _gausskronrod_integrate(self,
                                                  double a,
                                                  double b,
                                                  int dimension):
        """
        /** @brief Compute integral using Gauss-Kronrod quadrature
         *
         * This function computes \f$\int_a^b \mathrm{d}x f(x)\f$ using Gauss-Kronrod
         * quadrature formula. The integral is transformed according to
         * \f$z  = 2 \frac{x-a}{b-a}-1\f$
         * \f$x  = \frac{b-a}{2} (z+1) + a\f$
         * \f$dz = 2 \frac{dx}{b-a}\f$
         * \f$dx = \frac{b-a}{2} dz\f$
         * \f$\int_a^b \mathrm{d}x f(x) = \frac{b-a}{2} \int_{-1}^1 \mathrm{d}z f((z+1)*(b-a)/2+a)\f$
         *
         * @param [in]  f callback to integrand
         * @param [in]  a lower limit of integration
         * @param [in]  b upper limit of integration
         * @param [in]  args pointer to arbitrary data that is passed to f
         * @param [out] I calculated value of integral
         * @param [out] err estimated error
         */
        """
        cdef unsigned int i
        cdef double dx = (b-a)/2
        cdef double integral_G7  = 0.0
        cdef double integral_K15 = 0.0

        cdef double xi, wiG, wiK
        cdef double fzi
        cdef double err = 2828.42712474619*fabs(dx) # 2828.42712474619 = pow(200,1.5)
        
        for i in range(15):

            xi  = gausskronrod[i][0];
            wiG = gausskronrod[i][1];
            wiK = gausskronrod[i][2];

            self.y[dimension-1] = (xi+1)*dx+a
            
            if dimension == 1:
                fzi = self.integrand(self.y, self.args)
            else:
                fzi, err = self._gausskronrod_integrate_adaptive(dimension-1)
            if fzi != 0.0:
                integral_G7  += wiG*fzi;
                integral_K15 += wiK*fzi;
        
        cdef double tmp = fabs(integral_G7-integral_K15)
        err *= sqrt(tmp*tmp*tmp)
        return dx*integral_K15, err



    cdef (double, double) _gausskronrod_integrate_adaptive(self, int dimension):
        """
        /** Compute integral using adaptive Gauss-Kronrod quadrature
         *
         * Do adaptive integration using Gauss-Kronrod.
         *
         * @param [in]  f callback to integrand
         * @param [in]  a lower limit of integration
         * @param [in]  b upper limit of integration
         * @param [in]  minintervals split integral in at least minintervals subintervals and perform Gauss-Kronrod quadrature
         * @param [in]  limit maximum number of subintervals
         * @param [in]  tol relative error tolerance
         * @param [in]  args pointer to arbitrary data that is passed to f
         * @param [out] I computed value of integral
         * @param [out] err estimated error
         *
         * @retval -1 if no convergence
         * @retval subintervals number of intervals used
        """

        cdef interval_t *intervals = <interval_t *>malloc(self.limit*sizeof(interval_t))
        cdef unsigned int i,len
        cdef interval_t *interval
        
        # compute the integral in each subinterval
        for len in range(self.minintervals):
        
            interval = &intervals[len]
            interval.left  = self.a[dimension-1] + len   *(self.b[dimension-1]-self.a[dimension-1])/self.minintervals
            interval.right = self.a[dimension-1] +(len+1)*(self.b[dimension-1]-self.a[dimension-1])/self.minintervals
            interval.I, interval.err = self._gausskronrod_integrate(interval.left, interval.right, dimension)
            
        cdef double err2 = 0, Itotal = 0, err_max = 0
        cdef unsigned int maximum = 0
        cdef double I_i, err_i
        cdef double left, right, mid
        cdef double err_left, err_right, I_left, I_right
        
        while True:
     
            err2 = 0
            Itotal = 0
            err_max = 0
            maximum = 0

            # search for largest error and its index, calculate integral and
            # error²
            #
            for i in range(len+1):
            
                I_i   = intervals[i].I
                err_i = intervals[i].err

                Itotal += I_i

                if(err_i > err_max):
                
                    err_max = err_i
                    maximum = i

                err2 += err_i*err_i
#                print(i,len,intervals[i],dimension, Itotal)
            # we reached the required tolerance, return result and error
            if(fabs(sqrt(err2)/Itotal) < self.tolerance):
                free(intervals)
                return Itotal, sqrt(err2)

            #/* no convergence */
            if(len >= self.limit):
                print("warning! integration did not converge properly")
                free(intervals)
                return Itotal, sqrt(err2)
            
            """
            /* accuracy is still not good enough, so we split up the partial
             * integral with the largest error:
             * [left,right] => [left,mid], [mid,right]
             */
            """
            left  = intervals[maximum].left
            right = intervals[maximum].right
            mid   = left+(right-left)/2

            """
            /* calculate integrals and errors, replace one item in the list and
             * append the other item to the end of the list
             */
            """
            I_left, err_left   = self._gausskronrod_integrate(left, mid, dimension)
            I_right, err_right = self._gausskronrod_integrate(mid,  right, dimension)
            
            intervals[maximum].left  = left
            intervals[maximum].right = mid
            intervals[maximum].I     = I_left
            intervals[maximum].err   = err_left
           
            intervals[len+1].left  = mid
            intervals[len+1].right = right
            intervals[len+1].I     = I_right
            intervals[len+1].err   = err_right

            # increase len of array intervals */
            len += 1
