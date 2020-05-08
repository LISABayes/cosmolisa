cdef extern from "lal/LALCosmologyCalculator.h" nogil:
    ctypedef struct LALCosmologicalParameters:
        double h;
        double om;
        double ol;
        double ok;
        double w0;
        double w1;
        double w2;

    cdef double XLALLuminosityDistance(
            LALCosmologicalParameters *omega,
            double z)

    cdef double XLALAngularDistance(
            LALCosmologicalParameters *omega,
            double z)

    cdef double XLALComovingLOSDistance(
            LALCosmologicalParameters *omega,
            double z)

    cdef double XLALComovingTransverseDistance(
            LALCosmologicalParameters *omega,
            double z)

    cdef double XLALHubbleDistance(
            LALCosmologicalParameters *omega
            )

    cdef double XLALHubbleParameter(double z,
            void *omega
            )

    cdef double XLALIntegrateHubbleParameter(LALCosmologicalParameters *omega, double z)

    cdef double XLALUniformComovingVolumeDistribution(
            LALCosmologicalParameters *omega,
            double z,
            double zmax)

    cdef double XLALUniformComovingVolumeDensity(
            double z,
            void *omega)

    cdef double XLALIntegrateComovingVolumeDensity(LALCosmologicalParameters *omega, double z)

    cdef double XLALIntegrateComovingVolume(LALCosmologicalParameters *omega, double z)

    cdef double XLALComovingVolumeElement(double z, void *omega)

    cdef double XLALComovingVolume(LALCosmologicalParameters *omega, double z)

    cdef LALCosmologicalParameters *XLALCreateCosmologicalParameters(double h, double om, double ol, double w0, double w1, double w2)

    cdef void XLALDestroyCosmologicalParameters(LALCosmologicalParameters *omega)

    cdef double XLALGetHubbleConstant(LALCosmologicalParameters *omega)

    cdef double XLALGetOmegaMatter(LALCosmologicalParameters *omega)

    cdef double XLALGetOmegaLambda(LALCosmologicalParameters *omega)

    cdef double XLALGetOmegaK(LALCosmologicalParameters *omega)

    cdef double XLALGetW0(LALCosmologicalParameters *omega)

    cdef double XLALGetW1(LALCosmologicalParameters *omega)

    cdef double XLALGetW2(LALCosmologicalParameters *omega)

cdef class CosmologicalParameters:
    cdef LALCosmologicalParameters* _LALCosmologicalParameters
    cdef public double h
    cdef public double om
    cdef public double ol
    cdef public double w0
    cdef public double w1
    cpdef void SetH(self, double h)
    cpdef void SetOM(self, double om)
    cpdef void SetOL(self, double ol)
    cpdef double HubbleParameter(self,double z)
    cpdef double LuminosityDistance(self, double z)
    cpdef double HubbleDistance(self)
    cpdef double IntegrateComovingVolumeDensity(self, double zmax)
    cpdef double IntegrateComovingVolume(self, double zmax)
    cpdef double UniformComovingVolumeDensity(self, double z)
    cpdef double UniformComovingVolumeDistribution(self, double z, double zmax)
    cpdef double ComovingVolumeElement(self,double z)
    cpdef double ComovingVolume(self,double z)
    cpdef void DestroyCosmologicalParameters(self)
