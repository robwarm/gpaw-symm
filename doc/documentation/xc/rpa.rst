.. _rpa:

=======================
RPA correlation energy
=======================

The correlation energy within the Random Phase Approximation (RPA) can be written

.. math::

  E_c^{RPA} = \int_0^{\infty}\frac{d\omega}{2\pi}\text{Tr}\Big[\text{ln}\{1-\chi^0(i\omega)v\}+\chi^0(i\omega)v\Big],
 
where `\chi^0(i\omega)` is the non-interacting (Kohn-Sham) response function evaluated at complex frequencies, `\text{Tr}` is the Trace and `\it{v}` is the Coulomb interaction. The response function and Coulomb interaction are evaluted in a plane wave basis as described in :ref:`df_tutorial` and :ref:`df_theory` and for periodic systems the Trace therefore involves a summation over `\mathbf{q}`-points, which are determined from the Brillouin zone sampling used when calculating `\chi^0(i\omega)`. 

The RPA correlation energy is obtained by::
    
    from gpaw.xc.rpa_correlation_energy import RPACorrelation
    rpa = RPACorrelation(calc, txt='rpa_correlation.txt')   
    E_rpa = rpa.get_rpa_correlation_energy()

where calc is a calculator object containing converged wavefunctions from a ground state calculation and txt denotes the output file. The function get_rpa_correlation_energy() takes a number of keyword arguments specified below.


Parameters
==========

=================== ================== =================== ==================================================================
keyword             type               default value       description
=================== ================== =================== ==================================================================
``ecut``            ``float``          100.		   Sets the number of plane waves
							   and bands (if nbands is None) included in 
 							   the response function
``nbands``	    ``int``	       None		   Sets the number of bands included in the 
							   response function. If None, nbands is set 
							   equal to the number of plave waves, which is determined by 
 							   ecut
``gauss_legendre``  ``int``            16                  Number of Gauss-legendre points used in the 
							   integration. Presently one can choose between
							   8, 16, 24 or 32
``frequency_cut``   ``float``	       800. (eV)           The frequency cut is the largest frequency 
							   included in the Gauss-Legendre integration.
``frequency_scale`` ``float``	       2.0 (eV)		   The frequency scale sets the density of frequency 
							   points near :math:`\omega = 0`. 
``w``               ``numpy.ndarray``  None                Specifies frequency points used to integrate the 
							   correlation integrand. A simple trapezoid integration is 
							   performed on the specified points. 
							   Ex: numpy.linspace(0,20,201). If None, the Gauss-legendre 
							   method is used.
``direction``	    ``int``	       [[0, 1/3.],	   List of directions and corresponding weights 
				       [1, 1/3.],	   for the :math:`\mathbf{q} = 0` point. 
				       [2, 1/3.]]	   This point essentially needs to be evaluated
				                 	   as an average over values close to zero in the three 
				                 	   directions. If the system has symmetry
							   one may save time by specifying the directions needed.
							   Ex: for a diatomic molecule with the molecular axis in 
							   the z direction, one may use [[0, 2/3.], [2, 1/3.]], since
							   x and y (0 and 1 directions are equivalent).
``skip_gamma``      ``bool``	       False		   For metals the :math:`\mathbf{q} = 0` point can give rise
                                                           to divergent contributions and it may be faster to converge 
							   the k-point sampling if this point is excluded. This should be 
                                                           then also be done for the HF energy which also has this keyword.
                                                           (See Ref. \ [#Harl2]_ for details)
``extrapolate``     ``bool``	       False		   If w is not None, the specified frequency points are 
							   extrapolated to infinity by assuming a squared Lorentzian 
							   decay of the integrand.
``kcommsize``       ``int``            None                The parsize for parallelization
                                                           over kpoints.
=================== ================== =================== ==================================================================

In addition to the usual kpoint and grid sampling, the RPA correlation energy needs to be converged with respect to the plane wave cutoff (set by ecut) and the frequency integration. As it turns out, the integrand is usually  rather smooth and one can perform the integration with 8-16 (special!) Gauss-Legendre frequency points, but see the tutorial :ref:`rpa_tut` for an example of converging the frequency integration.
	
Convergence
===========

A major complication with the RPA correlation energy is that it converges very slowly with the number of unoccupied bands included in the evaluation of `\chi^0(i\omega)`. However, as described in Ref. \ [#Harl1]_ the high energy part of the response function resembles the Lindhard function, which for high energies gives a correlation energy converging as

.. math::

  E_c^{Lindhard}(E^{\chi}_{cut}) = E_c^{\infty}+\frac{A}{(E^{\chi}_{cut})^{3/2}},

where `E^{\chi}_{cut}` is cutoff energy used in the evaluation of `\chi^0`. With an external potential, the number of unoccupied bands is an additional convergence parameter, but for reproducing the scaling of the Lindhard function, it is natural to set the total number of bands equal to the number of plane waves used. Thus, to obtain a converged RPA correlation energy one should proceed in three steps.

* Perform a ground state calculation with a lot of converged unoccupied bands.
  
* Define a list of cutoff energies - typically something like [200, 225, 250, 275, 300] (eV). For each cutoff energy perform an RPA correlation energy calculation with the number bands `n` set equal to the number of plane waves defined by that cutoff energy. 

* Fit the list of obtained correlation energies to `E_c^{RPA}(E) = E_c^{\infty}+A/E^{3/2}` to obtain `E_c^{\infty}=E_c^{RPA}`.

If one is not interested in the total correlation energy, but only energy differences between similar systems, it is sometimes possible to avoid the extrapolation procedure.

.. [#Harl1] J. Harl and G. Kresse,
            *Phys. Rev. B* **77**, 045136 (2008)

.. [#Harl2] J. Harl and L. Schimka and G. Kresse,
            *Phys. Rev. B* **81**, 115126 (2010)
