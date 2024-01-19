# Chiral Actvie Brownian Motion

- Equation of motion
  - chiral Active Brownian particle (cABP) : Adding constant angular velocity $\omega$ to ABP.
  
  $\frac{dr_j}{dt} = \frac{1}{\zeta}F_j + v e_j$, $(e_j(\phi_j)) = (cos\theta_j, sin\theta_j)$
  
  
  $\frac{d\phi_j}{dt} = \sqrt{\frac{2}{\tau_P}}\eta_j(t) + \omega$, $\left< \eta_j(t)\eta_k(t') \right> = \delta_{jk}\delta (t-t')$
  
  You can choose potentials between WCA and LJ.
  
  And, I use __<ins>CUDA (you must need nvidia GPU) and MPI.</ins>__
  
  You should always check compiler's addresses. 

- Measurements
  
  I added the measurements that are radial distribution function, structure factor, mean square displacement, and Intermediate scattering function.
  
