import numpy as np

import readdata

events = readdata.read_event("EMRI", "./data/EMRI_SAMPLE_MODEL101_TTOT10yr_SIG2_GAUSS", None)

densities = np.array([e.n_hosts/e.VC for e in events])
densities.sort()
print([d for d in densities])
print(np.average(densities, weights = [e.VC for e in events]))
exit()
import matplotlib.pyplot as plt
zs = []
for e in events:
    zs.append(np.average([g.redshift for g in e.potential_galaxy_hosts]))

x = [e.VC for e in events]
y = densities
p = np.polyfit(np.log(x),np.log(y),1)
print(p)
plt.scatter(np.log(x),np.log(y))
xp = np.linspace(np.log(x).min(),np.log(x).max(),1000)
plt.plot(xp,np.poly1d(p)(xp))
plt.show()
