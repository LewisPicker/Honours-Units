
# import mesa_reader
import mesa_reader as mr
import matplotlib.pyplot as plt


# load and plot data
h = mr.MesaData('/home/lewis/Documents/Honours_Research/data/LOGS/LOGS_1Zsun_updatedovershoot/8Msun_1Zsun_sc01/history.data')
initial_star_mass = str(round(h.star_mass[0],2))
title = 'HR diagram of an '+initial_star_mass +' solar mass star'

rsol = 6.96e10 #cm
msol = 1.989e33 #grams
lsol = 3.839e33 #ergs
G = 6.67259e-8 #grav constant CGS
z  = 0.03 #solar metalicity

# plt.plot(h.model_number,h.radius_cm/rsol)
# plt.show()
# # plt.plot(h.model_number,h.log_LH, c = 'r')
# # plt.plot(h.model_number,h.log_LZ, c = 'b')
# # plt.plot(h.model_number,h.log_LHe, c = 'g')
# # plt.plot(h.model_number,h.log_Lnuc, c = 'c')
# # plt.ylim(-1,7)
# # plt.show()
# # plt.plot(h.model_number,h.center_he4)
# # plt.show()
# exit()

print('core he burn',(h.star_age[2888]-h.star_age[2060])/1e6)
print('min radius = ', min(h.radius_cm/rsol) )


b = 936
c = 1190
d=2070
2060
2888
print('B = ', h.star_age[b])
print('A = ', h.star_age[c])
print("final age = ", h.star_age[-1]/1e6)
# exit()
plt.scatter(4.3,3.9,marker = "*", c = 'b',zorder=1, s = 1e2, label = 'End of MS')
plt.scatter(3.555,4.096,marker = "*", c = 'darkorange',zorder=1, s = 3e2, label = 'Red Giant')
plt.scatter(3.52,4.47,marker = "*", c = 'r',zorder=1, s = 4.5e2, label = 'AGB star')
plt.plot(h.log_Teff[800:c], h.log_L[800:c], color ='mediumorchid', label = 'MS track')
plt.plot(h.log_Teff[c:d], h.log_L[c:d], linestyle='--', color = 'mediumorchid', label = 'HG track')
plt.plot(h.log_Teff[d:], h.log_L[d:], linestyle='dotted', color = 'mediumorchid', label = 'AGB track')

plt.xlabel('$lg(T_{eff})$')
plt.ylabel('$lg(L/L_{\odot})$')
plt.legend(loc = 0)
# plt.title(title)
plt.gca().invert_xaxis()
plt.savefig('Hr_Msol.pdf', format = 'pdf')
plt.show()
