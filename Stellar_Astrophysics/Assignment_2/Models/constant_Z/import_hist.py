import os
import subprocess

solar_mass = [23,25]#,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,50,100]

for i in solar_mass:
    #os.mkdir('mkdir '+str(i)+'Msun_2Z_sc01')
    os.chdir(str(i) + 'M_005Z')
    os.system('pwd')
    os.system('rsync -rtvz lpicker@ozstar.swin.edu.au:/fred/oz148/lewis/mesa_dir/'+str(i)+'Msun_005Z_A2stars/history.data .')
    #os.system('Rickolas88%')
    os.chdir('../')



