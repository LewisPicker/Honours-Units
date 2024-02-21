import os
import subprocess

solar_mass = [23,25]#,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,50,100]
Z_dir = ['0','6','5','4','3','002','01','02','03','04']

for i in Z_dir:
    os.mkdir('23M_' + i+'Z')
    os.chdir('23M_' + i + 'Z')
    os.system('pwd')
    os.system('rsync -rtvz lpicker@ozstar.swin.edu.au:/fred/oz148/lewis/mesa_dir/23Msun_' + i + 'Z_A2stars/history.data .')
    os.system('Rickolas88%')
    os.chdir('../')



