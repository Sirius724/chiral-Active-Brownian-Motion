#!/bin/sh

#phi=(0.64)
omega=(0.001 0.01 0.02 0.04 0.1 0.3 1.0 3.0 5.0 0.0)
rho=(0.2 0.4 0.6 0.8,1.0)
tau_p=(1.0 2.0 5.0 10.0 20.0 50.0 100.0 200.0 400.0)
#tau_p=(2.0 5.0 50.0 100.0)
#tau_p=20.0
potential=(0 1 2)

#for ((i=0 ; i<4 ; i++)) # tau_p index
#	do
	for ((j=0 ; j<9 ; j++)) #omega index
			do
			#for ((k = 0 ; k < 4 ; k++)) #rho index
			#	do
				#for ((n = 0 ; n < 3 ; n++))
					qsub qsub.sh ${tau_p[i]} ${omega[j]} ${rho[1]} ${potential[1]} #${gamma[k*2]} ${gamma[k*2+1]} #${phi[i]} #variable input in bat file
#do rm ${a[i]}/*
			done
#		done
	#done
#done
