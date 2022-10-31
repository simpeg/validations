#%% IMPORT PACKAGES
import numpy as np


#%% DEFINE SURVEY PARAMETERS
xyz_tx = np.c_[0., 0., 5.]           # Transmitter location
xyz_rx = np.c_[10., 0., 5.]          # Receiver location
frequencies = np.logspace(2,5,10)    # Frequencies
tx_moment = 1.                       # Dipole moment of the transmitter

#%% LOOPS (RIGHT-HANDED)
x_loop_array = (np.sqrt(tx_moment)/2)*np.r_[
    np.c_[0, -1, -1], np.c_[0, 1, -1], np.c_[0, 1, 1], np.c_[0, -1, 1], np.c_[0, -1, -1]
]

y_loop_array = (np.sqrt(tx_moment)/2)*np.r_[
    np.c_[-1, 0, -1], np.c_[-1, 0, 1], np.c_[1, 0, 1], np.c_[1, 0, -1], np.c_[-1, 0, -1]
]

z_loop_array = (np.sqrt(tx_moment)/2)*np.r_[
    np.c_[-1, -1, 0], np.c_[1, -1, 0], np.c_[1, 1, 0], np.c_[-1, 1, 0], np.c_[-1, -1, 0]
]


#%% WRITE UBC FORMAT V1
fname = './survey.loc'
fid = open(fname, 'w')
fid.write("\nN_TRX {}\n\n".format(3*len(frequencies)))

tx_array = []

tx_array.append(np.c_[
    xyz_tx[0, 0] + x_loop_array[:, 0],
    xyz_tx[0, 1] + x_loop_array[:, 1],
    xyz_tx[0, 2] + x_loop_array[:, 2]
])

tx_array.append(np.c_[
    xyz_tx[0, 0] + y_loop_array[:, 0],
    xyz_tx[0, 1] + y_loop_array[:, 1],
    xyz_tx[0, 2] + y_loop_array[:, 2]
])

tx_array.append(np.c_[
    xyz_tx[0, 0] + z_loop_array[:, 0],
    xyz_tx[0, 1] + z_loop_array[:, 1],
    xyz_tx[0, 2] + z_loop_array[:, 2]
])

for tx in tx_array:
	for jj in range(0, len(frequencies)):

	    fid.write("TRX_ORIG\n5\n")
	    np.savetxt(fid, tx, fmt='%.6e', delimiter=' ')

	    fid.write('\nFREQUENCY {}\n'.format(frequencies[jj]))
	    fid.write('N_RECV 1\n')
	    fid.write('{} {} {}\n\n'.format(xyz_rx[0, 0], xyz_rx[0, 1], xyz_rx[0, 2]))

fid.close()


#%% WRITE UBC FORMAT V2

# WRITE TRANSMITTER FILE
fname = './transmitters.txt'
fid = open(fname, 'w')
fid.close()

fid = open(fname, 'a')

fid.write("\n1 5 1\n")
tx_array = np.c_[
    xyz_tx[0, 0] + x_loop_array[:, 0],
    xyz_tx[0, 1] + x_loop_array[:, 1],
    xyz_tx[0, 2] + x_loop_array[:, 2]
]
np.savetxt(fid, tx_array, fmt='%.6e', delimiter=' ')

fid.write("\n\n2 5 1\n")
tx_array = np.c_[
    xyz_tx[0, 0] + y_loop_array[:, 0],
    xyz_tx[0, 1] + y_loop_array[:, 1],
    xyz_tx[0, 2] + y_loop_array[:, 2]
]
np.savetxt(fid, tx_array, fmt='%.6e', delimiter=' ')

fid.write("\n\n3 5 1\n")
tx_array = np.c_[
    xyz_tx[0, 0] + z_loop_array[:, 0],
    xyz_tx[0, 1] + z_loop_array[:, 1],
    xyz_tx[0, 2] + z_loop_array[:, 2]
]
np.savetxt(fid, tx_array, fmt='%.6e', delimiter=' ')

fid.close()

# WRITE RECEIVERS FILE
fname = './receivers.txt'
fid = open(fname, 'w')
fid.close()

fid = open(fname, 'a')

fid.write("\n1 5 1\n")
rx_array = np.c_[
    xyz_rx[0, 0] + x_loop_array[:, 0],
    xyz_rx[0, 1] + x_loop_array[:, 1],
    xyz_rx[0, 2] + x_loop_array[:, 2]
]
np.savetxt(fid, rx_array, fmt='%.6e', delimiter=' ')

fid.write("\n\n2 5 1\n")
rx_array = np.c_[
    xyz_rx[0, 0] + y_loop_array[:, 0],
    xyz_rx[0, 1] + y_loop_array[:, 1],
    xyz_rx[0, 2] + y_loop_array[:, 2]
]
np.savetxt(fid, rx_array, fmt='%.6e', delimiter=' ')

fid.write("\n\n3 5 1\n")
rx_array = np.c_[
    xyz_rx[0, 0] + z_loop_array[:, 0],
    xyz_rx[0, 1] + z_loop_array[:, 1],
    xyz_rx[0, 2] + z_loop_array[:, 2]
]
np.savetxt(fid, rx_array, fmt='%.6e', delimiter=' ')

fid.close()

## WRITE FREQUENCIES FILE
fname = './frequencies.txt'

f_array = np.c_[np.linspace(1, len(frequencies), len(frequencies)), frequencies]
fid = open(fname, 'w')
np.savetxt(fid, f_array, fmt='%g', delimiter=' ')
fid.close()

# WRITE SURVEY INDEX FILE
fname = './index.loc'
fid = open(fname, 'w')

for ii in range(0, 3):
	for jj in range(0, len(frequencies)):
	    fid.write("{} {} 1 1\n".format(ii+1, jj+1))
	    fid.write("{} {} 2 1\n".format(ii+1, jj+1))
	    fid.write("{} {} 3 1\n".format(ii+1, jj+1))

fid.close()