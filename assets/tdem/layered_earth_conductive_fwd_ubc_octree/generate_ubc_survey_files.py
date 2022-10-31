#%% IMPORT PACKAGES
import numpy as np


#%% DEFINE SURVEY PARAMETERS
xyz_tx = np.c_[0., 0., 5.]          # Transmitter location
xyz_rx = np.c_[10., 0., 5.]         # Receiver location
times = np.logspace(-5,-2,10)       # Times channels
tx_moment = 1.                      # Dipole moment of the transmitter
time_steps = [(5e-07, 40), (2.5e-06, 40), (1e-5, 40), (5e-5, 40), (2.5e-4, 40)]

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

fname = './survey.loc'
fid = open(fname, 'w')
fid.write("\nN_TRX 3")

for tx in tx_array:

    fid.write("\n\nTRX_ORIG\n5\n")
    np.savetxt(fid, tx, fmt='%.6e', delimiter=' ')

    fid.write('\nN_RECV 1\n')
    fid.write('N_TIME {}\n'.format(len(times)))

    out_array = np.c_[np.tile(xyz_rx, (len(times), 1)), times]
    np.savetxt(fid, out_array, fmt='%.6e', delimiter=' ')

fid.close()

# WAVEFORM
t = []
n = []
for x in time_steps:
    t.append(np.prod(x))
    n.append(x[1])

t = np.hstack(t)
n = np.hstack(n)

A = np.c_[np.cumsum(t), np.zeros(len(t)), n]

fname = './waveform_v1.txt'
fid = open(fname, 'w')
fid.write('0 1\n')
np.savetxt(fid, A, fmt='%g', delimiter=' ')
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

## WRITE TIME CHANNELS
fname = './times.txt'

f_array = np.c_[np.linspace(1, len(times), len(times)), times]
fid = open(fname, 'w')
np.savetxt(fid, f_array, fmt='%g', delimiter=' ')
fid.close()

# WRITE SURVEY INDEX FILES
fname = './index.loc'
fid = open(fname, 'w')
for ii in range(0, 3):
    for jj in range(0, 3):
        for kk in range(0, len(times)):
            fid.write("{} {} {} 1\n".format(ii+1, jj+1, kk+1))
fid.close()

# WAVEFORM
t = []
for x in time_steps:
    t.append(x[0]*np.ones(x[1]))

t = np.hstack(t)

A = np.c_[np.cumsum(t), np.ones(len(t)), np.zeros(len(t))]

fname = './waveform_v2.txt'
fid = open(fname, 'w')
fid.write('0 1 1\n')
np.savetxt(fid, A, fmt='%g', delimiter=' ')
fid.close()