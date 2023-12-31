import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc
import os

gpu_num = 0 # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import Sionna
try:
    import sionna as sn
    if(sn.__version__ != '0.15.1'):
        print("WARNING: Sionna version is not 0.15.1, but " + sn.__version__)
except ImportError as e:
    # Install Sionna if package is not already installed
    os.system("pip install sionna")
    import sionna as sn

print(sn.__version__)

# Import TensorFlow and NumPy
import tensorflow as tf
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

from sionna.utils import ebnodb2no, compute_ber
from sionna.mapping import Demapper


#--------------------------------------------------------------------------------------#
#                                Functions                                             #
#--------------------------------------------------------------------------------------#

def Alamouti_decoder_TX2_RX1(y_Alamouti_in : np.ndarray , h_Alamouti : np.ndarray):
    """ Alamouti decoder
    Args:
        y_Alamouti_in (np.ndarray): Received symbols (2xN)
        h_Alamouti (np.ndarray): Channel matrix (2x2)
    Returns:
        z (np.ndarray): Decoded symbols
    """

    y1 = y_Alamouti_in[0,:]
    batch_size = len(y1)
    y_Alamouti = np.stack([y1], axis=1)
    z1 = np.zeros(batch_size, dtype=np.complex64)
    for i in range(len(y1)//2):
        r0 = y1[2*i]     # received symbol by antenna 1 at time t1
        r1 = y1[2*i+1]   # received symbol by antenna 1 at time t2
        h0 = h_Alamouti[0,2*i]     # rx0 -> tx0
        h1 = h_Alamouti[1,2*i]       # rx0 -> tx1

        z1[2*i]   = np.conj(h0)*r0 + h1*np.conj(r1)
        z1[2*i+1] = np.conj(h1)*r0 - h0*np.conj(r1)
    z = tf.stack([z1], axis=1)
    z = tf.convert_to_tensor(z)
    return z


def Alamouti_Sionna_TX2_RX1(y_Alamouti : np.ndarray , h_Alamouti : np.ndarray, rg : sn.ofdm.ResourceGrid, batch_size : int, OFDM_pilots_time : list):
    """ Alamouti decoder
    Args:
        y_Alamouti (np.ndarray) [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size] : Received symbols
        h_Alamouti (np.ndarray) [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]: Channel matrix
        rg (sn.ofdm.ResourceGrid): Resource grid -> will be used to know where the data bits
    Returns:
        z (np.ndarray): Decoded symbols
    """
    num_ofdm_symbols = rg.num_ofdm_symbols
    effective_subcarrier_ind = rg.effective_subcarrier_ind
    #time slot that has data
    effective_time_ind = np.arange(len(y_Alamouti[0,0,0,:,0]))
    effective_time_ind = np.delete(effective_time_ind, OFDM_pilots_time)

    #delete the guard bands and the DC
    y_Alamouti_effective_subcarrier = y_Alamouti.numpy()[:,:,:,:,effective_subcarrier_ind]
    h_Alamouti_effective_subcarrier = h_Alamouti.numpy()[:,:,:,:,:,:,effective_subcarrier_ind]

    #delete the pilots
    y_Alamouti_effective_subcarrier = y_Alamouti_effective_subcarrier[:,:,:,effective_time_ind]
    h_Alamouti_effective_subcarrier = h_Alamouti_effective_subcarrier[:,:,:,:,:,effective_time_ind]

    z = np.zeros((batch_size, 1, 1, num_ofdm_symbols - len(OFDM_pilots_time) , 64), dtype = np.complex64)
    for i in range(batch_size):
        for j in range(num_ofdm_symbols - len(OFDM_pilots_time)):
            y_rx_1 = y_Alamouti_effective_subcarrier[i,0,0,j,:] #received symbol by antenna 1 at time time j
            h_11 = h_Alamouti_effective_subcarrier[i,0,0,0,0,j,:] #h11 -> rx0 -> tx0
            h_12 = h_Alamouti_effective_subcarrier[i,0,0,1,0,j,:] #h12 -> rx0 -> tx1
            y_temp = np.zeros((1,64), dtype = np.complex64)
            y_temp[0,:] = y_rx_1
            h_temp = np.zeros((2,64), dtype = np.complex64)
            h_temp[0,:] = h_11
            h_temp[1,:] = h_12
            z_temp = Alamouti_decoder_TX2_RX1(y_temp, h_temp).numpy()
            z[i,:,0,j,:] = z_temp.T
    return z

def Alamouti_1_BER_different_SNR_TX2_RX1(x_Alamouti_OFDM , SNR , batch_size,
                                         rg, h_start_21, h_end_21,b, num_bits_per_symbol=2, num_ofdm_symbols=2, OFDM_pilots_time=[]):
  """ Alamouti BER for different SNR when 2TX and 1 RX
  Args:
        x_Alamouti_OFDM (np.ndarray): Input symbols
        SNR (int): SNR value
        batch_size (int):
        rg ():
        h_start_21 (np.ndarray): H matrix from the first chosen position
        h_end_21 (np.ndarray): H matrix from the second chosen position
        b (): 
        num_bits_per_symbols (int): Number of bits per symbols, default: 2
        num_ofdm_symbols (int): Number of OFDM symbols, default: 2
        OFDM_pilots_time (array): Position of the OFDM pilots in time domain, default: []
  Returns:
       np.mean(BER) (int): Mean of the BER values computed 
  """  
  
  # Noise level
  no = ebnodb2no(ebno_db = SNR, num_bits_per_symbol = num_bits_per_symbol, coderate=1, resource_grid = rg)
  rayleigh = sn.channel.RayleighBlockFading(num_rx = 1,
                                num_rx_ant = 1,
                                num_tx = 2,
                                num_tx_ant = 1)
  #Generation of the OFDM Channel
  generate_channel = sn.channel.GenerateOFDMChannel(channel_model = rayleigh,
                            resource_grid = rg)
  apply_channel = sn.channel.ApplyOFDMChannel(add_awgn = True)
  # Generation of the channel
  h = generate_channel(batch_size=batch_size)

  # we use this if we use the precomputed channel with Ray tracing    
  if(h_start_21 is not None and h_end_21 is not None):
    h_new = np.zeros_like(h)
    h_new[0,:,0,:,:,0,:] = h_start_21[0,:,0,:,:,0,:]
    h_new[0,:,0,:,:,1,:] = h_end_21[0,:,0,:,:,0,:]
    for i in range(h_new.shape[0]):
      h_new[i] = h_new[0]
    h = tf.convert_to_tensor(h_new)

  # Application of the genated channel and recuperation of the received signal y
  y_Alamouti = apply_channel([x_Alamouti_OFDM, h ,no])

  y_hat = Alamouti_Sionna_TX2_RX1(y_Alamouti, h, rg, batch_size, OFDM_pilots_time)
  y_hat = tf.convert_to_tensor(y_hat)

  #Demapping
  demapper = Demapper("maxlog","qam", num_bits_per_symbol, hard_out=True)

  b_hat = demapper([y_hat,no])
  b_hat_refactor = np.zeros((batch_size,1,1,128*(num_ofdm_symbols-len(OFDM_pilots_time))))
  for j in range(batch_size):
      for i in range(num_ofdm_symbols-len(OFDM_pilots_time)):
          b_hat_refactor[j,:,:,i*128:(i+1)*128] = b_hat[j,:,:,i,:]

  ber = []
  for j in range(batch_size):
      ber_temp = compute_ber(b , b_hat_refactor[j,:,:,:]).numpy()
      ber.append(ber_temp)

  return np.mean(ber)








def Alamouti_2_Sionna_TX2_RX1(y_Alamouti : np.ndarray , h_Alamouti : np.ndarray, rg : sn.ofdm.ResourceGrid, batch_size : int, OFDM_pilots_time : list):
    """ Alamouti decoder
    Args:
        y_Alamouti (np.ndarray) [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size] : Received symbols
        h_Alamouti (np.ndarray) [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]: Channel matrix
        rg (sn.ofdm.ResourceGrid): Resource grid -> will be used to know where the data bits
    Returns:
        np.ndarray: Decoded symbols
    """
    num_ofdm_symbols = rg.num_ofdm_symbols
    num_tx = rg.num_tx
    num_streams_per_tx = rg.num_streams_per_tx

    effective_subcarrier_ind = rg.effective_subcarrier_ind
    #time slot that has data
    effective_time_ind = np.arange(len(y_Alamouti[0,0,0,:,0]))
    effective_time_ind = np.delete(effective_time_ind, OFDM_pilots_time)
    #delete the guard bands and the DC
    y_Alamouti_effective_subcarrier = y_Alamouti.numpy()[:,:,:,:,effective_subcarrier_ind]
    h_Alamouti_effective_subcarrier = h_Alamouti.numpy()[:,:,:,:,:,:,effective_subcarrier_ind]

    #delete the pilots
    y_Alamouti_effective_subcarrier = y_Alamouti_effective_subcarrier[:,:,:,effective_time_ind]
    h_Alamouti_effective_subcarrier = h_Alamouti_effective_subcarrier[:,:,:,:,:,effective_time_ind]
    z = np.zeros((batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols - len(OFDM_pilots_time) , 64), dtype = np.complex64)
    for i in range(batch_size):
        j = 0
        while j <= ((num_ofdm_symbols - len(OFDM_pilots_time))//2):
            r0 = y_Alamouti_effective_subcarrier[i,0,0,j,:] #received symbol by antenna 1 at time time j
            r1 = y_Alamouti_effective_subcarrier[i,0,0,j+1,:]
         
            # attention channel needs to be constant over 2 or 3 time slots
            h_11 = h_Alamouti_effective_subcarrier[i,0,0,0,0,j,:] #h11 -> rx0 -> tx0
            h_12 = h_Alamouti_effective_subcarrier[i,0,0,1,0,j,:] #h12 -> rx0 -> tx1


            s0 = np.conj(h_11)*r0 + h_12*np.conj(r1) # + np.conj(h_21)*r2 + h_22*np.conj(r3)
            s1 = np.conj(h_12)*r0 - h_11*np.conj(r1) # + np.conj(h_22)*r2 - h_21*np.conj(r3)
            z[i,0,0,j,:] = s0
            z[i,1,0,j,:] = s0
            z[i,0,0,j+1,:] = s1
            z[i,1,0,j+1,:] = s1
            j += 2


    return z


def Alamouti_2_BER_different_SNR_TX2_RX1(x_Alamouti_OFDM , SNR , batch_size, h_start,h_end,
                                  num_bits_per_symbol = 2, num_ofdm_symbols=2,
                                  OFDM_pilots_time=[], rg=None , b=None):
    """ Alamouti BER for different SNR (2nd scheme)
    Args:
        x_Alamouti_OFDM (np.ndarray): Input symbols
        SNR (int): SNR value
        batch_size (int):
        h_start (np.ndarray): H matrix from the first chosen position
        h_end (np.ndarray): H matrix from the second chosen position
        num_ofdm_symbols (int): Number of OFDM symbols, default: 2
        num_bits_per_symbols (int): Number of bits per symbols, default: 2
        OFDM_pilots_time (array): Position of the OFDM pilots in time domain, default: []
        rg (): ...., default: None
        b (): ..., default: None 
    Returns:
        np.mean(ber) (int): Mean of the BER values computed 
    """

    # Noise level
    no = ebnodb2no(ebno_db = SNR, num_bits_per_symbol = num_bits_per_symbol, coderate=1, resource_grid = rg)
    rayleigh = sn.channel.RayleighBlockFading(num_rx = 2,
                                num_rx_ant = 1,
                                num_tx = 2,
                                num_tx_ant = 1)
    #Generation of the OFDM Channel
    generate_channel = sn.channel.GenerateOFDMChannel(channel_model = rayleigh,
                            resource_grid = rg)
    apply_channel = sn.channel.ApplyOFDMChannel(add_awgn = True)
    # Generation of the channel
    h = generate_channel(batch_size=batch_size)
    h = h.numpy()
    for i in range(batch_size):
        for j in range((num_ofdm_symbols - len(OFDM_pilots_time))):
            h[i,:,:,:,:,j,:] = h[i,:,:,:,:,0,:]
    h = np.ones_like(h)
    h = tf.convert_to_tensor(h)

    if(h_start is not None and h_end is not None):
    #print("h_start and h_end are not None")
        h_new = np.zeros_like(h)
        h_new[0,:,0,:,:,0,:] = h_start[0,:,0,:,:,0,:]
        h_new[0,:,0,:,:,1,:] = h_end[0,:,0,:,:,0,:]
        for i in range(h_new.shape[0]):
            h_new[i] = h_new[0]
    h = tf.convert_to_tensor(h_new)

    # Application of the genated channel and recuperation of the received signal y
    y_Alamouti = apply_channel([x_Alamouti_OFDM, h ,no])

    y_hat = Alamouti_2_Sionna_TX2_RX1(y_Alamouti, h, rg, batch_size, OFDM_pilots_time)
    y_hat = tf.convert_to_tensor(y_hat)
    #Demapping
    demapper = Demapper("maxlog","qam", num_bits_per_symbol, hard_out=True)
    b_hat = demapper([y_hat,no])
    b_hat_refactor = np.zeros((batch_size,2,1,128*(num_ofdm_symbols-len(OFDM_pilots_time))))
    for j in range(batch_size):
        for i in range(num_ofdm_symbols-len(OFDM_pilots_time)):
            b_hat_refactor[j,:,:,i*128:(i+1)*128] = b_hat[j,:,:,i,:]

    ber = []
    for j in range(batch_size):
        ber_temp = compute_ber(b , b_hat_refactor[j,:,:,:]).numpy()
        ber.append(ber_temp)

    return np.mean(ber)
