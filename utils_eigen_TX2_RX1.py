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

def eigen_decoder_TX2_RX1(y_eigen : np.ndarray , H : np.ndarray, u_max : np.ndarray):
    """ Eigen decoder
    Args:
        y_eigen (np.ndarray): Received symbols (2xN)
        H (np.ndarray): Channel matrix (2x2)
    Returns:
        z (np.ndarray): Decoded symbols
    """

    y1 = y_eigen[0,:]
    batch_size = len(y1)
    y_eigen = np.stack([y1], axis=1)
    z1 = np.zeros(batch_size, dtype=np.complex64)
    umax_array = np.conj(u_max)

    for i in range(len(y1)):
        r1 = y1[i]     # received symbol by antenna 1 at time ti
        umax = umax_array[:,i]
        z1[i] = umax[0] * r1
    z = tf.stack([z1], axis=1)
    z = tf.convert_to_tensor(z)
    return z

def eigen_Sionna_TX2_RX1(y_eigen : np.ndarray , h_eigen : np.ndarray, rg : sn.ofdm.ResourceGrid, batch_size : int, OFDM_pilots_time : list, u_max : np.ndarray, H : np.ndarray):
    """ eigen decoder
    Args:
        y_eigen (np.ndarray) [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size] : Received symbols
        h_eigen (np.ndarray) [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]: Channel matrix
        rg (sn.ofdm.ResourceGrid): Resource grid -> will be used to know where the data bits
    Returns:
        z (np.ndarray): Decoded symbols
    """
    num_ofdm_symbols = rg.num_ofdm_symbols
    effective_subcarrier_ind = rg.effective_subcarrier_ind

    #time slot that has data
    effective_time_ind = np.arange(len(y_eigen[0,0,0,:,0]))
    effective_time_ind = np.delete(effective_time_ind, OFDM_pilots_time)

    y_eigen_effective_subcarrier = y_eigen.numpy()

    z = np.zeros((batch_size, 1, 1, num_ofdm_symbols , 76), dtype = np.complex64)
    for i in range(batch_size):
        for j in range(num_ofdm_symbols):
            y_rx_1 = y_eigen_effective_subcarrier[i,0,0,j,:] #received symbol by antenna 1 at time time j

            y_temp = np.zeros((1,76), dtype = np.complex64)
            y_temp[0,:] = y_rx_1
            H_temp = H[j,:,:,:]
            u_max_temp = u_max[j,:,:]
            z_temp = eigen_decoder_TX2_RX1(y_temp, H_temp , u_max_temp).numpy()
            z[i,:,0,j,:] = z_temp.T

    z_res = z[:,:,:,effective_time_ind,:]
    #only keep the subcarrier index present in the effective subcarrier index
    z_res = z_res[:,:,:,:,effective_subcarrier_ind]

    return z_res

def eigen_Sionna_ENCODER_TX2_RX1(y_eigen : np.ndarray , h_eigen : np.ndarray, rg : sn.ofdm.ResourceGrid, batch_size : int, OFDM_pilots_time : list):
    """ eigen decoder
    Args:
        y_eigen (np.ndarray) [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size] : Sent symbols
        h_eigen (np.ndarray) [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]: Channel matrix
        rg (sn.ofdm.ResourceGrid): Resource grid -> will be used to know where the data bits
    Returns:
        z (np.ndarray): Decoded symbols
    """
    num_ofdm_symbols = rg.num_ofdm_symbols
    num_tx = rg.num_tx
    num_streams_per_tx = rg.num_streams_per_tx

    #time slot that has data
    effective_time_ind = np.arange(len(y_eigen[0,0,0,:,0]))
    effective_time_ind = np.delete(effective_time_ind, OFDM_pilots_time)

    z = np.zeros((batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, y_eigen.shape[-1]), dtype = np.complex64)
    u_max = np.zeros((num_ofdm_symbols, num_tx,y_eigen.shape[-1]),np.complex64)
    H = np.zeros((num_ofdm_symbols, num_tx,num_tx,y_eigen.shape[-1]),np.complex64)

    for i in range(1):
        for j in range(num_ofdm_symbols):
            y_rx_1 = y_eigen[i,0,0,j,:]     #received symbol by antenna 1 at time time j
            h_11 = h_eigen[i,0,0,0,0,j,:]   #h11 -> rx0 -> tx0
            h_12 = h_eigen[i,0,0,1,0,j,:]   #h12 -> rx0 -> tx1
            y_temp = np.zeros((1,y_eigen.shape[-1]), dtype = np.complex64)
            y_temp[0,:] = y_rx_1
            H_temp = np.array([[h_11,h_12]])
            H[j,:,:,:] = H_temp


            v_max = np.zeros((H_temp.shape[1],H_temp.shape[2]),np.complex64)
            for k in range(H_temp.shape[2]):
              U_temp,S_temp,Vh_temp = np.linalg.svd(H_temp[:,:,k],full_matrices=True)
              u_max[j,0,k] = U_temp[0,0]
              v_max[0,k] = np.conj(Vh_temp[0,0])
              v_max[1,k] = np.conj(Vh_temp[0,1])
            x_temp_1 = v_max[0,:]*y_rx_1
            x_temp_2 = v_max[1,:]*y_rx_1
            z[i,:,0,j,:] = tf.stack([x_temp_1, x_temp_2], axis=1).numpy().T
            
    return z , u_max , H

def eigen_BER_different_SNR_TX2_RX1(x_eigen_OFDM , SNR , batch_size, rg,
                                    h_start_21,h_end_21, b, OFDM_pilots_time=[],
                                    num_bits_per_symbol=2, num_ofdm_symbols=2):
  """ Dominant eigenmode transmission BER for different SNR with 2 TX and 1 RX
  Args:
        x_eigen_OFDM (np.ndarray): Input symbols
        SNR (int): SNR value
        batch_size (int): Number of batches taken into account in the differents symbols (1)
        rg (sn.ofdm.ResourceGrid): Resource grid -> will be used to know where the data bits
        h_start_21 (np.ndarray): H matrix from the first chosen position
        h_end_21 (np.ndarray): H matrix from the second chosen position
        b (np.ndarray): binary bits encoded before transmission 
        OFDM_pilots_time (array): Position of the OFDM pilots in time domain, default: []
        num_bits_per_symbols (int): Number of bits per symbols, default: 2
        num_ofdm_symbols (int): Number of OFDM symbols, default: 2        
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
                            resource_grid = rg, normalize_channel= True)
  apply_channel = sn.channel.ApplyOFDMChannel(add_awgn = True)
  # Generation of the channel
  h = generate_channel(batch_size)

    # we use this if we use the precomputed channel with Ray tracing    
  if(h_start_21 is not None and h_end_21 is not None):
    h_new = np.zeros_like(h)
    h_new[0,:,0,:,:,0,:] = h_start_21[0,:,0,:,:,0,:]
    h_new[0,:,0,:,:,1,:] = h_end_21[0,:,0,:,:,0,:]
    for i in range(h_new.shape[0]):
      h_new[i] = h_new[0]
    h = tf.convert_to_tensor(h_new)

  x_eigen_OFDM, u_max, H = eigen_Sionna_ENCODER_TX2_RX1(x_eigen_OFDM , h_eigen = h, rg =rg, batch_size = batch_size, OFDM_pilots_time = OFDM_pilots_time)

  # Application of the genated channel and recuperation of the received signal y
  y_eigen = apply_channel([x_eigen_OFDM, h ,no])

  y_hat = eigen_Sionna_TX2_RX1(y_eigen, h, rg, batch_size, OFDM_pilots_time , u_max , H)
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