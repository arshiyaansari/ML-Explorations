# Arshiya Ansari 
# Matician Coding Challenge

import numpy as np
from typing import List, Any 
import time

def convolve(m: List[List[int]], kernel: List[List[int]]) -> List[List[int]]:
    """
    Purpose: 
        Convolve the matrix horizontally and vertically. Used FFT for the linear convolutions based on literature.  
        To deal with border conditions, padding was added for both horizontal and vertical convolutions. 
    Params: 
        - m: 2D array, type: List[List[int]], dimensions: [nrows, ncols]
        - kernel: 2D array, type: List[List[int]] although it is only holds 1D data, dimensions: [nrows, 1] or [1, ncols]
    Return:
        Output matrix after linear convolution, dimensions: [nrows, ncols]
    """
    m_size = np.array(m.shape) # Find the dimensions of input matrix
    kernel_size = np.array(kernel.shape) # Find the dimensions of the kernel matrix 

    # Need to add padding so we can perform linear convolution 
    # Adding the padding will allow us to handle border cases
    padding = m_size + kernel_size # Add the dimension of input matrix and kernel

    # FFT convolution uses the principle that multiplication in the frequency domain corresponds to convolution in the time domain
    # More computationally efficient for large arrays 
    # Use try/except blocks in order to catch any discrepencies in the input data. 
    try:
        m_FFT = np.fft.fft2(m, padding) # Calculate the fourier transform of original matrix with padding
    except:
        raise ValueError("The matrix is not usable. Check to be sure it is a matrix.")

    try:
        kernel_FFT = np.fft.fft2(kernel, padding) # Calculate the fourier transform of kernel with padding 
    except:
        raise ValueError("The kernel is not usable. Check to be sure it is a matrix.")

    convolved_matrix = m_FFT * kernel_FFT # Convolve in the frequency domain by multiplying the two fourier transformed arrays
    convolved_matrix_inv = np.fft.ifft2(convolved_matrix) # Calculate the inverse fourier transform to restore to the original domain

    # Goal is our output matrix is the same dimensions as our input matrix so we need to mine only those values 
    ver_values = range(int((padding[1] - m_size[1]) // 2), int(m_size[1] + (padding[1] - m_size[1])//2)) # Calculate the vertical values for the unpadded matrix.
    hor_values = range(int((padding[0] - m_size[0]) // 2), int(m_size[0] + (padding[0] - m_size[0])//2)) # Calculate the horizontal values for the unpadded matrix

    result = convolved_matrix_inv[hor_values][:, ver_values] # Unpad the convolved matrix

    output = np.real(result) # Remove imaginary numbers that fft produces 

    return output # Return the convolved array


def find_max(m: List[List[int]]) -> int:
    """
    Purpose: 
        Find the maximum value in a 2D array (matrix)
    Params: 
        - m: 2D array, type: List[List[int]], dimensions: [nrows, ncols]
    Return: 
        Maximum integer value in a 2D array
    """
    return (max(map(max, m))) # Map the max function on each element in the matrix (each row), find the max of that 

def find_min(m: List[List[int]]) -> int:
    """
    Purpose: 
        Find the minimum value in a 2D array (matrix)
    Params: 
        m: 2D array, type: List[List[int]], dimensions: [nrows, ncols]
    Return: 
        Minimum integer value in a 2D array
    """
    return (min(map(min, m))) # Map the min function on each element in the matrix (each row), find the min of that 


if __name__ == "__main__":
    """
    Purpose: 
        Main function that reads matrix rows and columns from the command line and creates an unsigned char matrix M of size [rows x cols]
        Calls a convolve function that ensues 1D convolution on the matrix (both vertical and horizontal convolution)
    Return: 
        - Total time taken by local machine in computing the output matrices of Dx (horizontal convolution) and Dy (vertical convolution)
        - Min and max values of Dx and Dy 
    """
    # user input
    nrows = int(input("Number of rows: "))
    ncols = int(input("Number of columns: "))
    
    # creation of matrix using user input 
    # assumption: image matrix so integers are between 0 and 255
    mat = np.random.randint(0, 255,(nrows, ncols))

    # define horizontal kernel (given)
    hor_kernel = np.array([[-1, 0, 1]])

    # define vertical kernel (transposed horizontal kernel for vertical convolution)
    ver_kernel = hor_kernel.transpose() 


    print("**************\n")
    ticx = time.perf_counter()
    # create Dx: output for horizontal convolution
    Dx = convolve(m=mat, kernel=hor_kernel)
    tocx = time.perf_counter()
    print(f'Convoluted matrix along the x-axis in {tocx - ticx:0.4f} seconds')
    print(Dx)

    ticy = time.perf_counter()
    # create Dy: output for vertical convolution 
    Dy = convolve(m=mat, kernel=ver_kernel)
    tocy = time.perf_counter()
    print(f'Convoluted matrix along the y-axis in {tocy - ticy:0.4f} seconds')
    print(Dy)


    # find Dx min and max
    Dx_min = find_min(Dx)
    Dx_max = find_max(Dx)

    # find Dy min and max
    Dy_min = find_min(Dy)
    Dy_max = find_max(Dy)

    print("***********\n")
    print("DX Order Statistics\n")
    print(f'Dx Minimum Value: {Dx_min:0.1f}\n')
    print(f'Dx Maximum Value: {Dx_max:0.1f}\n')
    print("***********\n")
    print("DY Order Statistics\n")
    print(f'Dy Minimum Value: {Dy_min:0.1f}\n')
    print(f'Dy Maximum Value: {Dy_max:0.1f}\n')

    
    

