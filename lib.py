import numpy as np
import random

def generate_zero_determinant_matrix(i):
    random.seed(i)
    matrix_elements = [random.randint(1, 10) for _ in range(16)]
    matrix = np.array(matrix_elements).reshape(4, 4)
    while np.linalg.det(matrix) != 0:
        matrix_elements = [random.randint(1, 10) for _ in range(16)]
        matrix = np.array(matrix_elements).reshape(4, 4)
    return matrix
def fun(i):
    return [generate_zero_determinant_matrix(j) for j in range(i, -1, -1)]
def interlaced_matrix(input):
    interlaced = interlace(input)
    input_matrix = np.array(input).reshape(4, 4)
    interlaced_matrix = np.array(interlaced).reshape(4, 4)
    out = np.zeros((4, 4), dtype=np.uint8)
    for i in range(4):
        for j in range(4):
            out[i][j] = (input_matrix[i][j] * interlaced_matrix[i][j]) % 256
    return out.tolist()
def interlace(input):
    tmp = input
    for i in range(8):
      tmp[2 * i], tmp[2 * i + 1] = ( input[2*i] + input[2*i + 1] % 256 , input[2*i] * input[2*i + 1] % 256  )
    for i in range(4):
      tmp[4 * i], tmp[4 * i + 2] =  ( tmp[4 * i] + tmp[4 * i + 2] % 256 , tmp[4 * i] * tmp[4 * i + 2] % 256 )
      tmp[4 * i + 1], tmp[4 * i + 3] =  ( tmp[4 * i + 1] + tmp[4 * i + 3] % 256 , tmp[4 * i + 1] * tmp[4 * i + 3] % 256 )
    for i in range(2):
      for j in range(4):
        tmp[8 * i + j], tmp[8 * i + j + 4] = ( tmp[8 * i + j] + tmp[8 * i + j + 4] % 256 ,tmp[8 * i + j] * tmp[8 * i + j + 4] % 256 )
    for i in range(8):
      tmp[i], tmp[i + 8] = ( input[i] + input[i + 8] % 256 , input[i] * input[i + 8] % 256  )
    input_matrix = np.array(tmp).reshape(4, 4)
    out = np.zeros((4, 4), dtype=np.uint8)
    for i in range(4):
        out[i][0], out[i][1] = interlace8(input_matrix[i][0], input_matrix[i][1])
        out[i][2], out[i][3] = interlace8(input_matrix[i][2], input_matrix[i][3])
    tmp = out
    for i in range(8):
        out[i % 4][ i // 4 ], out[i % 4, (i // 4) + 2] = interlace8(tmp[i % 4,  i // 4], out[i % 4,  i // 4  + 2 ])
    for i in range(4):
      for j in range(4):
        out[i][j] = input_matrix[i][j]
    return out.flatten().tolist()
def interlace8(a, b):
    x = 0
    y = 0
    for i in reversed(range(4)):
        y <<= 1
        y |= (reverse_bits(a,8) >> i) & 1
        y <<= 1
        y |= (b >> i) & 1

    for i in reversed(range(4, 8)):
        x <<= 1
        x |= (reverse_bits(a,8) >> i) & 1
        x <<= 1
        x |= (b >> i) & 1
    return x, y
def reverse_bits(num, num_bits):
    # Mask to extract the least significant num_bits bits
    mask = (1 << num_bits) - 1
    # Extract the least significant num_bits bits
    bits = num & mask
    # Reverse the bits
    reversed_bits = 0
    for _ in range(num_bits):
        reversed_bits <<= 1
        reversed_bits |= bits & 1
        bits >>= 1
    # Combine the reversed bits with the rest of the number
    result = (num >> num_bits) << num_bits | reversed_bits
    return result

def hash_nim(input_str):
    input_bytes = bytearray(input_str.encode())
    input_bytes = [int(x) for x in input_bytes ]
    while len(input_bytes) % 16 != 0:
        input_bytes.append( (len(input_bytes) % 16) + 32 )
    chunks = [input_bytes[i:i + 16] for i in range(0, len(input_bytes), 16)]
    length = len(chunks)
    irreversible = fun(length)
    interlaced = [interlaced_matrix(chunk) for chunk in chunks]
    tmp = [ np.dot(irreversible[i], interlaced[i]) % 256 for i in range(0, length)]
    out = tmp[0]
    for i in range(1, length):
      out = np.dot(out, tmp[i]) % 256
    tmp = ""
    for matrix in out:
        for row in matrix.tolist():
            tmp += "".join(f"{row:02x}")
    return tmp
