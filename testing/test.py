import os
import numpy as np
import time
import csv
from qr import Gram_Schimidt, householder_reduce, Givens


def test_mean_error():
    ax=0
    for file in os.listdir("data/q_r_data"):
            # if file[-5:-4] not in ["1", "2", "3", "4", "5", "6", "7", "8", "0"]:
            # if file[-6:-4] == "00":

        print(f"testing mean error for {file}")
        if file.endswith(".npy"):
            path = os.path.join("data/q_r_data", file)
            Q, R, A = np.load(path)
            size = file[:-4]
            file_gram = open('test_results/MSE_Gram.txt', 'a')
            file_house = open('test_results/MSE_house.txt', 'a')

            start_time = time.time()

            Q1, R1  = Gram_Schimidt(A)
            # A1 = np.dot(Q1,R1)
            mse_gram = ((R - R1)**2).mean()
            file_gram.write(f'{str(size)}, {time.time() - start_time}, {mse_gram}')
            file_gram.write("\n")
            ### testing Gram
            start_time = time.time()

            Q2, R2  = householder_reduce(A)
            # A2 = np.dot(Q2,R2)
            mse_house = ((R - R2)**2).mean()
            file_house.write(f'{str(size)}, {time.time() - start_time}, {mse_house}')
            file_house.write("\n")

            # Close the file
            file_gram.close()
            file_house.close()  
    return 

def test_sparce():
    gram = []
    g_lines = csv.reader('gram_sparce_results.txt', delimiter=',')
    for row in g_lines:
        gram.append(int(row[0]))
    house = []
    h_lines = csv.reader('house_sparce_results.txt', delimiter=',')
    for row in h_lines:
        house.append(int(row[0]))

    for file in os.listdir("data/sparce_data"):
        print(f"testing sparce decomp. for {file}")
        if file.endswith(".npy"):
            path = os.path.join("data/sparce_data", file)
            A = np.load(path)
            size = file[:-4]
            file_gram = open('gram_sparce_results.txt', 'a')
            file_house = open('house_sparce_results.txt', 'a')

            if size not in gram:
                start_time = time.time()
                try:

                    Q1, R1  = Gram_Schimidt(A)
                
                    file_gram.write(f'{str(size)}, {time.time() - start_time}')
                    file_gram.write("\n")
                except Exception as e:
                    print(e)
            
            if size not in gram:
                ### testing House
                start_time = time.time()
                Q2, R2  = householder_reduce(A)
                file_house.write(f'{str(size)}, {time.time() - start_time}')
                file_house.write("\n")

            # Close the file
            file_gram.close()
            file_house.close()
    return 

def test_sparce_m():
    gram = []
    g_lines = csv.reader('gram_sparce_m_results.txt', delimiter=',')
    for row in g_lines:
        gram.append(int(row[0]))
    house = []
    h_lines = csv.reader('house_sparce_m_results.txt', delimiter=',')
    for row in h_lines:
        house.append(int(row[0]))

    for file in os.listdir("data/sparce_data/fixed_m"):
        print(f"testing sparce decomp. for {file}")
        if file.endswith(".npy"):
            path = os.path.join("data/sparce_data/fixed_m", file)
            A = np.load(path)
            size = file[:-4]
            file_gram = open('gram_sparce_m_results.txt', 'a')
            file_house = open('house_sparce_m_results.txt', 'a')

            if size not in gram:
                start_time = time.time()
                try:

                    Q1, R1  = Gram_Schimidt(A)
                
                    file_gram.write(f'{str(size)}, {time.time() - start_time}')
                    file_gram.write("\n")
                except Exception as e:
                    print(e)
            
            if size not in gram:
                ### testing House
                start_time = time.time()
                Q2, R2  = householder_reduce(A)
                file_house.write(f'{str(size)}, {time.time() - start_time}')
                file_house.write("\n")

            # Close the file
            file_gram.close()
            file_house.close()
    return 

def test_sparce_n():
    gram = []
    g_lines = csv.reader('gram_sparce_n_results.txt', delimiter=',')
    for row in g_lines:
        gram.append(int(row[0]))
    house = []
    h_lines = csv.reader('house_sparce_n_results.txt', delimiter=',')
    for row in h_lines:
        house.append(int(row[0]))

    for file in os.listdir("data/sparce_data/fixed_n"):
        print(f"testing sparce decomp. for {file}")
        if file.endswith(".npy"):
            path = os.path.join("data/sparce_data/fixed_n", file)
            A = np.load(path)
            size = file[:-4]
            file_gram = open('gram_sparce_n_results.txt', 'a')
            file_house = open('house_sparce_n_results.txt', 'a')

            if size not in gram:
                start_time = time.time()
                try:

                    Q1, R1  = Gram_Schimidt(A)
                
                    file_gram.write(f'{str(size)}, {time.time() - start_time}')
                    file_gram.write("\n")
                except Exception as e:
                    print(e)
            
            if size not in gram:
                ### testing House
                start_time = time.time()
                Q2, R2  = householder_reduce(A)
                file_house.write(f'{str(size)}, {time.time() - start_time}')
                file_house.write("\n")

            # Close the file
            file_gram.close()
            file_house.close()
    return 

def test_dense():
    gram = []
    g_lines = csv.reader('test_results/gram_dense_results.txt', delimiter=',')
    for row in g_lines:
        gram.append(int(row[0]))
    house = []
    h_lines = csv.reader('test_results/house_dense_results.txt', delimiter=',')
    for row in h_lines:
        house.append(int(row[0]))

    for file in os.listdir("data/dense_data"):
        print(f"testing dense decomp. for {file}")
        if file.endswith(".npy"):
            path = os.path.join("data/dense_data", file)
            A = np.load(path)
            size = file[:-4]
            file_gram = open('test_results/gram_dense_results.txt', 'a')
            file_house = open('test_results/house_dense_results.txt', 'a')

            if size not in gram:
                start_time = time.time()
                try:

                    Q1, R1  = Gram_Schimidt(A)
                
                    file_gram.write(f'{str(size)}, {time.time() - start_time}')
                    file_gram.write("\n")
                except Exception as e:
                    print(e)
            
            if size not in gram:
                ### testing House
                start_time = time.time()
                Q2, R2  = householder_reduce(A)
                file_house.write(f'{str(size)}, {time.time() - start_time}')
                file_house.write("\n")

            # Close the file
            file_gram.close()
            file_house.close()
    return    

def test_dense_m():
    gram = []
    g_lines = csv.reader('test_results/gram_dense_m_results.txt', delimiter=',')
    for row in g_lines:
        gram.append(int(row[0]))
    house = []
    h_lines = csv.reader('test_results/house_dense_m_results.txt', delimiter=',')
    for row in h_lines:
        house.append(int(row[0]))

    for file in os.listdir("data/dense_data/fixed_m"):
        print(f"testing dense decomp. for {file}")
        if file.endswith(".npy"):
            path = os.path.join("data/dense_data/fixed_m", file)
            A = np.load(path)
            size = file[:-4]
            file_gram = open('test_results/gram_dense_m_results.txt', 'a')
            file_house = open('test_results/house_dense_m_results.txt', 'a')

            if size not in gram:
                start_time = time.time()
                try:

                    Q1, R1  = Gram_Schimidt(A)
                
                    file_gram.write(f'{str(size)}, {time.time() - start_time}')
                    file_gram.write("\n")
                except Exception as e:
                    print(e)
            
            if size not in gram:
                ### testing House
                start_time = time.time()
                Q2, R2  = householder_reduce(A)
                file_house.write(f'{str(size)}, {time.time() - start_time}')
                file_house.write("\n")

            # Close the file
            file_gram.close()
            file_house.close()
    return  

def test_dense_n():
    gram = []
    g_lines = csv.reader('test_results/gram_dense_n_results.txt', delimiter=',')
    for row in g_lines:
        gram.append(int(row[0]))
    house = []
    h_lines = csv.reader('test_results/house_dense_n_results.txt', delimiter=',')
    for row in h_lines:
        house.append(int(row[0]))

    for file in os.listdir("data/dense_data/fixed_n"):
        print(f"testing dense decomp. for {file}")
        if file.endswith(".npy"):
            path = os.path.join("data/dense_data/fixed_n", file)
            A = np.load(path)
            size = file[:-4]
            file_gram = open('test_results/gram_dense_n_results.txt', 'a')
            file_house = open('test_results/house_dense_n_results.txt', 'a')

            if size not in gram:
                start_time = time.time()
                try:

                    Q1, R1  = Gram_Schimidt(A)
                
                    file_gram.write(f'{str(size)}, {time.time() - start_time}')
                    file_gram.write("\n")
                except Exception as e:
                    print(e)
            
            if size not in gram:
                ### testing House
                start_time = time.time()
                Q2, R2  = householder_reduce(A)
                file_house.write(f'{str(size)}, {time.time() - start_time}')
                file_house.write("\n")

            # Close the file
            file_gram.close()
            file_house.close()
    return  

  


test_mean_error()
test_sparce()
test_dense()
test_dense_m()
test_sparce_m()
test_dense_n()
test_sparce_n()
