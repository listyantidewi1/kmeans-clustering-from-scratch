import pandas as pd                          # Mengimpor modul pandas sebagai pd
import numpy as np                           # Mengimpor modul numpy sebagai np
import matplotlib.pyplot as plt              # Mengimpor modul matplotlib.pyplot sebagai plt
import warnings                              # Mengimpor modul warnings
from tkinter import Tk, Canvas, Button, Label # Mengimpor beberapa komponen tkinter
import seaborn as sns                        # Mengimpor modul seaborn sebagai sns

# Mengabaikan peringatan
warnings.filterwarnings('ignore', category=Warning)

# Membaca dataset
data = pd.read_csv('clustering.csv')          # Membaca file clustering.csv dan menyimpannya dalam variabel data
X = data[["LoanAmount","ApplicantIncome"]]    # Memilih kolom LoanAmount dan ApplicantIncome dari data

# Menentukan jumlah klaster (K)
K = 4                                          # Jumlah klaster yang ditentukan

# Langkah 1 dan 2 - Memilih centroid acak untuk setiap klaster
Centroids = X.sample(n=K, random_state=42)    # Memilih secara acak K titik dari data sebagai centroid awal dengan seed 42

# Mendefinisikan fungsi untuk visualisasi dan K-means clustering

# Fungsi untuk visualisasi scatter plot data mentah
def visualize_scatter_plot_raw_data():
    plt.scatter(X["LoanAmount"], X["ApplicantIncome"], c='black')  # Membuat scatter plot dari data LoanAmount dan ApplicantIncome dengan warna hitam
    plt.xlabel('Loan Amount')                                       # Memberi label sumbu x dengan "Loan Amount"
    plt.ylabel('Applicant Income')                                  # Memberi label sumbu y dengan "Applicant Income"
    plt.show()                                                      # Menampilkan plot

# Fungsi untuk visualisasi scatter matrix
def visualize_scatter_matrix():
    df = pd.DataFrame(X, columns=['ApplicantIncome', 'LoanAmount'])  # Membuat DataFrame dari data dengan kolom 'ApplicantIncome' dan 'LoanAmount'
    sns.pairplot(df)                                                # Membuat pairplot menggunakan seaborn
    plt.show()                                                      # Menampilkan plot

# Fungsi untuk visualisasi centroid awal
def initial_centroids():
    plt.scatter(X["LoanAmount"], X["ApplicantIncome"], c='black')   # Membuat scatter plot dari data LoanAmount dan ApplicantIncome dengan warna hitam
    plt.scatter(Centroids["LoanAmount"], Centroids["ApplicantIncome"], c='red')  # Menambahkan scatter plot untuk centroid awal dengan warna merah
    plt.xlabel('LoanAmount')                                        # Memberi label sumbu x dengan "LoanAmount"
    plt.ylabel('ApplicantIncome')                                   # Memberi label sumbu y dengan "ApplicantIncome"
    plt.show()                                                      # Menampilkan plot

# Fungsi untuk menghitung jarak antara dua titik
def euclidean_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5  # Menghitung jarak euclidean antara dua titik

# Fungsi untuk melakukan K-means clustering
def kmeans():
    centroids = Centroids.values                                    # Mengambil nilai centroid dari dataframe Centroids
    X_values = X.values                                             # Mengambil nilai data X
    
    while True:
        # Menetapkan setiap titik ke centroid terdekat
        cluster_assignments = []                                    # Inisialisasi daftar untuk menampung penugasan klaster
        for point in X_values:                                      # Iterasi melalui setiap titik dalam data
            distances = [euclidean_distance(point, centroid) for centroid in centroids]  # Menghitung jarak antara titik dan setiap centroid
            cluster_assignments.append(np.argmin(distances))       # Menambahkan indeks centroid terdekat ke daftar penugasan klaster
        
        # Memperbarui centroid
        new_centroids = []                                          # Inisialisasi daftar untuk menampung centroid baru
        for i in range(K):                                          # Iterasi melalui setiap klaster
            cluster_points = X_values[np.array(cluster_assignments) == i]  # Mengambil titik dalam klaster ke-i
            new_centroid = np.mean(cluster_points, axis=0)          # Menghitung centroid baru untuk klaster ke-i
            new_centroids.append(new_centroid)                      # Menambahkan centroid baru ke daftar centroid baru
        
        # Memeriksa konvergensi
        if np.allclose(centroids, new_centroids):                   # Memeriksa apakah centroid telah konvergen
            break                                                   # Keluar dari loop jika sudah konvergen
        
        centroids = np.array(new_centroids)                         # Memperbarui nilai centroid
    
    # Menghitung skor siluet
    silhouette_values = []                                          # Inisialisasi daftar untuk menyimpan nilai siluet
    for i, point in enumerate(X_values):                            # Iterasi melalui setiap titik dalam data
        cluster = cluster_assignments[i]                            # Mengambil klaster yang ditugaskan untuk titik tersebut
        cluster_points = X_values[np.array(cluster_assignments) == cluster]  # Mengambil titik dalam klaster yang sama
        a_i = np.mean([euclidean_distance(point, other_point) for other_point in cluster_points])  # Menghitung rata-rata jarak intra-klaster
        
        b_i = np.inf                                                # Menginisialisasi jarak minimum inter-klaster sebagai tak hingga
        for j in range(K):                                          # Iterasi melalui setiap klaster
            if j != cluster:                                        # Melewati klaster saat ini
                other_cluster_points = X_values[np.array(cluster_assignments) == j]  # Mengambil titik dalam klaster yang berbeda
                b_ij = np.mean([euclidean_distance(point, other_point) for other_point in other_cluster_points])  # Menghitung rata-rata jarak inter-klaster
                b_i = min(b_i, b_ij)                                # Memperbarui jarak minimum inter-klaster
        
        silhouette_values.append((b_i - a_i) / max(a_i, b_i))       # Menambahkan nilai siluet untuk titik tersebut ke daftar
        
    silhouette_avg = np.mean(silhouette_values)                    # Menghitung rata-rata skor siluet
    
    # Plot klaster
    for i in range(K):                                              # Iterasi melalui setiap klaster
        cluster_points = X_values[np.array(cluster_assignments) == i]  # Mengambil titik dalam klaster ke-i
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], cmap='viridis')  # Membuat scatter plot untuk titik dalam klaster dengan skala warna viridis
        plt.scatter(centroids[i][0], centroids[i][1], c='red', marker='x')  # Menambahkan scatter plot untuk centroid dengan warna merah dan tanda silang
        plt.text(centroids[i][0], centroids[i][1], f'Cluster {i+1}', fontsize=12, color='red')  # Menambahkan teks label untuk centroid
    
    plt.xlabel('Loan Amount (Dalam Ribuan)')                        # Memberi label sumbu x dengan "Loan Amount (Dalam Ribuan)"
    plt.ylabel('Pendapatan')                                        # Memberi label sumbu y dengan "Pendapatan"
    plt.show()                                                      # Menampilkan plot
    
    # Menampilkan skor siluet
    silhouette_label = Label(root, text=f"Skor Siluet: {silhouette_avg}", font=("Arial", 12))  # Membuat label tkinter untuk skor siluet
    silhouette_label.pack()                                         # Menampilkan label di GUI

# Menampilkan GUI
root = Tk()                                                        # Membuat objek root Tkinter
root.title("K-Means Clustering")                                   # Mengatur judul jendela
canvas = Canvas(root, width=300, height=100)                       # Membuat kanvas di jendela
canvas.pack()                                                      # Menampilkan kanvas
canvas.create_text(100, 10, text="K-Means Clustering", font=("Arial", 14), fill="black")  # Menambahkan teks di kanvas

description_label = Label(root, text="Deskripsi Data", font=("Arial", 14, "bold"))  # Membuat label deskripsi data
description_label.pack(pady=5)                                     # Menampilkan label di GUI

# Menghitung statistik tentang data
avgIncome = np.mean(X["ApplicantIncome"])                          # Menghitung rata-rata pendapatan
avgLoan = np.mean(X["LoanAmount"])                                 # Menghitung rata-rata pinjaman
stdDevIncome = np.std(X['ApplicantIncome'])                        # Menghitung standar deviasi pendapatan
stdDevLoan = np.std(X["LoanAmount"])                               # Menghitung standar deviasi pinjaman
maxIncome = np.max(X["ApplicantIncome"])                           # Menghitung pendapatan maksimum
minIncome = np.min(X["ApplicantIncome"])                           # Menghitung pendapatan minimum
maxLoan = np.max(X['LoanAmount'])                                  # Menghitung pinjaman maksimum
minLoan = np.min(X['LoanAmount'])                                  # Menghitung pinjaman minimum

# Menampilkan statistik di GUI
avgIncomeLabel = Label(root, text=f"Rata-rata Pendapatan : {avgIncome}", font=("Arial", 12))
avgIncomeLabel.pack()

avgLoanLabel = Label(root, text=f"Rata-rata Pinjaman : {avgLoan}", font=("Arial", 12))
avgLoanLabel.pack()

stdDevIncomeLabel = Label(root, text=f"Standar Deviasi Pendapatan : {stdDevIncome}", font=("Arial", 12))
stdDevIncomeLabel.pack()

stdDevLoanLabel = Label(root, text=f"Standar Deviasi Pinjaman : {stdDevLoan}", font=("Arial", 12))
stdDevLoanLabel.pack()

minIncomeLabel = Label(root, text=f"Pendapatan Minimum : {minIncome}", font=("Arial", 12))
minIncomeLabel.pack()

maxIncomeLabel = Label(root, text=f"Pendapatan Maksimum : {maxIncome}", font=("Arial", 12))
maxIncomeLabel.pack()

minLoanLabel = Label(root, text=f"Jumlah Pinjaman Minimum : {minLoan}", font=("Arial", 12))
minLoanLabel.pack()

maxLoanLabel = Label(root, text=f"Jumlah Pinjaman Maksimum : {maxLoan}", font=("Arial", 12))
maxLoanLabel.pack()

# Tombol untuk memicu tindakan berbeda
visualizeScatterMatrixButton = Button(root, text="Tampilkan Scatter Matrix", command=visualize_scatter_matrix)
visualizeScatterMatrixButton.pack()

visualizeScatterPlotRawButton = Button(root, text="Tampilkan Plot Data Mentah", command=visualize_scatter_plot_raw_data)
visualizeScatterPlotRawButton.pack()

displayInitialCentroidsButton = Button(root, text="Tampilkan Centroid Awal", command=initial_centroids)
displayInitialCentroidsButton.pack()

buttonClustering = Button(root, text="Analisis Klaster", command=kmeans)
buttonClustering.pack()

root.mainloop()                                                    # Menjalankan loop utama GUI
