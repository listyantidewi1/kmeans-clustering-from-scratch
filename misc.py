import pandas as pd                          # Mengimpor modul pandas sebagai pd
import numpy as np                           # Mengimpor modul numpy sebagai np
import matplotlib.pyplot as plt              # Mengimpor modul matplotlib.pyplot sebagai plt
import warnings                              # Mengimpor modul warnings
from tkinter import Tk, Canvas, Button, Label, Entry  # Mengimpor beberapa komponen tkinter
import seaborn as sns                        # Mengimpor modul seaborn sebagai sns
from sklearn.cluster import KMeans            # Mengimpor KMeans dari modul sklearn.cluster

# Mengabaikan peringatan
warnings.filterwarnings('ignore', category=Warning)

# Membaca dataset
data = pd.read_csv('clustering.csv')          # Membaca file clustering.csv dan menyimpannya dalam variabel data
X = data[["LoanAmount","ApplicantIncome"]]    # Memilih kolom LoanAmount dan ApplicantIncome dari data

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
    k = int(entry.get())                                            # Mengambil nilai K dari input pengguna
    kmeans = KMeans(n_clusters=k)                                   # Membuat objek KMeans dengan jumlah klaster K
    kmeans.fit(X)                                                   # Melatih model KMeans pada data
    centroids = kmeans.cluster_centers_                             # Mengambil koordinat centroid
    
    # Plot klaster
    for i in range(k):                                              # Iterasi melalui setiap klaster
        cluster_points = X[kmeans.labels_ == i]                      # Mengambil titik dalam klaster ke-i
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], cmap='viridis')  # Membuat scatter plot untuk titik dalam klaster dengan skala warna viridis
        plt.scatter(centroids[i][0], centroids[i][1], c='red', marker='x')  # Menambahkan scatter plot untuk centroid dengan warna merah dan tanda silang
        plt.text(centroids[i][0], centroids[i][1], f'Cluster {i+1}', fontsize=12, color='red')  # Menambahkan teks label untuk centroid
    
    plt.xlabel('Loan Amount (Dalam Ribuan)')                        # Memberi label sumbu x dengan "Loan Amount (Dalam Ribuan)"
    plt.ylabel('Pendapatan')                                        # Memberi label sumbu y dengan "Pendapatan"
    plt.show()                                                      # Menampilkan plot
    
    # Menghitung nilai distorti
    distortion = kmeans.inertia_ / len(X)                           # Menghitung distorti
    distortion_label.config(text=f"Nilai Distorsi: {distortion}")   # Menampilkan nilai distorti di GUI
    
    # Menghitung skor siluet
    silhouette_avg = silhouette_score(X, kmeans.labels_)             # Menghitung rata-rata skor siluet
    silhouette_label.config(text=f"Skor Siluet: {silhouette_avg}")  # Menampilkan skor siluet di GUI

# Menampilkan GUI
root = Tk()                                                        # Membuat objek root Tkinter
root.title("K-Means Clustering")                                   # Mengatur judul jendela

# Menambahkan entry untuk input jumlah klaster
entry_label = Label(root, text="Masukkan Jumlah Klaster:", font=("Arial", 12))  # Label untuk input pengguna
entry_label.pack()                                                 # Menampilkan label di GUI
entry = Entry(root, font=("Arial", 12))                            # Entry untuk input pengguna
entry.pack()                                                       # Menampilkan entry di GUI

# Tombol untuk memicu tindakan berbeda
visualizeScatterMatrixButton = Button(root, text="Tampilkan Scatter Matrix", command=visualize_scatter_matrix)
visualizeScatterMatrixButton.pack()                                # Menampilkan tombol di GUI

visualizeScatterPlotRawButton = Button(root, text="Tampilkan Plot Data Mentah", command=visualize_scatter_plot_raw_data)
visualizeScatterPlotRawButton.pack()                               # Menampilkan tombol di GUI

displayInitialCentroidsButton = Button(root, text="Tampilkan Centroid Awal", command=initial_centroids)
displayInitialCentroidsButton.pack()                               # Menampilkan tombol di GUI

buttonClustering = Button(root, text="Analisis Klaster", command=kmeans)
buttonClustering.pack()                                            # Menampilkan tombol di GUI

# Label untuk menampilkan nilai distortion
distortion_label = Label(root, text="", font=("Arial", 12))
distortion_label.pack()                                            # Menampilkan label di GUI

# Label untuk menampilkan skor siluet
silhouette_label = Label(root, text="", font=("Arial", 12))
silhouette_label.pack()                                            # Menampilkan label di GUI

root.mainloop()                                                    # Menjalankan loop utama GUI
