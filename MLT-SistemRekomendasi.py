#!/usr/bin/env python
# coding: utf-8

# # 0. Import Library

# In[ ]:


import numpy as np
import pandas as pd
import random

from zipfile import ZipFile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, Model

from pathlib import Path
import matplotlib.pyplot as plt

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import mean_squared_error, root_mean_squared_error
import optuna
from catboost import CatBoostRegressor


# # 1. Business Understanding

# # 2. Data Understanding

# ## 2.1 Load Data dan Cetak Atribut Tabel

# In[7]:


dataUser = pd.read_csv("/kaggle/input/book-reccomendation-dataset/Data/Users.csv")
dataBuku = pd.read_csv("/kaggle/input/book-reccomendation-dataset/Data/Books.csv")
dataRating = pd.read_csv("/kaggle/input/book-reccomendation-dataset/Data/Ratings.csv")

dataUser.info()
dataBuku.info()
dataRating.info()


# ## 2.2 Identifikasi Karakteristik Data dan Cetak Ringkasan Statistik

# In[8]:


dataUser = dataUser.rename(columns={
    'User-ID': 'userID'
})
dataRating = dataRating.rename(columns={
    'User-ID': 'userID'
})

print('Banyak User : ', len(dataUser.userID.unique()))
print('Banyak Buku : ', len(dataBuku.ISBN.unique()))
print('Banyak User yang Memberi Rating : ', len(dataRating.userID.unique()))


# ## 2.3 Identifikasi Ringkasan Statistik Data

# ### 2.3.1 Ringkasan Statistik Dataset Books

# In[9]:


dataBuku.drop(columns=['Image-URL-S','Image-URL-M','Image-URL-L','ISBN']).describe(include='all')


# ### 2.3.2 Ringkasan Statistik Dataset Ratings

# In[11]:


dataRating.drop(columns=['userID']).describe(include='all')


# ### 2.3.3 Ringkasan Statistik Dataset Users

# In[12]:


dataUser.drop(columns=['userID']).describe(include='all')


# ## 2.4 Pengecekan Missing Value

# In[13]:


dataUser.isnull().sum()


# **Penjelasan :**
# 
# Pada dataset Buku terlihat terdapat missing value pada data yaitu pada kolom Age. Hal ini menunjukkan diperlukan perlakuan agar tidak mengganggu analisis selanjutnya.

# In[14]:


dataBuku.isnull().sum()


# **Penjelasan :**
# 
# Pada dataset Buku terlihat terdapat missing value pada data yaitu pada kolom Book-Author, Publisher, dan Image-URL-L. Hal ini menunjukkan diperlukan perlakuan agar tidak mengganggu analisis selanjutnya.

# In[15]:


dataRating.isnull().sum()


# **Penjelasan :**
# 
# Pada dataset rating terlihat tidak terdapat missing value pada data.

# # 3. Data Preparation

# ## 3.1 Join Tabel

# In[16]:


# Gabungkan Data Rating dengan Data User untuk memperoleh dataset rating beserta atribut usernya
dataRatingUser = pd.merge(dataRating, dataUser, on='userID', how='left')
# Gabungkan dataset sebelumnya dengan dataset buku untuk memperoleh keterangan buku yang diberi rating
dataset = pd.merge(dataRatingUser, dataBuku, on='ISBN', how='left')


# In[17]:


dataset.info()
dataset.isnull().sum()


# ## 3.2 Eliminasi Observasi dengan Missing Value

# In[18]:


dataFinal = dataset.dropna()


# **Penjelasan :**
# 
# Setelah proses joining table, data missing value yang masih ada diberi perlakuan berupa Eliminasi Observasi. Hal ini bertujuan untuk tidak memberikan bias pada data jika dilakukan imputasi. Bias dapat muncul karena jumlah missing value yang cukup besar, sehingga berisiko jika diimputasi.

# ### 3.3.1 Preprocessing untuk Content Based Filtering

# In[ ]:


# Membentuk Variabel Baru yang menangkap atribut dari Buku
dataFinal['AtributBuku'] = dataFinal['Book-Title'] + ' ' + dataFinal['Book-Author'] + ' ' + dataFinal['Year-Of-Publication'].astype(str)

# Preprocessing: Convert to lowercase, remove punctuation
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    return text

dataFinal['AtributBuku'] = dataFinal['AtributBuku'].apply(preprocess_text)

# Undersampling untuk Reduksi Dataset
dataBaru = dataFinal.dropna(subset=['AtributBuku']).sample(n=90000, random_state=42).reset_index(drop=True)
# Buat dataframe baru berisi data unik dari AtributBuku
dataBaru = dataBaru.drop_duplicates(subset='AtributBuku').reset_index(drop=False)

# Membentuk Matriks Term Frequency - Inverse Document Frequency
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(dataBaru['AtributBuku'])


# ### 3.3.2 Preprocessing untuk Colaborative Filtering

# In[19]:


# Mengubah userID menjadi list unik
user_ids = dataFinal['userID'].unique().tolist()
user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}

# Mengubah ISBN menjadi list unik
buku_ids = dataFinal['ISBN'].unique().tolist()
buku_to_ids_encoded = {x: i for i, x in enumerate(buku_ids)}
ids_to_buku_encoded = {i: x for i, x in enumerate(buku_ids)}

# Melakukan encoding ke dataframe
dataFinal['user_encoded'] = dataFinal['userID'].map(user_to_user_encoded)
dataFinal['ISBN_encoded'] = dataFinal['ISBN'].map(buku_to_ids_encoded)

# Cek apakah ada yang gagal di-encode
if dataFinal[['user_encoded', 'ISBN_encoded']].isnull().values.any():
    print("🚨 Ada userID atau ISBN yang tidak berhasil di-encode!")
    print("Jumlah user_encoded NaN:", dataFinal['user_encoded'].isnull().sum())
    print("Jumlah ISBN_encoded NaN:", dataFinal['ISBN_encoded'].isnull().sum())
    # Bisa drop baris NaN atau isi dengan nilai default
    dataFinal = dataFinal.dropna(subset=['user_encoded', 'ISBN_encoded'])

# Convert ke integer untuk keperluan input model embedding
dataFinal['user_encoded'] = dataFinal['user_encoded'].astype(np.int32)
dataFinal['ISBN_encoded'] = dataFinal['ISBN_encoded'].astype(np.int32)

# Konversi rating ke float32
dataFinal['Book-Rating'] = dataFinal['Book-Rating'].astype(np.float32)

# Info statistik
num_users = len(user_to_user_encoded)
num_items = len(ids_to_buku_encoded)
min_rating = dataFinal['Book-Rating'].min()
max_rating = dataFinal['Book-Rating'].max()

print(f'Jumlah User: {num_users}, Jumlah Buku: {num_items}, Min Rating: {min_rating}, Max Rating: {max_rating}')


# In[20]:


# Mengacak dataset
df = dataFinal.sample(frac=1, random_state=42)
df

# Membuat variabel x untuk mencocokkan data user dan resto menjadi satu value
x = df[['user_encoded', 'ISBN_encoded']].values
 
# Membuat variabel y untuk membuat rating dari hasil 
y = df['Book-Rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
 
# Membagi menjadi 80% data train dan 20% data validasi
train_indices = int(0.8 * df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)


# # 4. Modelling

# ## 4.1 Content Based Filtering

# ### 4.1.1 Membentuk Matriks Cosine Similarity

# In[17]:


cosine_sim = cosine_similarity(tfidf_matrix, dense_output=False)


# ### 4.1.2 Pendefinisian Fungsi Rekomendasi Produk

# In[18]:


def recommend(book_title, cosine_sim=cosine_sim, jmlbuku = 5):
    print(f"Menampilkan Rekomendasi Buku yang sesuai dengan '{book_title}' ...")
    
    matches = dataBaru[dataBaru['Book-Title'] == book_title]
    if matches.empty:
        return f"Tidak ditemukan judul: {book_title}"
    
    # matches.index sudah 0..n-1, cocok untuk cosine_sim
    idx = matches.index[0]

    sim_scores_row = cosine_sim[idx].toarray().flatten()

    sim_scores = list(enumerate(sim_scores_row))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:jmlbuku+1]  # skip buku itu sendiri

    book_indices = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]

    result = dataBaru.iloc[book_indices][['Book-Title', 'Book-Author', 'Publisher', 'Year-Of-Publication']].copy()
    result['similarity_score'] = scores

    return result.reset_index(drop=True)


# In[19]:


recommend(dataBaru['Book-Title'].dropna().sample(1).iloc[0])


# **Penjelasan :**
# 
# Terlihat bahwa sistem rekomendasi yang dibangun dengan Content Based Filtering dapat mengembalikan buku yang memiliki karakteristik mirip dengan inputan yang diberikan. Pada pengujian tersebut, diberikan judul 'Life in the Rainforests (Life in the Series)' yang digunakan sebagai input ke fungsi. Terlihat bahwa fungsi mengembalikan 5 buku yang memiliki similarity tertinggi dengan judul sebelumnya. Terlihat pada Gambar 4, Nilai cosine similarity tertinggi yaitu 0,29 pada buku berjudul Life 101 : Everything We Wish We Had Learned A yang ditulis oleh author Peter McWilliams. Kesamaan judul ini dapat menjadi penyebab buku tersebut menjadi hasil kembalian fungsi. 

# ## 4.2 Collaborative Filtering

# ### 4.2.1 Pendefinisian Model RecommenderNet (Neural Colaborative Filtering)

# In[21]:


class RecommenderNet(Model):

    def __init__(self, num_users, num_buku, embedding_size, dropout_rate=0.3, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_buku = num_buku
        self.embedding_size = embedding_size
        self.dropout_rate = dropout_rate

        # User Embedding Layer
        self.user_embedding = layers.Embedding(
            input_dim=num_users,
            output_dim=embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=regularizers.l2(1e-6)
        )
        self.user_bias = layers.Embedding(input_dim=num_users, output_dim=1)

        # Item Embedding Layer
        self.buku_embedding = layers.Embedding(
            input_dim=num_buku,
            output_dim=embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=regularizers.l2(1e-6)
        )
        self.buku_bias = layers.Embedding(input_dim=num_buku, output_dim=1)

        # Dropout Layer
        self.dropout = layers.Dropout(dropout_rate)

        # Fully Connected Layer
        self.fc1 = layers.Dense(embedding_size, activation='relu')
        self.batch_norm = layers.BatchNormalization()
        self.fc2 = layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        # Extract user and item embeddings
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        buku_vector = self.buku_embedding(inputs[:, 1])
        buku_bias = self.buku_bias(inputs[:, 1])

        # Normalize embeddings
        user_vector = tf.nn.l2_normalize(user_vector, axis=1)
        buku_vector = tf.nn.l2_normalize(buku_vector, axis=1)

        # Compute dot product
        dot_product = tf.reduce_sum(user_vector * buku_vector, axis=1, keepdims=True)

        # Add biases
        x = dot_product + user_bias + buku_bias

        # Apply dropout and dense layers
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.batch_norm(x)
        x = self.fc2(x)

        return x


# In[22]:


model = RecommenderNet(num_users, num_items, 1024) # inisialisasi model
 
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='mse',
    metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')]
)


# ### 4.2.2 Pelatihan Model Rekomendasi

# In[23]:


# Memulai training
 
history = model.fit(
    x = x_train,
    y = y_train,
    batch_size = 1024,
    epochs = 10,
    validation_data = (x_val, y_val)
)


# ### 4.2.3 Visualisasi Hasil Pelatihan

# In[24]:


plt.plot(history.history['rmse'])
plt.plot(history.history['val_rmse'])
plt.title('model_metrics')
plt.ylabel('Root Mean Squared Error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# **Penjelasan :**
# 
# Terlihat bahwa model yang dibangun cenderung overfitting. Hal ini dilihat dar nilai evaluasi pada data uji dan data latih yang cukup besar Gap nya setelah beberapa epoch. 

# ### 4.2.4 Merekomendasikan Buku

# In[26]:


def recommend_books_for_user(model, user_id_asli, data, buku_df, user_to_user_encoded, buku_to_ids_encoded, ids_to_buku_encoded, top_n=10):
    # Cek apakah user_id valid
    if user_id_asli not in user_to_user_encoded:
        print("User ID tidak ditemukan.")
        return []

    user_encoded = user_to_user_encoded[user_id_asli]
    buku_user_sudah_rating = data[data['userID'] == user_id_asli]['ISBN'].tolist()
    buku_user_sudah_encoded = [buku_to_ids_encoded[buku] for buku in buku_user_sudah_rating]

    # Semua buku yang belum diberi rating oleh user
    buku_tidak_dirating = list(set(buku_to_ids_encoded.values()) - set(buku_user_sudah_encoded))
    user_encoded_array = np.full(len(buku_tidak_dirating), user_encoded)
    
    # Membuat data pasangan user-buku
    input_pairs = np.array(list(zip(user_encoded_array, buku_tidak_dirating)))

    # Prediksi skor
    ratings = model.predict(input_pairs).flatten()
    
    # Ambil top-N
    top_indices = ratings.argsort()[-top_n:][::-1]
    top_buku_encoded = [buku_tidak_dirating[i] for i in top_indices]
    top_buku_isbn = [ids_to_buku_encoded[i] for i in top_buku_encoded]

    # Gabungkan dengan info buku (jika ada dataframe buku_df)
    rekomendasi = buku_df[buku_df['ISBN'].isin(top_buku_isbn)].copy() if buku_df is not None else pd.DataFrame({'ISBN': top_buku_isbn})
    rekomendasi['Predicted_Score'] = ratings[top_indices]
    
    return rekomendasi.sort_values(by='Predicted_Score', ascending=False)

# Menampilkan 5 userID unik pertama dari data
sample_user_ids = dataFinal['userID'].unique()[:5]
user_id_sample = sample_user_ids[0]  # atau pilih indeks lain
print('Memberikan Rekomendasi untuk User : ',user_id_sample)

# Contoh penggunaan:
rekomendasi_df = recommend_books_for_user(model, user_id_asli=user_id_sample, data=dataFinal, buku_df=dataBuku,
    user_to_user_encoded=user_to_user_encoded,
    buku_to_ids_encoded=buku_to_ids_encoded,
    ids_to_buku_encoded=ids_to_buku_encoded,
    top_n=5)

# Tampilkan hasil
display(rekomendasi_df)


# **Penjelasan :**
# 
# Berdasarkan pengujian yang dilakukan terhadap sistem rekomendasi yang dibangun dengan RecommenderNet, terlihat bahwa sistem dapat mengembalikan daftar Buku melalui karakteristik buku disertai prediksi rating yang telah distandardisasi.

# ## 4.3.1 Pendefinisian Model Rekomendasi dengan Catboost

# In[26]:


def sistemRekomendasiCatboost(x_train, y_train, x_val, y_val, n_trials=10, random_state=42):
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'random_state': random_state,
            'verbose': 0
        }

        model = CatBoostRegressor(**params)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        return rmse

    # Optimasi dengan Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    print("Best Parameters:", study.best_params)
    print("Best RMSE:", study.best_value)

    # Latih ulang model terbaik
    best_params = study.best_params

    best_model = CatBoostRegressor(**best_params)
    best_model.fit(x_train, y_train)

    return best_model, study


# ### 4.3.2 Rekomendasi Buku

# In[30]:


def recommend_books_cb(
    model, 
    user_id_asli, 
    data, 
    buku_df, 
    user_to_user_encoded, 
    buku_to_ids_encoded, 
    ids_to_buku_encoded, 
    top_n=5
):
    if user_id_asli not in user_to_user_encoded:
        return pd.DataFrame()

    user_enc = user_to_user_encoded[user_id_asli]
    all_buku_encoded = list(buku_to_ids_encoded.values())

    # Buat kombinasi user_encoded dengan semua buku_encoded
    user_buku_pairs = np.array([[user_enc, buku_enc] for buku_enc in all_buku_encoded])

    # Prediksi rating
    pred_ratings = model.predict(user_buku_pairs)

    # Ambil top_n hasil
    top_indices = np.argsort(pred_ratings)[::-1][:top_n]
    top_buku_encoded = [all_buku_encoded[i] for i in top_indices]

    top_isbns = [ids_to_buku_encoded[enc] for enc in top_buku_encoded]
    top_pred_scores = pred_ratings[top_indices]

    # Buat DataFrame hasil rekomendasi
    recommendations = pd.DataFrame({
        'ISBN': top_isbns,
        'predicted_rating': top_pred_scores
    })

    # Merge dengan buku_df untuk menambahkan informasi buku
    buku_info_cols = ['ISBN', 'Book-Title', 'Book-Author', 'Publisher', 'Year-Of-Publication']
    final_df = recommendations.merge(buku_df[buku_info_cols], on='ISBN', how='left')

    return final_df


# ### 4.3.3 Pengujian Sistem Rekomendasi

# In[28]:


model_cb, study = sistemRekomendasiCatboost(x_train, y_train, x_val, y_val)


# In[32]:


# Ambil 1 user unik dari data
sample_user = dataFinal['userID'].iloc[0]
print("Sample user:", sample_user)
top_books = recommend_books_cb(model_cb, sample_user, dataFinal, dataBuku, user_to_user_encoded, buku_to_ids_encoded, ids_to_buku_encoded, top_n=5)
top_books


# **Penjelasan :**
# 
# Berdasarkan pengujian yang dilakukan terhadap sistem rekomendasi yang dibangun dengan Algoritma Catboost, terlihat bahwa sistem dapat mengembalikan daftar Buku melalui nilai ISBN nya.

# # 5. Evaluasi

# ## 5.1 Evaluasi Content Based Filtering : Cosine Similarity

# In[ ]:


def evaluate_cbf_recommender(dataBaru, recommend_fn, users_to_evaluate, k=5):
    results = []
    user_books = dataBaru.groupby("userID")["Book-Title"].apply(set).to_dict()

    for user in users_to_evaluate:
        books_read = list(user_books.get(user, []))
        if not books_read:
            continue
        seed_book = pd.Series(books_read).dropna().sample(1).iloc[0]
        try:
            rec_df = recommend_fn(seed_book, cosine_sim, len(books_read))
        except:
            continue
        rec_books = rec_df["Book-Title"].tolist()[:k]
        actual_books = set(books_read)
        relevance_vector = [1 if book in actual_books else 0 for book in rec_books]
        true_positives = sum(relevance_vector)

        precision = true_positives / k if k > 0 else 0
        recall = true_positives / len(actual_books) if actual_books else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        hit_rate = 1 if true_positives > 0 else 0

        results.append({
            "userID": user,
            "seed_book": seed_book,
            "recommended": rec_books,
            "actual": list(actual_books),
            "relevance_vector": relevance_vector,
            "precision@k": precision,
            "recall@k": recall,
            "f1_score@k": f1_score,
            "hit_rate@k": hit_rate
        })

    results_df = pd.DataFrame(results)
    avg_precision = results_df["precision@k"].mean()
    avg_recall = results_df["recall@k"].mean()
    avg_f1 = results_df["f1_score@k"].mean()
    avg_hit_rate = results_df["hit_rate@k"].mean()

    return results_df, avg_precision, avg_recall, avg_f1, avg_hit_rate

# Hitung jumlah buku per user
user_book_counts = dataBaru.groupby("userID")["Book-Title"].nunique()

# Filter user dengan minimal 5 buku
eligible_users = user_book_counts[user_book_counts >= 5].index.tolist()

# Ambil sample user dari yang eligible
sample_users = random.sample(eligible_users, 10)

# Jalankan evaluasi
results_df, avg_prec, avg_rec, avg_f1, avg_hr = evaluate_cbf_recommender(dataBaru, recommend, sample_users, k=5)

print(f"Avg Precision@5: {avg_prec:.4f}")
print(f"Avg Recall@5: {avg_rec:.4f}")
print(f"Avg F1-Score@5: {avg_f1:.4f}")
print(f"Avg Hit Rate@5: {avg_hr:.4f}")

results_df.head()


# **Penjelasan :**
# 
# Berdasarkan ukuran evaluasi yang didefinisikan, kedua sistem rekomendasi yang dibangun dievaluasi untuk mengetahui seberapa baik sistem dapat memberikan rekomendasi pengguna berdasarkan konten yang serupa ataupun preferensi pengguna lain yang mirip. 
# Pada Content Based Filtering, sistem dievaluasi terhadap 10 pengguna diperoleh ukuran evaluasi rata-rata sebagai berikut.
# 
# Avg Precision@5: 0.0200<br>
# Avg Recall@5: 0.0200<br>
# Avg F1-Score@5: 0.0200<br>
# Avg Hit Rate@5: 0.1000<br>

# ## 5.2 Evaluasi Colaborative Filtering

# ### Recommender Net

# In[27]:


def evaluate_cf_model(
    model, 
    x_val, 
    y_val, 
    dataFinal, 
    buku_df,
    user_to_user_encoded, 
    buku_to_ids_encoded, 
    ids_to_buku_encoded, 
    k=5
):
    # Buat DataFrame val dari x_val dan y_val
    val_df = pd.DataFrame({
        "user_encoded": x_val[:, 0],
        "buku_encoded": x_val[:, 1],
        "interaction": y_val
    })

    # Filter interaksi positif
    val_pos_df = val_df[val_df["interaction"] > 0]

    # Mapping user_encoded -> set buku_encoded
    user_items_val = val_pos_df.groupby("user_encoded")["buku_encoded"].apply(set).to_dict()

    # Ambil 10 user secara acak
    users_to_eval = list(user_items_val.keys())
    if len(users_to_eval) > 10:
        users_to_eval = random.sample(users_to_eval, 10)

    results = []

    for user_encoded in users_to_eval:
        user_id_asli = next((k_ for k_, v_ in user_to_user_encoded.items() if v_ == user_encoded), None)
        if user_id_asli is None:
            continue

        actual_items = user_items_val[user_encoded]

        rekomendasi_df = recommend_books_for_user(
            model=model,
            user_id_asli=user_id_asli,
            data=dataFinal,
            buku_df=buku_df,
            user_to_user_encoded=user_to_user_encoded,
            buku_to_ids_encoded=buku_to_ids_encoded,
            ids_to_buku_encoded=ids_to_buku_encoded,
            top_n=k
        )

        rec_books_encoded = [
            buku_to_ids_encoded[isbn] for isbn in rekomendasi_df['ISBN'] if isbn in buku_to_ids_encoded
        ] if not rekomendasi_df.empty else []

        relevance_vector = [1 if book in actual_items else 0 for book in rec_books_encoded]
        true_positives = sum(relevance_vector)

        precision = true_positives / k if k > 0 else 0
        recall = true_positives / len(actual_items) if actual_items else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        hit_rate = 1 if true_positives > 0 else 0

        results.append({
            "user_encoded": user_encoded,
            "user_id_asli": user_id_asli,
            "recommended_encoded": rec_books_encoded,
            "actual_encoded": list(actual_items),
            "relevance_vector": relevance_vector,
            "precision@k": precision,
            "recall@k": recall,
            "f1_score@k": f1_score,
            "hit_rate@k": hit_rate
        })

    results_df = pd.DataFrame(results)
    avg_precision = results_df["precision@k"].mean()
    avg_recall = results_df["recall@k"].mean()
    avg_f1 = results_df["f1_score@k"].mean()
    avg_hit_rate = results_df["hit_rate@k"].mean()

    # Hitung RMSE pada seluruh validation set
    y_pred = model.predict(x_val).flatten()
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    return results_df, avg_precision, avg_recall, avg_f1, avg_hit_rate, rmse


# In[28]:


results_df, avg_prec, avg_rec, avg_f1, avg_hr, rmse = evaluate_cf_model(
    model, x_val, y_val, dataFinal, dataBuku,
    user_to_user_encoded, buku_to_ids_encoded, ids_to_buku_encoded,
    k=5
)

print(f"Avg Precision@5: {avg_prec:.4f}")
print(f"Avg Recall@5: {avg_rec:.4f}")
print(f"Avg F1-Score@5: {avg_f1:.4f}")
print(f"Avg Hit Rate@5: {avg_hr:.4f}")
print(f"RMSE: {rmse:.4f}")


# ### Sistem Rekomendasi CatBoost

# In[31]:


def evaluate_cb(
    model, 
    x_val, 
    y_val, 
    dataFinal, 
    buku_df,
    user_to_user_encoded, 
    buku_to_ids_encoded, 
    ids_to_buku_encoded, 
    k=5
):
    val_df = pd.DataFrame({
        "user_encoded": x_val[:, 0],
        "buku_encoded": x_val[:, 1],
        "interaction": y_val
    })

    val_pos_df = val_df[val_df["interaction"] > 0]
    user_items_val = val_pos_df.groupby("user_encoded")["buku_encoded"].apply(set).to_dict()

    users_to_eval = list(user_items_val.keys())
    if len(users_to_eval) > 10:
        users_to_eval = random.sample(users_to_eval, 10)

    results = []

    for user_encoded in users_to_eval:
        user_id_asli = next((k_ for k_, v_ in user_to_user_encoded.items() if v_ == user_encoded), None)
        if user_id_asli is None:
            continue

        actual_items = user_items_val[user_encoded]

        rekomendasi_df = recommend_books_cb(
            model=model,
            user_id_asli=user_id_asli,
            data=dataFinal,
            buku_df=buku_df,
            user_to_user_encoded=user_to_user_encoded,
            buku_to_ids_encoded=buku_to_ids_encoded,
            ids_to_buku_encoded=ids_to_buku_encoded,
            top_n=k
        )

        rec_books_encoded = [
            buku_to_ids_encoded[isbn] for isbn in rekomendasi_df['ISBN'] if isbn in buku_to_ids_encoded
        ] if not rekomendasi_df.empty else []

        relevance_vector = [1 if book in actual_items else 0 for book in rec_books_encoded]
        true_positives = sum(relevance_vector)

        precision = true_positives / k if k > 0 else 0
        recall = true_positives / len(actual_items) if actual_items else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        hit_rate = 1 if true_positives > 0 else 0

        results.append({
            "user_encoded": user_encoded,
            "user_id_asli": user_id_asli,
            "recommended_encoded": rec_books_encoded,
            "actual_encoded": list(actual_items),
            "relevance_vector": relevance_vector,
            "precision@k": precision,
            "recall@k": recall,
            "f1_score@k": f1_score,
            "hit_rate@k": hit_rate
        })

    results_df = pd.DataFrame(results)
    avg_precision = results_df["precision@k"].mean()
    avg_recall = results_df["recall@k"].mean()
    avg_f1 = results_df["f1_score@k"].mean()
    avg_hit_rate = results_df["hit_rate@k"].mean()

    y_pred = model.predict(x_val).flatten()
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    return results_df, avg_precision, avg_recall, avg_f1, avg_hit_rate, rmse


# In[32]:


results_df, avg_prec, avg_rec, avg_f1, avg_hit, rmse = evaluate_cb(
    model_cb,
    x_val,
    y_val,
    dataFinal,
    dataBuku,
    user_to_user_encoded,
    buku_to_ids_encoded,
    ids_to_buku_encoded,
    k=5
)

print(f"Precision@5: {avg_prec:.4f}")
print(f"Recall@5: {avg_rec:.4f}")
print(f"F1-score@5: {avg_f1:.4f}")
print(f"Hit Rate@5: {avg_hit:.4f}")
print(f"RMSE: {rmse:.4f}")


# **Penjelasan :**
# 
# Pada Collaborative Filtering menggunakan RecommenderNet dan Algoritma Catboost, model dievaluasi terhadap pengguna yang didefinisikan pada validation set, diperoleh evaluasi sebagai berikut.
# 
# =====Recommender Net====

# Avg Precision@5: 0.0000<br>
# Avg Recall@5: 0.0000<br>
# Avg F1-Score@5: 0.0000<br>
# Avg Hit Rate@5: 0.0000<br>
# RMSE: 0.3754

# =====CatBoost=====

# Avg Precision@5: 0.0000<br>
# Avg Recall@5: 0.0000<br>
# Avg F1-Score@5: 0.0000<br>
# Avg Hit Rate@5: 0.0000<br>
# RMSE: 0.3741
# 
# Berdasarkan kedua jenis sistem rekomendasi yang dibangun, terlihat model masih belum cukup baik untuk memberikan rekomendasi pada user, namun sudah memiliki keterkaitan dengan konsep yang digunakan. Hal ini dapat dilihat pada Content Based Filtering, hasil return untuk input user pada judul atau author buku tertentu sesuai dengan hasil yang diharapkan. Kondisi yang membuat evaluasi kecil dapat terjadi karena terdapat kecenderungan data user yang membeli buku secara acak, bukan berdasarkan kesamaan karakteristiknya.
# 
# Selain itu pada Collaborative filtering diperoleh bahwa CatBoost memiliki performa sistem rekomendasi yang lebih baik dibandingkan dengan sistem rekomendasi yang dihasilkan RecommenderNet. Hal ini menunjukkan bahwa Machine Learning tidak selalu lebih buruk performanya dibanding Deep Learning pada dataset yang besar.
