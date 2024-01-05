import pandas as pd
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neural_network import BernoulliRBM
import matplotlib.pyplot as plt
import numpy as np


#Treinando rede neural classificadora para avaliar performance dos seguintes metodos de Redução de Dimensionalidade
# - Autoencoders
# - PCA
# - RBM (Boltzman Machine)

# - Conclusão, Apesar de ser o mais dificil de implementar Autoincoder se mostrou superior para redimensionamento em caso de imagens

# - PCA apenas com dados lineares
# - RBM alto custo computacional

epochs = 100
batch_size = 8

def CarregarDados():
    #Carregando dados
    (previsores_treino, classe_treino), (previsores_teste, classe_teste) = mnist.load_data()

    #Categorizando classe
    classe_treino = to_categorical(classe_treino, 10)
    classe_teste = to_categorical(classe_teste, 10)

    #Normalizando previsores
    previsores_treino = previsores_treino.astype('float32') / 255
    previsores_teste = previsores_teste.astype('float32') / 255

    #Transformando matriz de dados em vetor
    previsores_treino = previsores_treino.reshape((len(previsores_treino), np.prod(previsores_treino.shape[1:])))
    previsores_teste = previsores_teste.reshape((len(previsores_teste), np.prod(previsores_teste.shape[1:])))

    return (previsores_treino, classe_treino), (previsores_teste, classe_teste)

def CriaRede():
    #Rede Neural Classificadora
    #Camada de Pre-processamento
    modelo = Sequential()

    #Camadas densas
    modelo.add(Dense(units=250, activation='relu', input_dim=128))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(units=250, activation='relu'))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(units=250, activation='relu'))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(units=10, activation='softmax'))

    modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return modelo

def AutoEncoder_():
    #Carregando dados
    (previsores_treino, classe_treino), (previsores_teste, classe_teste) = CarregarDados()

    #Limitando quantidade de dados para agilizar processamento
    previsores_treino = previsores_treino[0:3000]
    previsores_teste = previsores_teste[0:1000]
    classe_treino = classe_treino[0:3000]
    classe_teste = classe_teste[0:1000]

    #Definindo callbacks
    es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, min_delta= 1e-10)
    rlp = ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1)
    md_encoder = ModelCheckpoint(filepath='Encoder.0.1', save_best_only=True, verbose=1)
    md = ModelCheckpoint(filepath='Modelo.0.1', save_best_only=True, verbose=1)

    #Carregando modelo de Autoencoder caso existente ou treinando novo caso inexistente
    try:
        encoder = load_model('Encoder.0.1')
    except:
        encoder = Sequential()
        encoder.add(Dense(units=512, activation='relu', input_dim=784))
        encoder.add(Dense(units=256, activation='relu'))
        encoder.add(Dense(units=128, activation='relu'))
        encoder.add(Dense(units=256, activation='relu'))
        encoder.add(Dense(units=512, activation='relu'))
        encoder.add(Dense(units=784, activation='sigmoid'))

        encoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        encoder.fit(previsores_treino, previsores_treino, batch_size=batch_size, epochs=epochs, callbacks=[es, rlp, md_encoder], validation_data=(previsores_teste, previsores_teste))

    #Definindo encoder a partir do modelo de encode/decode
    dimensao_original = Input(shape=(784,))
    camada_encoder1 = encoder.layers[0]
    camada_encoder2 = encoder.layers[1]
    camada_encoder3 = encoder.layers[2]

    encode = Model(dimensao_original, camada_encoder3(camada_encoder2(camada_encoder1(dimensao_original))))

    modelo = CriaRede()

    #Aplicando redução de dimensionalidade
    previsores_treino = encode.predict(previsores_treino)
    previsores_teste = encode.predict(previsores_teste)

    modelo.fit(previsores_treino, classe_treino, batch_size=batch_size, epochs=epochs, validation_data=(previsores_teste,classe_teste), callbacks=[es, rlp, md])

    #Coletando dados de acuracia
    predicao = modelo.predict(previsores_teste)
    predicao = np.argmax(predicao, axis=1)
    classe_teste = np.argmax(classe_teste, axis=1)

    accuracy = accuracy_score(predicao, classe_teste)

    return accuracy

def PCA_():
    #Carregando dados
    (previsores_treino, classe_treino), (previsores_teste, classe_teste) = CarregarDados()

    #Limitando quantidade de dados para agilizar processamento
    previsores_treino = previsores_treino[0:3000]
    previsores_teste = previsores_teste[0:1000]
    classe_treino = classe_treino[0:3000]
    classe_teste = classe_teste[0:1000]

    modelo = CriaRede()

    #Aplicando redução de dimensionalidade através do PCA
    pca = PCA(n_components=128)
    previsores_treino = pca.fit_transform(previsores_treino)
    previsores_teste = pca.fit_transform(previsores_teste)

    #Definindo callbacks
    es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, min_delta= 1e-10)
    rlp = ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1)
    md = ModelCheckpoint(filepath='Modelo.0.1', save_best_only=True, verbose=1)

    modelo.fit(previsores_treino, classe_treino, batch_size=batch_size, epochs=epochs, validation_data=(previsores_teste,classe_teste), callbacks=[es, rlp, md])

    #Coletando dados de acuracia
    predicao = modelo.predict(previsores_teste)
    predicao = np.argmax(predicao, axis=1)
    classe_teste = np.argmax(classe_teste, axis=1)

    accuracy = accuracy_score(predicao, classe_teste)

    return accuracy

def RBM_():
    #Carregando dados
    (previsores_x, classe_treino), (previsores_y, classe_teste) = CarregarDados()

    #Limitando quantidade de dados para agilizar processamento
    previsores_x = previsores_x[0:3000]
    previsores_y = previsores_y[0:1000]
    classe_treino = classe_treino[0:3000]
    classe_teste = classe_teste[0:1000]

    modelo = CriaRede()

    #Aplicando redução de dimensionalidade através de um Loop For para melhor acompanhamento
    rbm = BernoulliRBM(n_components=128, n_iter=500)

    previsores_treino = np.empty((0,128))
    i = 1
    for x in previsores_x:
        print(i)
        x = (np.expand_dims(x, axis=0))
        previsores_treino = np.vstack((previsores_treino, rbm.fit_transform(x)))
        i = i+1

    i = 1
    previsores_teste = np.empty((0,128))
    for x in previsores_y:
        print(i)
        x = (np.expand_dims(x, axis=0))
        previsores_teste = np.vstack((previsores_teste, rbm.fit_transform(x)))
        i = i+1



    #Definindo callbacks
    es = EarlyStopping(monitor='val_loss', patience=20, verbose=1, min_delta= 1e-10)
    rlp = ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1)
    md = ModelCheckpoint(filepath='Modelo.0.1', save_best_only=True, verbose=1)

    modelo.fit(previsores_treino, classe_treino, batch_size=batch_size, epochs=epochs, validation_data=(previsores_teste, classe_teste), callbacks=[es, rlp, md])

    #Coletando dados de acuracia
    predicao = modelo.predict(previsores_teste)
    predicao = np.argmax(predicao, axis=1)
    classe_teste = np.argmax(classe_teste, axis=1)

    accuracy = accuracy_score(predicao, classe_teste)

    return accuracy

accuracy_rbm = RBM_()
accuracy_pca = PCA_()
accuracy_encoder = AutoEncoder_()

height = [accuracy_rbm, accuracy_pca, accuracy_encoder]
bars = ('RBM', 'PCA', 'AutoEncoder')

df_accuracy = pd.DataFrame({'height' : height,
                            'bars' : bars})

plt.bar(df_accuracy['bars'], df_accuracy['height'], color=['blue', 'purple', 'red'])
plt.show()







