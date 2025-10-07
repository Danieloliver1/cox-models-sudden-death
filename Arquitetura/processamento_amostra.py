# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
import torch
import duckdb
import torch
from logger_config import setup_logger # biblioteca local

from pycox.models import LogisticHazard
from sklearn.model_selection import train_test_split
import torchtuples as tt
from torch.utils.data import Dataset


logger = setup_logger(r"D:\cox-models-sudden-death\Arquitetura\logs","processamento_dados.log", name="Processamento")


link_csv = r'D:\cox-models-sudden-death\01_Dataset\dados_csv_info_definitions\subject-info_formatado.csv'


df = pd.read_csv(link_csv, sep=';', usecols=['Patient ID','Follow-up period from enrollment (days)','Cause of death','Gender (male=1)','Age'])
df.rename(columns={'Patient ID':'id_paciente',
    'Follow-up period from enrollment (days)':'tempo',
                   'Cause of death':'evento',
                   'Gender (male=1)':'genero',
                   'Age':'idade'}, inplace=True)


dados = df.query('(evento == 0) or (evento == 6) or (evento == 7)') # 0 sobreviventes, censurados 1 (6 e 7 arritmias)
dados = dados.reset_index(drop=True).sort_values("tempo")           # lipando os indices e ordenando
dados['evento'] = dados['evento'].replace(6, 1).replace(7, 1)       # passando evento tipo 6 para 1 e 7 para 1 (considerando como evento)



import neurokit2 as nk
import pywt  # Biblioteca para transformada wavelet


linkdb = "D:/cox-models-sudden-death/01_Dataset/Duckedb/Holter_ECG/1min/banco_ecg_limpo.duckdb"
tabela = 'ecg_pacientes_limpo'
usar_limpeza = False

# ==========================================limpeza do ECG==============================================
class ECGProcessor:
    def __init__(self):
        pass

    
    def wavelet_filter_preservando_qrs(self, signal, wavelet='db6', level=4, atenuacao=0.2):
        coeffs = pywt.wavedec(signal, wavelet, level=level) # menos agressivo

        # Mantém detalhes de nível 1 e 2 (alta frequência = QRS)
        for i in range(3, len(coeffs)):  # Atenua detalhes de nível mais alto
            coeffs[i] *= atenuacao # mantém 20% da energia

        ecg_wavelet_filtered = pywt.waverec(coeffs, wavelet)
        return ecg_wavelet_filtered
    

    def process(self, signal, sampling_rate=200, metodo='neurokit'):
        """
        Função principal de processamento.
        Limpa o sinal, calcula as métricas HRV, QRS e QTc, e retorna um DataFrame com os resultados.
        """
        #self.signal = signal
        self.fs = sampling_rate
        

        #teste2
        # Por exemplo: preserve cD1 e cD2
        # zera só os detalhes de níveis mais altos
        ecg_wavelet_filtered = self.wavelet_filter_preservando_qrs(signal, wavelet='db6', level=2, atenuacao=0.2)


        # Limpeza do sinal com o método de Pan-Tompkins
        ecg_limpo = nk.ecg_clean(ecg_wavelet_filtered, sampling_rate=200, method=metodo)

        # Ajuste de escala (opcional): desloca o sinal para cima, garantindo valores positivos
        # self.ecg_limpo = ecg_limpo + 2
        
    
        return ecg_limpo


# DATASET E DATALOADER 
class ECGDatasetBatch(Dataset):
    def __init__(self, sinais, durations, events):
        self.durations, self.events = tt.tuplefy(durations, events).to_tensor()
        self.sinais = sinais
    def __len__(self):
        return len(self.durations)
    def __getitem__(self, index):
        if not hasattr(index, '__iter__'): index = [index]
        batch_sinais = self.sinais[index]
        return tt.tuplefy(batch_sinais, (self.durations[index], self.events[index]))
    
    
# =================================================preprocessamento dos dados==================================

class PassandoDadosAmostra:
    """_summary_

        Args:
            limite_features (int, optional): _description_. Defaults to 12000.
            limite_labels (int, optional): _description_. Defaults to None.
            metodo (str, optional): _description_. Defaults to 'elgendi2010'.
        """
    def __init__(self, limite_features=12000, limite_labels=None, metodo='elgendi2010'):
        
        # Por padrão vou definir sinais de 1 minutos (12000 amostras a 200 Hz) 
        self.limite_features  = limite_features  #  quantidade de dados
        self.limite_labels  = limite_labels # quantidade de pacientes
        self.metodo = metodo
        self.batch_size = 128
        self.labtrans = None
        
        self.amostra_sinal = None
        
        self.evento_testeget = None
        self.tempo_testeget = None
        self.sinal_testeget = None
        
    
    def carregamento(self):
        
        # TESTE PARA FLUXO DO PROCESSAMENTO DO ECG COM 1 MINUTO (12000 AMOSTRAS A 200 Hz)


        # Conexões separadas
        conn_leitura = duckdb.connect(linkdb, read_only=True)
        #conn_escrita = duckdb.connect("D:/Projeto_Tese_mestrado/02_Dataset/Duckedb/Holter_ECG/1h/atributos_ecg.duckdb")

        # Listas para armazenar as amostras processadas
        atributo_x = []
        atributo_y = []
        atributo_z = []
        tempos = []
        eventos = []
    
        # Obtendo a lista de pacientes
        
        if self.limite_labels == None:
            lista1 = conn_leitura.execute(f"""
                SELECT DISTINCT id_paciente 
                FROM {tabela} 
                ORDER BY id_paciente;
            """).fetchdf()
        else:
            
            lista1 = conn_leitura.execute(f"""
                SELECT DISTINCT id_paciente 
                FROM {tabela} 
                ORDER BY id_paciente
                LIMIT {self.limite_labels};
            """).fetchdf()
            
        lista2 = dados['id_paciente'].unique()
        
        
        # Filtrando a lista de pacientes para incluir apenas aqueles que estão na lista de pacientes do ECG
        lista = lista1[lista1['id_paciente'].isin(lista2)]

        # Processamento dos dados dos pacientes
        for paciente in tqdm(lista['id_paciente'], desc="Processando pacientes", unit="paciente"):
            # Consulta para pegar os dados do ECG do paciente
            
            if self.limite_features == None:
                sinais = conn_leitura.execute(f"""
                    SELECT * FROM {tabela};
                """).fetchdf()   # Pegando todos os sinais disponíveis
            else:
                sinais = conn_leitura.execute(f"""
                    SELECT * FROM {tabela}
                    WHERE id_paciente = '{paciente}'
                    LIMIT {self.limite_features};
                """).fetchdf()   # Pegando sinais de 1 minutos (12000 amostras a 200 Hz) 

            # Extrair os sinais
            ecg_x = sinais['sinal_x'].values
            ecg_y = sinais['sinal_y'].values
            ecg_z = sinais['sinal_z'].values
            
            if usar_limpeza == True:
                # Criar o processador de ECG
                atributos = ECGProcessor()

                # Processar cada sinal
                amostra_x = atributos.process(ecg_x, sampling_rate=200, metodo=self.metodo)
                amostra_y = atributos.process(ecg_y, sampling_rate=200, metodo=self.metodo)
                amostra_z = atributos.process(ecg_z, sampling_rate=200, metodo=self.metodo)
            else:
                amostra_x = ecg_x
                amostra_y = ecg_y
                amostra_z = ecg_z
                

            # Adicionar os sinais processados às listas
            atributo_x.append(amostra_x)
            atributo_y.append(amostra_y)
            atributo_z.append(amostra_z)


            # Pegar o tempo e o evento para o paciente
            tempo = dados[dados['id_paciente'] == f'{paciente}']['tempo'].iloc[0]
            evento = dados[dados['id_paciente'] == f'{paciente}']['evento'].iloc[0]
            
            tempos.append(tempo)
            eventos.append(evento)


            # Log de processamento
            #logging.info(f"Processado paciente {paciente}, épocas processadas: 60, tempo: {tempo} dias, evento: {evento}")
        logger.info(f"Processado paciente {paciente}, épocas processadas: 60, tempo: {tempo} dias, evento: {evento}")
        print(f"Limite de features: {self.limite_features}, Limite de labels: {self.limite_labels}, Método: {self.metodo}")
            
        return atributo_x, atributo_y, atributo_z, tempos, eventos
    
    
#==================================Convertendo listas para tensores numpy===========================


    def convert_to_tensors(self):
        
        atributo_x, atributo_y, atributo_z, tempos, eventos = self.carregamento() 
        # Convertendo listas para arrays numpy
        atributo_x = np.array(atributo_x)
        atributo_y = np.array(atributo_y)
        atributo_z = np.array(atributo_z)
        #tempos = np.array(tempos)
        #eventos = np.array(eventos)
        
        # 1. Converter cada array para um tensor PyTorch sem modificações indesejadas
        tensor_x = torch.tensor(atributo_x, dtype=torch.float32)
        tensor_y = torch.tensor(atributo_y, dtype=torch.float32)
        tensor_z = torch.tensor(atributo_z, dtype=torch.float32)

        # OBS: Muitas camadas Conv1D do PyTorch preferem o formato (batch, channels, timesteps).
        # Para obter (channels, timesteps), você usaria dim=0:
        
        # 2. Empilhar os tensores para criar o formato (batch, channels, timesteps)
        # CORREÇÃO: Usar dim=1 para empilhar os canais corretamente.
        # O formato de entrada é [N, L], [N, L], [N, L]
        # O resultado será [N, 3, L], que é (batch, channels, timesteps)
        sinais_tensor = torch.stack([tensor_x, tensor_y, tensor_z], dim=1)
        
        
        
        self.amostra_sinal = sinais_tensor[:1].numpy()  # pega 1 amostra real
        
        # convertendo para array 
        tempos_array = np.array(tempos)
        eventos_array = np.array(eventos)
        
        
        indices = np.arange(len(sinais_tensor))
        idx_treino, idx_teste = train_test_split(indices, test_size=0.2, random_state=123)
        sinais_treino, sinais_teste = sinais_tensor[idx_treino], sinais_tensor[idx_teste]
        tempos_treino, tempos_teste = tempos_array[idx_treino], tempos_array[idx_teste]
        eventos_treino, eventos_teste = eventos_array[idx_treino], eventos_array[idx_teste]
        
        self.tempo_testeget = tempos_teste
        self.evento_testeget = eventos_teste
        #self.sinal_testeget = sinais_teste
        
        
        # 3. TRANSFORMAÇÃO DOS RÓTULOS (permanece igual)
        num_durations = 20
        
        self.labtrans = LogisticHazard.label_transform(num_durations)
        
        
        # Agora, use os arrays NumPy
        alvo_treino = self.labtrans.fit_transform(tempos_treino, eventos_treino)
        alvo_teste = self.labtrans.transform(tempos_teste, eventos_teste)
        
    
        dataset_treino = ECGDatasetBatch(sinais_treino, *alvo_treino)
        dataset_teste = ECGDatasetBatch(sinais_teste, *alvo_teste)
        
        # batch_size = 128
        dl_treino = tt.data.DataLoaderBatch(dataset_treino, self.batch_size, shuffle=True)
        dl_teste = tt.data.DataLoaderBatch(dataset_teste, self.batch_size, shuffle=False)

        # return sinais_tensor, tempos, eventos
        return dl_treino, dl_teste
    
   
    
    
    def get_labtrans(self):
            return self.labtrans
        
    def get_amostra_sinal(self):
        return self.amostra_sinal
    
    def get_evento_tempo(self):
        return self.tempo_testeget, self.evento_testeget
