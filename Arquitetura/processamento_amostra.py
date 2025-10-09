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


linkdb = "D:/cox-models-sudden-death/01_Dataset/Duckedb/Holter_ECG/1min/banco_ecg.duckdb"
tabela = 'ecg_pacientes'


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
    
    # def wavelet_filter_preservando_qrs(self, signal, wavelet='db6', level=4, atenuacao=0.2):
    #     coeffs = pywt.wavedec(signal, wavelet, level=level)

    #     # Atenua apenas o coeficiente de aproximação (baixa frequência)
    #     # O coeficiente cA está sempre no primeiro índice (coeffs[0])
    #     coeffs[0] *= atenuacao 

    #     # Opcional: você também pode atenuar os detalhes de frequência mais baixa,
    #     # como cD4, cD3 (índices 1 e 2 se level=4), se desejar.
    #     # Exemplo: for i in range(1, 3): coeffs[i] *= (atenuacao + 0.2)

    #     ecg_wavelet_filtered = pywt.waverec(coeffs, wavelet)
    #     return ecg_wavelet_filtered

    def process(self, signal, sampling_rate=200, metodo='elgendi2010'):
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
    """
    Classe para carregar e processar dados de ECG para análise de sobrevivência.
    
    Args:
        usar_limpeza: Se True, aplica filtros de limpeza no ECG
        limite_features: Número máximo de amostras por paciente (None = todas)
        limite_labels: Número máximo de pacientes (None = todos)
        metodo: Método de limpeza NeuroKit ('elgendi2010', 'pantompkins', etc.)
        batch_size: Tamanho do batch para DataLoader
        wavelet_level: Nível de decomposição wavelet
    """
    
    def __init__(self, usar_limpeza=False, limite_features=None, limite_labels=None, 
                 metodo='elgendi2010', batch_size=128):
        
        self.limite_features = limite_features
        self.limite_labels = limite_labels
        self.metodo = metodo
        self.batch_size = batch_size
        self.usar_limpeza = usar_limpeza
        
        
        # Atributos inicializados após processamento
        self.labtrans = None
        self.amostra_sinal = None
        self.tempo_teste = None
        self.evento_teste = None
    
    def carregamento(self):
        """
        Carrega dados do DuckDB e processa sinais ECG.
        
        Returns:
            Tupla (atributo_x, atributo_y, atributo_z, tempos, eventos)
        """
        conn_leitura = duckdb.connect(linkdb, read_only=True)
        
        atributo_x, atributo_y, atributo_z = [], [], []
        tempos, eventos = [], []
        
        # Obter lista de pacientes do banco ECG
        if self.limite_labels is None:
            query_pacientes = f"SELECT DISTINCT id_paciente FROM {tabela} ORDER BY id_paciente;"
        else:
            query_pacientes = f"SELECT DISTINCT id_paciente FROM {tabela} ORDER BY id_paciente LIMIT {self.limite_labels};"
        
        lista_ecg = conn_leitura.execute(query_pacientes).fetchdf()
        lista_labels = dados['id_paciente'].unique()
        
        # Interseção: pacientes com ECG E labels
        pacientes_validos = lista_ecg[lista_ecg['id_paciente'].isin(lista_labels)]
        
        if len(pacientes_validos) == 0:
            raise ValueError("Nenhum paciente válido encontrado (sem interseção entre ECG e labels)")
        
        logger.info(f"Processando {len(pacientes_validos)} pacientes válidos")
        
        # Processamento dos pacientes
        processor = ECGProcessor() if self.usar_limpeza else None
        
        for paciente in tqdm(pacientes_validos['id_paciente'], desc="Processando pacientes"):
            # Query para pegar sinais do paciente
            if self.limite_features is None:
                query_sinais = f"SELECT * FROM {tabela} WHERE id_paciente = '{paciente}';"
            else:
                query_sinais = f"SELECT * FROM {tabela} WHERE id_paciente = '{paciente}' LIMIT {self.limite_features};"
            
            sinais = conn_leitura.execute(query_sinais).fetchdf()
            
            if len(sinais) == 0:
                logger.warning(f"Paciente {paciente} sem sinais no banco - pulando")
                continue
            
            # Extrair sinais
            ecg_x = sinais['sinal_x'].values
            ecg_y = sinais['sinal_y'].values
            ecg_z = sinais['sinal_z'].values
            
            # Aplicar limpeza se solicitado
            if self.usar_limpeza:
                amostra_x = processor.process(ecg_x, sampling_rate=200, metodo=self.metodo)
                amostra_y = processor.process(ecg_y, sampling_rate=200, metodo=self.metodo)
                amostra_z = processor.process(ecg_z, sampling_rate=200, metodo=self.metodo)
            else:
                amostra_x, amostra_y, amostra_z = ecg_x, ecg_y, ecg_z
            
            # Adicionar às listas
            atributo_x.append(amostra_x)
            atributo_y.append(amostra_y)
            atributo_z.append(amostra_z)
            
            # Obter tempo e evento
            paciente_data = dados[dados['id_paciente'] == paciente]
            if len(paciente_data) == 0:
                logger.warning(f"Paciente {paciente} sem labels - pulando")
                continue
                
            tempo = paciente_data['tempo'].iloc[0]
            evento = paciente_data['evento'].iloc[0]
            
            tempos.append(tempo)
            eventos.append(evento)
        
        conn_leitura.close()
        
        logger.info(f"Processamento concluído: {len(tempos)} pacientes carregados")
        logger.info(f"Parâmetros: limite_features={self.limite_features}, "
                   f"limite_labels={self.limite_labels}, metodo={self.metodo}")
        
        return atributo_x, atributo_y, atributo_z, tempos, eventos
    
    def convert_to_tensors(self):
        """
        Converte dados carregados em tensores PyTorch e cria DataLoaders.
        
        Returns:
            Tupla (dl_treino, dl_teste) com DataLoaders
        """
        atributo_x, atributo_y, atributo_z, tempos, eventos = self.carregamento()
        
        # Verificar comprimentos consistentes
        comprimentos = [len(arr) for arr in atributo_x]
        if len(set(comprimentos)) > 1:
            logger.warning(f"Sinais com comprimentos diferentes detectados: {set(comprimentos)}")
            # Padronizar para o menor comprimento
            min_len = min(comprimentos)
            atributo_x = [arr[:min_len] for arr in atributo_x]
            atributo_y = [arr[:min_len] for arr in atributo_y]
            atributo_z = [arr[:min_len] for arr in atributo_z]
        
        # Converter para tensores PyTorch
        tensor_x = torch.tensor(np.array(atributo_x), dtype=torch.float32)
        tensor_y = torch.tensor(np.array(atributo_y), dtype=torch.float32)
        tensor_z = torch.tensor(np.array(atributo_z), dtype=torch.float32)
        
        # Formato: (batch, channels, timesteps)
        sinais_tensor = torch.stack([tensor_x, tensor_y, tensor_z], dim=1)
        
        # Salvar amostra para visualização
        self.amostra_sinal = sinais_tensor[:1].numpy()
        
        # Arrays numpy para labels
        tempos_array = np.array(tempos)
        eventos_array = np.array(eventos)
        
        # Split treino/teste
        indices = np.arange(len(sinais_tensor))
        idx_treino, idx_teste = train_test_split(indices, test_size=0.2, random_state=123)
        
        sinais_treino = sinais_tensor[idx_treino]
        sinais_teste = sinais_tensor[idx_teste]
        tempos_treino = tempos_array[idx_treino]
        tempos_teste = tempos_array[idx_teste]
        eventos_treino = eventos_array[idx_treino]
        eventos_teste = eventos_array[idx_teste]
        
        # Salvar dados de teste
        self.tempo_teste = tempos_teste
        self.evento_teste = eventos_teste
        
        # Transformação dos rótulos
        num_durations = 20
        self.labtrans = LogisticHazard.label_transform(num_durations)
        
        alvo_treino = self.labtrans.fit_transform(tempos_treino, eventos_treino)
        alvo_teste = self.labtrans.transform(tempos_teste, eventos_teste)
        
        # Criar datasets e dataloaders
        dataset_treino = ECGDatasetBatch(sinais_treino, *alvo_treino)
        dataset_teste = ECGDatasetBatch(sinais_teste, *alvo_teste)
        
        dl_treino = tt.data.DataLoaderBatch(dataset_treino, self.batch_size, shuffle=True)
        dl_teste = tt.data.DataLoaderBatch(dataset_teste, self.batch_size, shuffle=False)
        
        logger.info(f"DataLoaders criados - Treino: {len(dataset_treino)}, Teste: {len(dataset_teste)}")
        
        return dl_treino, dl_teste
    
    # Getters
    def get_labtrans(self):
        if self.labtrans is None:
            raise ValueError("labtrans não inicializado. Execute convert_to_tensors() primeiro.")
        return self.labtrans
    
    def get_amostra_sinal(self):
        if self.amostra_sinal is None:
            raise ValueError("amostra_sinal não inicializado. Execute convert_to_tensors() primeiro.")
        return self.amostra_sinal
    
    def get_evento_tempo(self):
        if self.tempo_teste is None or self.evento_teste is None:
            raise ValueError("Dados de teste não inicializados. Execute convert_to_tensors() primeiro.")
        return self.tempo_teste, self.evento_teste