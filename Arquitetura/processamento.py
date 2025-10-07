# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
import torch
import duckdb
from logger_config import setup_logger

from pycox.models import LogisticHazard
from sklearn.model_selection import train_test_split

logger = setup_logger(r"D:\cox-models-sudden-death\Arquitetura\logs","processamento_dados.log", name="Processamento")

link_csv = r'D:\cox-models-sudden-death\01_Dataset\dados_csv_info_definitions\subject-info_formatado.csv'

df = pd.read_csv(link_csv, sep=';', usecols=['Patient ID','Follow-up period from enrollment (days)','Cause of death','Gender (male=1)','Age'])

df.rename(columns={'Patient ID':'id_paciente',
    'Follow-up period from enrollment (days)':'tempo',
                   'Cause of death':'evento',
                   'Gender (male=1)':'genero',
                   'Age':'idade'}, inplace=True)

# 0 sobreviventes, censurados 1 (6 e 7 arritmias)
dados = df.query('(evento == 0) or (evento == 6) or (evento == 7)')
dados = dados.reset_index(drop=True).sort_values("tempo")
dados['evento'] = dados['evento'].replace(6, 1).replace(7, 1)


link = "D:/cox-models-sudden-death/01_Dataset/Duckedb/Holter_ECG/1min/banco_ecg_limpo.duckdb"

tabela = 'ecg_pacientes_limpo'

class PassandoDados:
    """
    Classe para processar dados de ECG usando gerador (lazy loading).
    
    Args:
        limite_features (int, optional): Quantidade de amostras por paciente
        limite_labels (int, optional): Quantidade de pacientes
        linkdb (str): Caminho para o banco DuckDB
        batch_size (int): Tamanho do batch para os geradores
    """
    def __init__(self, limite_features=None, limite_labels=None, linkdb=link, batch_size=32):
        
        self.linkdb = linkdb
        self.limite_features = limite_features
        self.limite_labels = limite_labels
        self.batch_size = batch_size
        self.labtrans = None
        self.amostra_sinal = None
        self.pacientes_treino = None
        self.pacientes_teste = None
        
    
    def _get_lista_pacientes(self, conn):
        """Obtém a lista de pacientes filtrada"""
        if self.limite_labels == None:
            lista1 = conn.execute(f"""
                SELECT DISTINCT id_paciente 
                FROM {tabela} 
                ORDER BY id_paciente;
            """).fetchdf()
        else:
            lista1 = conn.execute(f"""
                SELECT DISTINCT id_paciente 
                FROM {tabela} 
                ORDER BY id_paciente
                LIMIT {self.limite_labels};
            """).fetchdf()
            
        lista2 = dados['id_paciente'].unique()
        lista = lista1[lista1['id_paciente'].isin(lista2)]
        return lista['id_paciente'].values
    
    
    def _processar_paciente(self, conn, paciente):
        """Processa um único paciente e retorna os sinais e labels"""
        if self.limite_features == None:
            sinais = conn.execute(f"""
                SELECT * FROM {tabela}
                WHERE id_paciente = '{paciente}';
            """).fetchdf()
        else:
            sinais = conn.execute(f"""
                SELECT * FROM {tabela}
                WHERE id_paciente = '{paciente}'
                LIMIT {self.limite_features};
            """).fetchdf()

        ecg_x = sinais['sinal_x'].values
        ecg_y = sinais['sinal_y'].values
        ecg_z = sinais['sinal_z'].values
        
        tempo = dados[dados['id_paciente'] == f'{paciente}']['tempo'].iloc[0]
        evento = dados[dados['id_paciente'] == f'{paciente}']['evento'].iloc[0]
        
        return ecg_x, ecg_y, ecg_z, tempo, evento


    def ecg_generator(self, lista_pacientes, batch_size=None):
        """
        Gerador que carrega dados sob demanda do DuckDB.
        OTIMIZADO: Faz 1 query por batch ao invés de N queries.
        
        Args:
            lista_pacientes: Lista de IDs dos pacientes
            batch_size: Tamanho do batch (usa self.batch_size se None)
        
        Yields:
            sinais_batch: Tensor (batch_size, 3, timesteps)
            tempos_batch: Array (batch_size,)
            eventos_batch: Array (batch_size,)
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        num_pacientes = len(lista_pacientes)
        conn = duckdb.connect(self.linkdb, read_only=True)
        
        while True:  # Loop infinito para múltiplas épocas
            for offset in range(0, num_pacientes, batch_size):
                batch_pacientes = lista_pacientes[offset:offset + batch_size]
                
                # ✅ OTIMIZAÇÃO: 1 query para todos os pacientes do batch
                pacientes_str = "','".join(batch_pacientes)
                
                if self.limite_features is None:
                    query = f"""
                        SELECT id_paciente, sinal_x, sinal_y, sinal_z
                        FROM {tabela}
                        WHERE id_paciente IN ('{pacientes_str}')
                        ORDER BY id_paciente;
                    """
                else:
                    # Para limite de features, precisamos usar window function
                    query = f"""
                        WITH ranked AS (
                            SELECT 
                                id_paciente, 
                                sinal_x, 
                                sinal_y, 
                                sinal_z,
                                ROW_NUMBER() OVER (PARTITION BY id_paciente ORDER BY rowid) as rn
                            FROM {tabela}
                            WHERE id_paciente IN ('{pacientes_str}')
                        )
                        SELECT id_paciente, sinal_x, sinal_y, sinal_z
                        FROM ranked
                        WHERE rn <= {self.limite_features}
                        ORDER BY id_paciente;
                    """
                
                try:
                    sinais_df = conn.execute(query).fetchdf()
                    
                    sinais_x_batch = []
                    sinais_y_batch = []
                    sinais_z_batch = []
                    tempos_batch = []
                    eventos_batch = []
                    
                    # Processar cada paciente do batch
                    for paciente in batch_pacientes:
                        try:
                            # Filtrar sinais deste paciente
                            paciente_data = sinais_df[sinais_df['id_paciente'] == paciente]
                            
                            if len(paciente_data) == 0:
                                logger.warning(f"Paciente {paciente} não tem dados")
                                continue
                            
                            ecg_x = paciente_data['sinal_x'].values
                            ecg_y = paciente_data['sinal_y'].values
                            ecg_z = paciente_data['sinal_z'].values
                            
                            # Buscar tempo e evento
                            tempo = dados[dados['id_paciente'] == paciente]['tempo'].iloc[0]
                            evento = dados[dados['id_paciente'] == paciente]['evento'].iloc[0]
                            
                            sinais_x_batch.append(ecg_x)
                            sinais_y_batch.append(ecg_y)
                            sinais_z_batch.append(ecg_z)
                            tempos_batch.append(tempo)
                            eventos_batch.append(evento)
                            
                        except Exception as e:
                            logger.error(f"Erro ao processar paciente {paciente}: {e}")
                            continue
                    
                    # Converter para tensores
                    if len(sinais_x_batch) > 0:
                        tensor_x = torch.tensor(np.array(sinais_x_batch), dtype=torch.float32)
                        tensor_y = torch.tensor(np.array(sinais_y_batch), dtype=torch.float32)
                        tensor_z = torch.tensor(np.array(sinais_z_batch), dtype=torch.float32)
                        
                        sinais_tensor = torch.stack([tensor_x, tensor_y, tensor_z], dim=1)
                        tempos_array = np.array(tempos_batch)
                        eventos_array = np.array(eventos_batch)
                        
                        yield sinais_tensor, tempos_array, eventos_array
                    else:
                        logger.warning(f"Batch vazio no offset {offset}")
                        
                except Exception as e:
                    logger.error(f"Erro ao processar batch no offset {offset}: {e}")
                    continue
    

    def preparar_dados(self):
        """
        Prepara os dados para treinamento.
        Retorna geradores e informações necessárias.
        """
        conn = duckdb.connect(self.linkdb, read_only=True)
        lista_pacientes = self._get_lista_pacientes(conn)
        conn.close()
        
        print(f"📊 Total de pacientes encontrados: {len(lista_pacientes)}")
        
        # Split treino/teste (80/20)
        idx_treino, idx_teste = train_test_split(
            np.arange(len(lista_pacientes)), 
            test_size=0.2, 
            random_state=123
        )
        
        self.pacientes_treino = lista_pacientes[idx_treino]
        self.pacientes_teste = lista_pacientes[idx_teste]
        
        print(f"   - Treino: {len(self.pacientes_treino)} pacientes")
        print(f"   - Teste: {len(self.pacientes_teste)} pacientes")
        
        # Configurar labtrans (precisa de dados para fit)
        num_durations = 20
        self.labtrans = LogisticHazard.label_transform(num_durations)
        
        # Carregar um batch pequeno para configurar labtrans e pegar amostra
        print("🔄 Configurando labtrans...")
        gen_temp = self.ecg_generator(self.pacientes_treino, batch_size=min(16, len(self.pacientes_treino)))
        sinais_temp, tempos_temp, eventos_temp = next(gen_temp)
        
        self.amostra_sinal = sinais_temp[:1].numpy()
        self.labtrans.fit_transform(tempos_temp, eventos_temp)
        
        print(f"✅ Labtrans configurado com {self.labtrans.out_features} durações")
        
        # Calcular steps por época
        steps_per_epoch_train = max(1, len(self.pacientes_treino) // self.batch_size)
        steps_per_epoch_test = max(1, len(self.pacientes_teste) // self.batch_size)
        
        # Adicionar step extra se houver resto
        if len(self.pacientes_treino) % self.batch_size != 0:
            steps_per_epoch_train += 1
        if len(self.pacientes_teste) % self.batch_size != 0:
            steps_per_epoch_test += 1
        
        return {
            'train_generator': self.ecg_generator(self.pacientes_treino),
            'test_generator': self.ecg_generator(self.pacientes_teste),
            'steps_per_epoch_train': steps_per_epoch_train,
            'steps_per_epoch_test': steps_per_epoch_test,
            'num_train_samples': len(self.pacientes_treino),
            'num_test_samples': len(self.pacientes_teste)
        }
    
    
    def get_labtrans(self):
        """Retorna o label transformer"""
        return self.labtrans
        
    def get_amostra_sinal(self):
        """Retorna uma amostra de sinal para testes"""
        return self.amostra_sinal