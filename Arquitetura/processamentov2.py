# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
import torch
from pathlib import Path
from logger_config import setup_logger

from pycox.models import LogisticHazard
from sklearn.model_selection import train_test_split

logger = setup_logger(r"D:\cox-models-sudden-death\Arquitetura\logs","processamento_dados.log", name="Processamento")

# ============================================
# CAMINHOS CONFIGURÁVEIS
# ============================================
path_dados_parquet = r"D:\cox-models-sudden-death\01_Dataset\dados_ECG\dados_ECG_parquet"
path_filtro_pacientes_csv = r'D:\cox-models-sudden-death\02_Preprocessamento_filtro\resumo_dataset_ecg.csv'
path_info_clinica_csv = r'D:\cox-models-sudden-death\01_Dataset\dados_csv_info_definitions\subject-info_formatado.csv'

# ============================================
# CARREGAR DADOS CLÍNICOS
# ============================================
df = pd.read_csv(path_info_clinica_csv, sep=';', usecols=['Patient ID','Follow-up period from enrollment (days)','Cause of death','Gender (male=1)','Age'])

df.rename(columns={'Patient ID':'id_paciente',
    'Follow-up period from enrollment (days)':'tempo',
                   'Cause of death':'evento',
                   'Gender (male=1)':'genero',
                   'Age':'idade'}, inplace=True)

# 0 sobreviventes, censurados 1 (6 e 7 arritmias)
dados = df.query('(evento == 0) or (evento == 6) or (evento == 7)')
dados = dados.reset_index(drop=True).sort_values("tempo")
dados['evento'] = dados['evento'].replace(6, 1).replace(7, 1)

# ============================================
# CARREGAR FILTRO DE PACIENTES VÁLIDOS
# ============================================
df_filtro = pd.read_csv(path_filtro_pacientes_csv, sep=',')

# Filtrar apenas pacientes que TEM todos os sinais (X, Y e Z = "Sim")
df_lista_validos = df_filtro[(df_filtro['tem_X'] == 'Sim') & 
                              (df_filtro['tem_Y'] == 'Sim') & 
                              (df_filtro['tem_Z'] == 'Sim')]

pacientes_validos = set(df_lista_validos['paciente'].values)


class PassandoDados:
    """
    Classe para processar dados de ECG usando arquivos Parquet individuais por paciente.
    
    Args:
        data_folder_path (str): Caminho para a pasta com arquivos Parquet
        path_filtro_csv (str): Caminho para o CSV com filtro de pacientes válidos
        minutos_a_pular (int): Minutos para pular do início (padrão: 60 minutos = 1 hora)
        duracao_em_minutos (int): Minutos de dados para usar após pular o início
        batch_size (int): Tamanho do batch para os geradores
        limite_labels (int, optional): Quantidade máxima de pacientes
        frequencia_amostragem (int): Frequência de amostragem em Hz (padrão: 200Hz)
    """
    def __init__(self, 
                 data_folder_path=path_dados_parquet,
                 path_filtro_csv=path_filtro_pacientes_csv,
                 minutos_a_pular=60, 
                 duracao_em_minutos=20,
                 batch_size=32,
                 limite_labels=None,
                 frequencia_amostragem=200):
        
        self.data_folder_path = Path(data_folder_path)
        self.path_filtro_csv = path_filtro_csv
        self.minutos_a_pular = minutos_a_pular
        self.duracao_em_minutos = duracao_em_minutos
        self.batch_size = batch_size
        self.limite_labels = limite_labels
        self.frequencia_amostragem = frequencia_amostragem
        
        # Calcular amostras baseado na frequência
        self.amostras_por_minuto = frequencia_amostragem * 60
        self.idx_inicio = self.minutos_a_pular * self.amostras_por_minuto
        self.idx_fim = self.idx_inicio + (self.duracao_em_minutos * self.amostras_por_minuto)
        
        self.labtrans = None
        self.amostra_sinal = None
        self.pacientes_treino = None
        self.pacientes_validacao = None
        self.pacientes_teste = None
        
        logger.info(f"Configuração temporal: Pulando primeiros {minutos_a_pular} min, usando {duracao_em_minutos} min após isso")
        logger.info(f"Frequência de amostragem: {frequencia_amostragem} Hz")
        logger.info(f"Índices: início={self.idx_inicio}, fim={self.idx_fim}")
    
    def _get_lista_pacientes(self):
        """Obtém a lista de pacientes filtrada"""
        # Listar todos os arquivos parquet
        arquivos_parquet = list(self.data_folder_path.glob("*.parquet"))
        
        # Extrair IDs dos pacientes dos nomes dos arquivos
        ids_disponiveis = []
        for arquivo in arquivos_parquet:
            id_paciente = arquivo.stem  # Nome do arquivo sem extensão
            ids_disponiveis.append(id_paciente)
        
        # Filtrar: pacientes que estão nos dados clínicos E são válidos (têm X, Y, Z)
        lista_filtrada = [
            pid for pid in ids_disponiveis 
            if pid in dados['id_paciente'].values and pid in pacientes_validos
        ]
        
        # Aplicar limite de labels se especificado
        if self.limite_labels is not None:
            lista_filtrada = lista_filtrada[:self.limite_labels]
        
        logger.info(f"Total de arquivos Parquet: {len(arquivos_parquet)}")
        logger.info(f"Pacientes válidos (com X, Y, Z): {len(lista_filtrada)}")
        
        return np.array(lista_filtrada)
    
    
    def _carregar_paciente(self, paciente):
        """
        Carrega dados de um paciente do arquivo Parquet, aplicando filtros temporais.
        
        Args:
            paciente: ID do paciente
            
        Returns:
            tuple: (ecg_x, ecg_y, ecg_z, tempo, evento) ou None se houver erro
        """
        arquivo_parquet = self.data_folder_path / f"{paciente}.parquet"
        
        if not arquivo_parquet.exists():
            logger.warning(f"Arquivo não encontrado: {arquivo_parquet}")
            return None
        
        try:
            # Carregar o parquet completo
            df_paciente = pd.read_parquet(arquivo_parquet)
            
            total_amostras = len(df_paciente)
            
            # Garantir que não ultrapasse o tamanho dos dados
            idx_fim_real = min(self.idx_fim, total_amostras)
            
            if self.idx_inicio >= total_amostras:
                logger.warning(f"Paciente {paciente}: não há dados após {self.minutos_a_pular} minutos (total: {total_amostras} amostras)")
                return None
            
            # Filtrar dados temporalmente
            df_filtrado = df_paciente.iloc[self.idx_inicio:idx_fim_real]
            
            if len(df_filtrado) == 0:
                logger.warning(f"Paciente {paciente}: dados filtrados resultaram em 0 amostras")
                return None
            
            # Extrair sinais (assumindo que as colunas são 'sinal_x', 'sinal_y', 'sinal_z')
            ecg_x = df_filtrado['sinal_x'].values
            ecg_y = df_filtrado['sinal_y'].values
            ecg_z = df_filtrado['sinal_z'].values
            
            # Buscar informações clínicas
            tempo = dados[dados['id_paciente'] == paciente]['tempo'].iloc[0]
            evento = dados[dados['id_paciente'] == paciente]['evento'].iloc[0]
            
            return ecg_x, ecg_y, ecg_z, tempo, evento
            
        except Exception as e:
            logger.error(f"Erro ao carregar paciente {paciente}: {e}")
            return None


    def ecg_generator(self, lista_pacientes, batch_size=None):
        """
        Gerador que carrega dados sob demanda dos arquivos Parquet.
        
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
        
        while True:  # Loop infinito para múltiplas épocas
            for offset in range(0, num_pacientes, batch_size):
                batch_pacientes = lista_pacientes[offset:offset + batch_size]
                
                sinais_x_batch = []
                sinais_y_batch = []
                sinais_z_batch = []
                tempos_batch = []
                eventos_batch = []
                
                # Processar cada paciente do batch
                for paciente in batch_pacientes:
                    try:
                        resultado = self._carregar_paciente(paciente)
                        
                        if resultado is None:
                            continue
                        
                        ecg_x, ecg_y, ecg_z, tempo, evento = resultado
                        
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
                    # Encontrar o tamanho mínimo para padronizar
                    min_len = min(len(s) for s in sinais_x_batch)
                    
                    # Truncar todos os sinais para o mesmo tamanho
                    sinais_x_batch = [s[:min_len] for s in sinais_x_batch]
                    sinais_y_batch = [s[:min_len] for s in sinais_y_batch]
                    sinais_z_batch = [s[:min_len] for s in sinais_z_batch]
                    
                    tensor_x = torch.tensor(np.array(sinais_x_batch), dtype=torch.float32)
                    tensor_y = torch.tensor(np.array(sinais_y_batch), dtype=torch.float32)
                    tensor_z = torch.tensor(np.array(sinais_z_batch), dtype=torch.float32)
                    
                    sinais_tensor = torch.stack([tensor_x, tensor_y, tensor_z], dim=1)
                    tempos_array = np.array(tempos_batch)
                    eventos_array = np.array(eventos_batch)
                    
                    yield sinais_tensor, tempos_array, eventos_array
                else:
                    logger.warning(f"Batch vazio no offset {offset}")
    

    def preparar_dados(self):
        """
        Prepara os dados para treinamento, dividindo em TREINO, VALIDAÇÃO e TESTE.
        Retorna geradores e informações necessárias.
        """
        lista_pacientes = self._get_lista_pacientes()
        
        print(f"📊 Total de pacientes encontrados: {len(lista_pacientes)}")

        # Buscar eventos para estratificação
        eventos_pacientes = [dados[dados['id_paciente'] == p]['evento'].iloc[0] for p in lista_pacientes]

        # ETAPA 1: Divisão em Treino+Validação (80%) e Teste (20%)
        indices = np.arange(len(lista_pacientes))
        
        idx_train_val, idx_test = train_test_split(
            indices,
            test_size=0.2, 
            random_state=123,
            stratify=eventos_pacientes
        )

        self.pacientes_teste = lista_pacientes[idx_test]
        pacientes_train_val = lista_pacientes[idx_train_val]
        eventos_train_val = np.array(eventos_pacientes)[idx_train_val]
        
        # ETAPA 2: Divisão em Treino (80%) e Validação (20%)
        idx_train, idx_val = train_test_split(
            np.arange(len(pacientes_train_val)),
            test_size=0.25,
            random_state=123,
            stratify=eventos_train_val
        )

        self.pacientes_treino = pacientes_train_val[idx_train]
        self.pacientes_validacao = pacientes_train_val[idx_val]

        print(f"   - Treino:    {len(self.pacientes_treino)} pacientes")
        print(f"   - Validação: {len(self.pacientes_validacao)} pacientes")
        print(f"   - Teste:     {len(self.pacientes_teste)} pacientes")
        
        # Configurar labtrans
        num_durations = 20
        self.labtrans = LogisticHazard.label_transform(num_durations)
        
        print("🔄 Configurando labtrans (usando apenas dados de treino)...")
        gen_temp = self.ecg_generator(self.pacientes_treino, batch_size=min(16, len(self.pacientes_treino)))
        sinais_temp, tempos_temp, eventos_temp = next(gen_temp)
        
        self.amostra_sinal = sinais_temp[:1].numpy()
        self.labtrans.fit_transform(tempos_temp, eventos_temp)
        
        print(f"✅ Labtrans configurado com {self.labtrans.out_features} durações")
        
        # Calcular steps por época
        steps_per_epoch_train = (len(self.pacientes_treino) + self.batch_size - 1) // self.batch_size
        steps_per_epoch_val = (len(self.pacientes_validacao) + self.batch_size - 1) // self.batch_size
        steps_per_epoch_test = (len(self.pacientes_teste) + self.batch_size - 1) // self.batch_size
        
        return {
            'train_generator': self.ecg_generator(self.pacientes_treino),
            'val_generator': self.ecg_generator(self.pacientes_validacao),
            'test_generator': self.ecg_generator(self.pacientes_teste),
            'steps_per_epoch_train': steps_per_epoch_train,
            'steps_per_epoch_val': steps_per_epoch_val,
            'steps_per_epoch_test': steps_per_epoch_test,
            'num_train_samples': len(self.pacientes_treino),
            'num_val_samples': len(self.pacientes_validacao),
            'num_test_samples': len(self.pacientes_teste)
        }
    
    
    def get_labtrans(self):
        """Retorna o label transformer"""
        return self.labtrans
        
    def get_amostra_sinal(self):
        """Retorna uma amostra de sinal para testes"""
        return self.amostra_sinal