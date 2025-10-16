# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
import torch
from pathlib import Path
import glob
import os
import wfdb  # MUDANÇA: Importamos a biblioteca wfdb

from logger_config import setup_logger
from pycox.models import LogisticHazard
from sklearn.model_selection import train_test_split

# ... (toda a sua configuração inicial de logs e paths permanece a mesma)
# ... (o carregamento e tratamento do df de informações clínicas também permanece o mesmo)

# ============================================
# CAMINHOS CONFIGURÁVEIS
# ============================================

# MUDANÇA: Novo caminho para os dados brutos do Holter
path_dados_holter = r"D:\cox-models-sudden-death\01_Dataset\dados_ECG\Holter_ECG" 

path_filtro_pacientes_csv = r'D:\cox-models-sudden-death\02_Preprocessamento_filtro\resumo_dataset_ecg.csv'
path_info_clinica_csv = r'D:\cox-models-sudden-death\01_Dataset\dados_csv_info_definitions\subject-info_formatado.csv'

# ... (O código para carregar e preparar o dataframe 'dados' continua igual)
logger = setup_logger(r"D:\cox-models-sudden-death\Arquitetura\logs","processamento_dados.log", name="Processamento")
df = pd.read_csv(path_info_clinica_csv, sep=';', usecols=['Patient ID','Follow-up period from enrollment (days)','Cause of death','Gender (male=1)','Age'])
df.rename(columns={'Patient ID':'id_paciente',
    'Follow-up period from enrollment (days)':'tempo',
                  'Cause of death':'evento',
                  'Gender (male=1)':'genero',
                  'Age':'idade'}, inplace=True)
dados = df.query('(evento == 0) or (evento == 6) or (evento == 7)')
dados = dados.reset_index(drop=True).sort_values("tempo")
dados['evento'] = dados['evento'].replace(6, 1).replace(7, 1)
df_filtro = pd.read_csv(path_filtro_pacientes_csv, sep=',')
df_lista_validos = df_filtro[(df_filtro['tem_X'] == 'Sim') & 
                             (df_filtro['tem_Y'] == 'Sim') & 
                             (df_filtro['tem_Z'] == 'Sim')]
pacientes_validos = set(df_lista_validos['paciente'].values)
# ============================================

class PassandoDados: # MUDANÇA: Nome da classe para refletir a nova fonte de dados
    """
    Classe para processar dados de ECG usando arquivos WFDB (.hea, .dat).
    Carrega os dados sob demanda para evitar sobrecarga de memória.
    
    Args:
        data_folder_path (str): Caminho para a pasta com arquivos WFDB
        # ... (o resto dos argumentos é o mesmo)
    """
    def __init__(self, 
                 data_folder_path=path_dados_holter, # MUDANÇA: Path padrão aponta para a pasta Holter
                 path_filtro_csv=path_filtro_pacientes_csv,
                 minutos_a_pular=60, 
                 duracao_em_minutos=20,
                 batch_size=32,
                 limite_labels=None,
                 frequencia_amostragem=200):
        
        # MUDANÇA: Não usamos mais Pathlib aqui, pois glob lida bem com strings
        self.data_folder_path = data_folder_path
        self.path_filtro_csv = path_filtro_csv
        self.minutos_a_pular = minutos_a_pular
        self.duracao_em_minutos = duracao_em_minutos
        self.batch_size = batch_size
        self.limite_labels = limite_labels
        self.frequencia_amostragem = frequencia_amostragem
        
        # O cálculo dos índices permanece o mesmo
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
        """Obtém a lista de pacientes filtrada a partir dos arquivos WFDB."""
        # MUDANÇA: Usamos glob para encontrar todos os arquivos de cabeçalho (.hea)
        # O padrão 'P0*' busca todos os pacientes que começam com 'P0'
        padrao_busca = os.path.join(self.data_folder_path, 'P0*.hea')
        arquivos_hea = glob.glob(padrao_busca)
        
        # MUDANÇA: Extraímos o "nome do registro" sem a extensão e o caminho
        # wfdb.rdrecord precisa do caminho completo sem a extensão
        ids_disponiveis = [os.path.splitext(f)[0] for f in arquivos_hea]
        
        # MUDANÇA: Extrair apenas o nome do paciente (ex: 'P0001') para filtrar
        nomes_pacientes_disponiveis = [os.path.basename(id) for id in ids_disponiveis]

        # Filtrar: pacientes que estão nos dados clínicos E são válidos (têm X, Y, Z)
        lista_filtrada = []
        for i, nome_paciente in enumerate(nomes_pacientes_disponiveis):
            if nome_paciente in dados['id_paciente'].values and nome_paciente in pacientes_validos:
                # Guardamos o caminho completo sem extensão
                lista_filtrada.append(ids_disponiveis[i])
        
        if self.limite_labels is not None:
            lista_filtrada = lista_filtrada[:self.limite_labels]
        
        logger.info(f"Total de arquivos .hea encontrados: {len(arquivos_hea)}")
        logger.info(f"Pacientes válidos após filtro: {len(lista_filtrada)}")
        
        return np.array(lista_filtrada)
    
    # def _carregar_paciente(self, record_name): 
        """
        MUDANÇA: Carrega dados de um paciente a partir do seu registro WFDB.
        
        Args:
            record_name (str): Caminho para o registro, sem extensão (ex: 'D:/.../P0001')
            
        Returns:
            tuple: (ecg_x, ecg_y, ecg_z, tempo, evento) ou None se houver erro
        """
        try:
            # MUDANÇA: Carrega o registro WFDB. Não é preciso especificar a extensão.
            record = wfdb.rdrecord(record_name)
            
            # O sinal está em record.p_signal (formato: [amostras, canais])
            sinal_completo = record.p_signal
            total_amostras = len(sinal_completo)
            
            # Garantir que não ultrapasse o tamanho dos dados
            idx_fim_real = min(self.idx_fim, total_amostras)
            
            if self.idx_inicio >= total_amostras:
                logger.warning(f"Registro {os.path.basename(record_name)}: não há dados após {self.minutos_a_pular} minutos.")
                return None
            
            # MUDANÇA: Filtra o array numpy diretamente pelos índices
            sinal_filtrado = sinal_completo[self.idx_inicio:idx_fim_real]
            
            if len(sinal_filtrado) == 0:
                logger.warning(f"Registro {os.path.basename(record_name)}: dados filtrados resultaram em 0 amostras")
                return None
            
            # MUDANÇA: Extrai os sinais dos canais
            # Assumindo que o canal 0 é X, 1 é Y, e 2 é Z
            if sinal_filtrado.shape[1] < 3:
                logger.error(f"Registro {os.path.basename(record_name)} tem menos de 3 canais.")
                return None
                
            ecg_x = sinal_filtrado[:, 0]
            ecg_y = sinal_filtrado[:, 1]
            ecg_z = sinal_filtrado[:, 2]
            
            # A busca por informações clínicas continua igual
            paciente_id = os.path.basename(record_name)
            tempo = dados[dados['id_paciente'] == paciente_id]['tempo'].iloc[0]
            evento = dados[dados['id_paciente'] == paciente_id]['evento'].iloc[0]
            
            return ecg_x, ecg_y, ecg_z, tempo, evento
            
        except Exception as e:
            logger.error(f"Erro ao carregar registro {os.path.basename(record_name)}: {e}")
            return None

    def _carregar_paciente(self, record_name):
        """
        MUDANÇA: Carrega dados de um paciente a partir do seu registro WFDB.
        """
        try:
            # CORREÇÃO: Lê apenas o segmento necessário diretamente do disco.
            record = wfdb.rdrecord(record_name, 
                                sampfrom=self.idx_inicio, 
                                sampto=self.idx_fim)

            # Agora, record.p_signal JÁ É o sinal fatiado e pequeno.
            sinal_filtrado = record.p_signal
            
            if sinal_filtrado is None or len(sinal_filtrado) == 0:
                logger.warning(f"Registro {os.path.basename(record_name)}: dados filtrados resultaram em 0 amostras")
                return None
            
            # MUDANÇA: Extrai os sinais dos canais
            if sinal_filtrado.shape[1] < 3:
                logger.error(f"Registro {os.path.basename(record_name)} tem menos de 3 canais.")
                return None
                
            ecg_x = sinal_filtrado[:, 0]
            ecg_y = sinal_filtrado[:, 1]
            ecg_z = sinal_filtrado[:, 2]
            
            # A busca por informações clínicas continua igual
            paciente_id = os.path.basename(record_name)
            tempo = dados[dados['id_paciente'] == paciente_id]['tempo'].iloc[0]
            evento = dados[dados['id_paciente'] == paciente_id]['evento'].iloc[0]
            
            return ecg_x, ecg_y, ecg_z, tempo, evento
            
        except Exception as e:
            logger.error(f"Erro ao carregar registro {os.path.basename(record_name)}: {e}")
            return None



    # =========================================================================
    # NENHUMA MUDANÇA NECESSÁRIA DAQUI PARA BAIXO
    # A lógica do gerador e da preparação de dados é a mesma, pois ela
    # depende da saída de `_carregar_paciente`, que mantivemos consistente.
    # =========================================================================
    
    def ecg_generator(self, lista_pacientes, batch_size=None):
        """
        Gerador que carrega dados sob demanda.
        Funciona exatamente como antes.
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        num_pacientes = len(lista_pacientes)
        
        while True:
            for offset in range(0, num_pacientes, batch_size):
                batch_pacientes = lista_pacientes[offset:offset + batch_size]
                
                sinais_x_batch, sinais_y_batch, sinais_z_batch = [], [], []
                tempos_batch, eventos_batch = [], []
                
                for paciente_record_name in batch_pacientes:
                    try:
                        resultado = self._carregar_paciente(paciente_record_name)
                        if resultado is None:
                            continue
                        
                        ecg_x, ecg_y, ecg_z, tempo, evento = resultado
                        
                        sinais_x_batch.append(ecg_x)
                        sinais_y_batch.append(ecg_y)
                        sinais_z_batch.append(ecg_z)
                        tempos_batch.append(tempo)
                        eventos_batch.append(evento)
                        
                    except Exception as e:
                        logger.error(f"Erro ao processar paciente {os.path.basename(paciente_record_name)}: {e}")
                        continue
                
                if len(sinais_x_batch) > 0:
                    min_len = min(len(s) for s in sinais_x_batch)
                    
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
        Funciona exatamente como antes.
        """
        lista_pacientes_com_path = self._get_lista_pacientes()
        
        print(f"📊 Total de pacientes encontrados: {len(lista_pacientes_com_path)}")

        # MUDANÇA: Precisamos extrair o ID para buscar os eventos
        lista_pacientes_ids = [os.path.basename(p) for p in lista_pacientes_com_path]
        eventos_pacientes = [dados[dados['id_paciente'] == p]['evento'].iloc[0] for p in lista_pacientes_ids]

        indices = np.arange(len(lista_pacientes_com_path))
        
        idx_train_val, idx_test = train_test_split(indices, test_size=0.2, random_state=123, stratify=eventos_pacientes)

        self.pacientes_teste = lista_pacientes_com_path[idx_test]
        pacientes_train_val = lista_pacientes_com_path[idx_train_val]
        eventos_train_val = np.array(eventos_pacientes)[idx_train_val]
        
        idx_train, idx_val = train_test_split(np.arange(len(pacientes_train_val)), test_size=0.25, random_state=123, stratify=eventos_train_val)

        self.pacientes_treino = pacientes_train_val[idx_train]
        self.pacientes_validacao = pacientes_train_val[idx_val]

        print(f"   - Treino:     {len(self.pacientes_treino)} pacientes")
        print(f"   - Validação: {len(self.pacientes_validacao)} pacientes")
        print(f"   - Teste:      {len(self.pacientes_teste)} pacientes")
        
        num_durations = 20
        self.labtrans = LogisticHazard.label_transform(num_durations)
        
        print("🔄 Configurando labtrans (usando apenas dados de treino)...")
        # gen_temp = self.ecg_generator(self.pacientes_treino, batch_size=min(16, len(self.pacientes_treino)))
        # sinais_temp, tempos_temp, eventos_temp = next(gen_temp)
        
        # self.amostra_sinal = sinais_temp[:1].numpy()
        # self.labtrans.fit_transform(tempos_temp, eventos_temp)
        # Use um lote maior para garantir que teremos amostras de eventos e censuras
        fit_batch_size = min(len(self.pacientes_treino), 128) # Pega até 128 amostras ou o total se for menor
        gen_temp = self.ecg_generator(self.pacientes_treino, batch_size=fit_batch_size)
        sinais_temp, tempos_temp, eventos_temp = next(gen_temp)

        # Verificação opcional, mas recomendada, para garantir que há eventos no lote
        if np.sum(eventos_temp) == 0:
            raise RuntimeError("O lote de amostra para configurar o labtrans não contém eventos. Tente aumentar o 'fit_batch_size'.")

        self.amostra_sinal = sinais_temp[:1].numpy()
        self.labtrans.fit_transform(tempos_temp, eventos_temp)
        
        print(f"✅ Labtrans configurado com {self.labtrans.out_features} durações")
        
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
        return self.labtrans
        
    def get_amostra_sinal(self):
        return self.amostra_sinal