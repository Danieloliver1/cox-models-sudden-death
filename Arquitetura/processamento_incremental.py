# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import torch

from pycox.models import LogisticHazard
from sklearn.model_selection import train_test_split

# Assumindo que logger já está configurado
# logger = setup_logger(...)

# ============================================
# CAMINHOS CONFIGURÁVEIS
# ============================================
path_dados_parquet = r"D:\cox-models-sudden-death\01_Dataset\dados_ECG\dados_ECG_parquet"
path_filtro_pacientes_csv = r'D:\cox-models-sudden-death\02_Preprocessamento_filtro\resumo_dataset_ecg.csv'
path_info_clinica_csv = r'D:\cox-models-sudden-death\01_Dataset\dados_csv_info_definitions\subject-info_formatado.csv'

# ============================================
# CARREGAR DADOS CLÍNICOS (MANTÉM EM MEMÓRIA - PEQUENO)
# ============================================
df = pd.read_csv(path_info_clinica_csv, sep=';', 
                 usecols=['Patient ID','Follow-up period from enrollment (days)',
                         'Cause of death','Gender (male=1)','Age'])

df.rename(columns={
    'Patient ID':'id_paciente',
    'Follow-up period from enrollment (days)':'tempo',
    'Cause of death':'evento',
    'Gender (male=1)':'genero',
    'Age':'idade'
}, inplace=True)

# 0 sobreviventes, censurados 1 (6 e 7 arritmias)
dados = df.query('(evento == 0) or (evento == 6) or (evento == 7)')
dados = dados.reset_index(drop=True).sort_values("tempo")
dados['evento'] = dados['evento'].replace(6, 1).replace(7, 1)

# ============================================
# CARREGAR FILTRO DE PACIENTES VÁLIDOS
# ============================================
df_filtro = pd.read_csv(path_filtro_pacientes_csv, sep=',')
df_lista_validos = df_filtro[(df_filtro['tem_X'] == 'Sim') & 
                              (df_filtro['tem_Y'] == 'Sim') & 
                              (df_filtro['tem_Z'] == 'Sim')]
pacientes_validos = set(df_lista_validos['paciente'].values)


class PassandoDadosIncremental:
    """
    Classe para processamento INCREMENTAL de dados de ECG usando geradores eficientes.
    
    CORREÇÃO: Agora usa índices de pacientes e carrega dados sob demanda,
    sem manter tudo em memória.
    """
    def __init__(self, 
                 data_folder_path=path_dados_parquet,
                 path_filtro_csv=path_filtro_pacientes_csv,
                 minutos_a_pular=60, 
                 tamanho_janela_amostras=12000,
                 batch_size=32,
                 limite_labels=None,
                 frequencia_amostragem=200):
        
        self.data_folder_path = Path(data_folder_path)
        self.path_filtro_csv = path_filtro_csv
        self.minutos_a_pular = minutos_a_pular
        self.tamanho_janela_amostras = tamanho_janela_amostras
        self.batch_size = batch_size
        self.limite_labels = limite_labels
        self.frequencia_amostragem = frequencia_amostragem
        
        # Calcular offset inicial
        self.amostras_por_minuto = frequencia_amostragem * 60
        self.idx_offset_inicial = self.minutos_a_pular * self.amostras_por_minuto
        
        # Estado incremental: armazena apenas o índice atual de cada paciente
        self.estado_incremental = {}
        
        self.labtrans = None
        self.amostra_sinal = None
        self.pacientes_treino = None
        self.pacientes_validacao = None
        self.pacientes_teste = None
        
        print(f"⚙️  Configuração Incremental:")
        print(f"   - Pulando primeiros {minutos_a_pular} min ({self.idx_offset_inicial} amostras)")
        print(f"   - Janela incremental: {tamanho_janela_amostras} amostras")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Frequência: {frequencia_amostragem} Hz")
    
    
    def _get_lista_pacientes(self):
        """Obtém lista de pacientes válidos (mantém apenas IDs, não os dados)"""
        arquivos_parquet = list(self.data_folder_path.glob("*.parquet"))
        
        ids_disponiveis = [arquivo.stem for arquivo in arquivos_parquet]
        
        lista_filtrada = [
            pid for pid in ids_disponiveis 
            if pid in dados['id_paciente'].values and pid in pacientes_validos
        ]
        
        if self.limite_labels is not None:
            lista_filtrada = lista_filtrada[:self.limite_labels]
        
        print(f"📊 Total de arquivos Parquet: {len(arquivos_parquet)}")
        print(f"📊 Pacientes válidos (com X, Y, Z): {len(lista_filtrada)}")
        
        return np.array(lista_filtrada)
    
    
    def _carregar_chunk_paciente(self, paciente, inicio_chunk=None):
        """
        Carrega APENAS um chunk específico de um paciente.
        NÃO mantém dados anteriores em memória.
        """
        arquivo_parquet = self.data_folder_path / f"{paciente}.parquet"
        
        if not arquivo_parquet.exists():
            return None
        
        try:
            # Determinar índice inicial
            if inicio_chunk is None:
                idx_inicio = self.estado_incremental.get(paciente, self.idx_offset_inicial)
            else:
                idx_inicio = inicio_chunk
            
            # Calcular fim do chunk
            idx_fim = idx_inicio + self.tamanho_janela_amostras
            
            # Carregar APENAS as linhas necessárias (não o arquivo inteiro)
            # Infelizmente, parquet não suporta slice direto, mas podemos otimizar lendo tudo uma vez
            df_paciente = pd.read_parquet(arquivo_parquet)
            total_amostras = len(df_paciente)
            
            # Verificar disponibilidade
            if idx_inicio >= total_amostras:
                return None
            
            idx_fim = min(idx_fim, total_amostras)
            
            # Extrair chunk
            df_chunk = df_paciente.iloc[idx_inicio:idx_fim]
            del df_paciente  # ✅ LIBERA MEMÓRIA IMEDIATAMENTE
            
            if len(df_chunk) == 0:
                return None
            
            # Extrair sinais
            ecg_x = df_chunk['sinal_x'].values
            ecg_y = df_chunk['sinal_y'].values
            ecg_z = df_chunk['sinal_z'].values
            del df_chunk  # ✅ LIBERA MEMÓRIA
            
            # Buscar tempo e evento (dados clínicos são pequenos)
            tempo = dados[dados['id_paciente'] == paciente]['tempo'].iloc[0]
            evento = dados[dados['id_paciente'] == paciente]['evento'].iloc[0]
            
            idx_proximo = idx_fim
            
            return ecg_x, ecg_y, ecg_z, tempo, evento, idx_proximo
            
        except Exception as e:
            print(f"❌ Erro ao carregar paciente {paciente}: {e}")
            return None
    
    
    def inicializar_estado_incremental(self, lista_pacientes):
        """Inicializa estado (apenas índices)"""
        self.estado_incremental = {
            pid: self.idx_offset_inicial 
            for pid in lista_pacientes
        }
    
    
    def resetar_estado_incremental(self, lista_pacientes=None):
        """Reseta estado para início"""
        if lista_pacientes is None:
            lista_pacientes = list(self.estado_incremental.keys())
        
        for pid in lista_pacientes:
            self.estado_incremental[pid] = self.idx_offset_inicial
    
    
    def incrementar_janela(self, lista_pacientes, adicionar_amostras):
        """Incrementa o offset de janela para próxima iteração"""
        for pid in lista_pacientes:
            if pid in self.estado_incremental:
                self.estado_incremental[pid] += adicionar_amostras
    
    
    def ecg_generator_incremental(self, lista_pacientes, batch_size=None, 
                                  atualizar_estado=True, shuffle=False):
        """
        ✅ GERADOR VERDADEIRO: Carrega dados sob demanda, batch por batch.
        
        Não mantém dados em memória - cada batch é gerado quando solicitado.
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        # Shuffle da lista de pacientes (não dos dados)
        if shuffle:
            indices = np.random.permutation(len(lista_pacientes))
            lista_pacientes = lista_pacientes[indices]
        
        num_pacientes = len(lista_pacientes)
        
        # ✅ Itera sobre índices, não sobre dados carregados
        for offset in range(0, num_pacientes, batch_size):
            batch_pacientes = lista_pacientes[offset:offset + batch_size]
            
            sinais_x_batch = []
            sinais_y_batch = []
            sinais_z_batch = []
            tempos_batch = []
            eventos_batch = []
            pacientes_validos_batch = []
            
            # ✅ Carrega APENAS os pacientes deste batch
            for paciente in batch_pacientes:
                resultado = self._carregar_chunk_paciente(paciente)
                
                if resultado is None:
                    continue
                
                ecg_x, ecg_y, ecg_z, tempo, evento, idx_proximo = resultado
                
                # Atualizar estado se necessário
                if atualizar_estado:
                    self.estado_incremental[paciente] = idx_proximo
                
                sinais_x_batch.append(ecg_x)
                sinais_y_batch.append(ecg_y)
                sinais_z_batch.append(ecg_z)
                tempos_batch.append(tempo)
                eventos_batch.append(evento)
                pacientes_validos_batch.append(paciente)
            
            # Converter para tensores apenas se houver dados válidos
            if len(sinais_x_batch) == 0:
                continue
            
            # Padronizar tamanho (truncar ao mínimo)
            min_len = min(len(s) for s in sinais_x_batch)
            
            sinais_x_batch = [s[:min_len] for s in sinais_x_batch]
            sinais_y_batch = [s[:min_len] for s in sinais_y_batch]
            sinais_z_batch = [s[:min_len] for s in sinais_z_batch]
            
            # Criar tensores
            tensor_x = torch.tensor(np.array(sinais_x_batch), dtype=torch.float32)
            tensor_y = torch.tensor(np.array(sinais_y_batch), dtype=torch.float32)
            tensor_z = torch.tensor(np.array(sinais_z_batch), dtype=torch.float32)
            
            sinais_tensor = torch.stack([tensor_x, tensor_y, tensor_z], dim=1)
            tempos_array = np.array(tempos_batch)
            eventos_array = np.array(eventos_batch)
            
            # ✅ Yield libera controle - memória do batch anterior pode ser liberada
            yield sinais_tensor, tempos_array, eventos_array
            
            # ✅ Limpar referências explicitamente
            del sinais_tensor, tempos_array, eventos_array
            del sinais_x_batch, sinais_y_batch, sinais_z_batch
            del tensor_x, tensor_y, tensor_z
    
    
    def get_chunk_info(self, paciente):
        """Retorna informações sobre estado de um paciente"""
        arquivo_parquet = self.data_folder_path / f"{paciente}.parquet"
        
        if not arquivo_parquet.exists():
            return None
        
        # Ler apenas metadados (número de linhas)
        df_paciente = pd.read_parquet(arquivo_parquet, columns=[])
        total_amostras = len(df_paciente)
        del df_paciente
        
        idx_atual = self.estado_incremental.get(paciente, self.idx_offset_inicial)
        amostras_restantes = max(0, total_amostras - idx_atual)
        
        denominador = max(1, total_amostras - self.idx_offset_inicial)
        progresso = (idx_atual - self.idx_offset_inicial) / denominador * 100
        
        return {
            'idx_atual': idx_atual,
            'total_amostras': total_amostras,
            'amostras_restantes': amostras_restantes,
            'progresso_pct': progresso
        }
    

    def preparar_dados(self):
        """
        Prepara estrutura de dados (índices) para treinamento.
        NÃO carrega dados em memória - apenas organiza referências.
        """
        lista_pacientes = self._get_lista_pacientes()
        
        print(f"\n📊 Total de pacientes encontrados: {len(lista_pacientes)}")

        # Buscar eventos para estratificação
        eventos_pacientes = [
            dados[dados['id_paciente'] == p]['evento'].iloc[0] 
            for p in lista_pacientes
        ]

        # Divisão estratificada
        indices = np.arange(len(lista_pacientes))
        
        idx_train_val, idx_test = train_test_split(
            indices, test_size=0.2, random_state=123,
            stratify=eventos_pacientes
        )

        self.pacientes_teste = lista_pacientes[idx_test]
        pacientes_train_val = lista_pacientes[idx_train_val]
        eventos_train_val = np.array(eventos_pacientes)[idx_train_val]
        
        idx_train, idx_val = train_test_split(
            np.arange(len(pacientes_train_val)),
            test_size=0.25, random_state=123,
            stratify=eventos_train_val
        )

        self.pacientes_treino = pacientes_train_val[idx_train]
        self.pacientes_validacao = pacientes_train_val[idx_val]

        print(f"   - Treino:    {len(self.pacientes_treino)} pacientes")
        print(f"   - Validação: {len(self.pacientes_validacao)} pacientes")
        print(f"   - Teste:     {len(self.pacientes_teste)} pacientes")
        
        # Inicializar estado
        self.inicializar_estado_incremental(lista_pacientes)
        
        # Configurar labtrans (carrega APENAS dados necessários)
        num_durations = 20
        self.labtrans = LogisticHazard.label_transform(num_durations)
        
        print("\n🔄 Configurando labtrans...")
        
        # Usa apenas 1 batch pequeno para configurar
        gen_temp = self.ecg_generator_incremental(
            self.pacientes_treino[:min(16, len(self.pacientes_treino))], 
            batch_size=min(16, len(self.pacientes_treino)),
            atualizar_estado=False
        )
        
        sinais_temp, tempos_temp, eventos_temp = next(gen_temp)
        self.amostra_sinal = sinais_temp[:1].cpu().numpy()
        self.labtrans.fit_transform(tempos_temp, eventos_temp)
        
        # Limpar
        del sinais_temp, tempos_temp, eventos_temp, gen_temp
        
        print(f"✅ Labtrans configurado com {self.labtrans.out_features} durações")
        
        # Resetar estado
        self.resetar_estado_incremental()
        
        # Calcular steps
        steps_per_epoch_train = (len(self.pacientes_treino) + self.batch_size - 1) // self.batch_size
        steps_per_epoch_val = (len(self.pacientes_validacao) + self.batch_size - 1) // self.batch_size
        steps_per_epoch_test = (len(self.pacientes_teste) + self.batch_size - 1) // self.batch_size
        
        return {
            'train_generator': lambda: self.ecg_generator_incremental(
                self.pacientes_treino, shuffle=True
            ),
            'val_generator': lambda: self.ecg_generator_incremental(
                self.pacientes_validacao, atualizar_estado=False
            ),
            'test_generator': lambda: self.ecg_generator_incremental(
                self.pacientes_teste, atualizar_estado=False
            ),
            'steps_per_epoch_train': steps_per_epoch_train,
            'steps_per_epoch_val': steps_per_epoch_val,
            'steps_per_epoch_test': steps_per_epoch_test,
            'num_train_samples': len(self.pacientes_treino),
            'num_val_samples': len(self.pacientes_validacao),
            'num_test_samples': len(self.pacientes_teste),
            'tamanho_janela_atual': self.tamanho_janela_amostras
        }
    
    
    def get_labtrans(self):
        """Retorna label transformer"""
        return self.labtrans
        
    def get_amostra_sinal(self):
        """Retorna amostra de sinal"""
        return self.amostra_sinal