# -*- coding: utf-8 -*-
import pandas as pd
import glob
import wfdb
import os
import duckdb
import time
import pandas as pd
import duckdb

import neurokit2 as nk
import pywt  # Biblioteca para transformada wavelet


# Define o padrão para os arquivos CSV

#path = r'D:\Projeto_Tese_mestrado\02_Dataset\dados_ECG\High-resolution_ECG\P0*'
#path_holter = r'D:\Projeto_Tese_mestrado\02_Dataset\dados_ECG\Holter_ECG\P0*'
path =         r'D:\cox-models-sudden-death\01_Dataset\dados_ECG\Holter_ECG\P*'

#path = r'E:/Repositorio_Git/zzz-projeto_final/dados/ca-*.csv'


# Usando glob para pegar todos os arquivos que seguem o padrão
arquivos = glob.glob(path)
#arquivos_path_holter = glob.glob(path_holter)

arquivos = pd.Series(arquivos).unique().tolist() # removando duplicadas

arquivos = [os.path.splitext(arquivo)[0] for arquivo in arquivos] # tirando as a extensões

#arquivos_path_holter = [os.path.splitext(arquivos_path_holter)[0] for arquivos_path_holter in arquivos_path_holter] # tirando as a extensões

lista_sem_duplicatas = pd.Series(arquivos).unique().tolist() # removendo duplicadas

#link = "D:/cox-models-sudden-death/01_Dataset/Duckedb/Holter_ECG/1h/banco_ecg.duckdb"

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
        #self.id_paciente = id_paciente  # salva o ID internamente
        

        #teste2
        # Por exemplo: preserve cD1 e cD2
        # zera só os detalhes de níveis mais altos
        ecg_wavelet_filtered = self.wavelet_filter_preservando_qrs(signal, wavelet='db6', level=2, atenuacao=0.2)


        # Limpeza do sinal com o método de Pan-Tompkins
        ecg_limpo = nk.ecg_clean(ecg_wavelet_filtered, sampling_rate=200, method=metodo)

        # Ajuste de escala (opcional): desloca o sinal para cima, garantindo valores positivos
        # self.ecg_limpo = ecg_limpo + 2
        
    
        return ecg_limpo
    
    
class ECGduckdb:
    """_summary_

        Args:
            link (_type_): _description_
            hora_min (_type_): _description_
            valor_hora (_type_): _description_
            inicio_amostra (int, optional): _description_. Defaults to 720000.
    """
    
    def __init__(self, link, hora_min,valor_hora, inicio_amostra = 720000,metodo='neurokit'):
        
        
        self.link = link
           
        # Parâmetros de segmentação
        self.inicio_amostra = inicio_amostra  

       
        if hora_min == 'h':
            self.duracao_segundos = valor_hora * 3600 # duração em segundos
        elif hora_min == 'm':
            self.duracao_segundos = valor_hora * 60   # duração em segundos
       
        # Dentro do método mapeando, após ler o record:
        # record = wfdb.rdrecord(endereco)
        record = wfdb.rdrecord(lista_sem_duplicatas[0]) # como todos os pacientes tem a mesma frequencia vou usar o primeiro como padrão
        fs = record.fs  # Pega a frequência de amostragem (ex: 200)
        
        # Calcula a duração em amostras dinamicamente
        self.duracao_amostras = int(self.duracao_segundos * fs)
        
        self.tamanho_lote = 200
        self.metodo = metodo
       
        

    def mapeando(self):
        
        # Iniciar contagem do tempo
        tempo_inicio = time.time()

        # # Conectar ao banco de dados DuckDB
        conn = duckdb.connect(self.link)


        # Criar a tabela se não existir
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ecg_pacientes_limpo (
                id_paciente TEXT,
                amostra INTEGER,
                sinal_x FLOAT,
                sinal_y FLOAT,
                sinal_z FLOAT
            )
        """)
        # A criação do índice só precisa ser feita uma vez.
        # O try/except evita erro se o índice já existir.
        try:
            conn.execute("CREATE INDEX idx_paciente ON ecg_pacientes_limpo (id_paciente, amostra);")
        except duckdb.duckdb.CatalogException:
            print("ℹ️ Índice 'idx_paciente' já existe.")


        # Loop sobre os arquivos sem duplicatas
        for idx, endereco in enumerate(lista_sem_duplicatas):
            try:
                print(f"🔄 Processando paciente {idx+1}/{len(lista_sem_duplicatas)}: {endereco}")

                # Lendo o sinal do ECG
                record = wfdb.rdrecord(endereco)
                p_signal = record.p_signal

                # Verificar se há pelo menos 3 canais
                if p_signal.shape[1] < 3:
                    print(f"⚠️ Paciente {record.record_name} tem menos de 3 canais de ECG. Pulando...")
                    continue

                # Obter sinais X, Y e Z
                sinal_x = p_signal[:, 0]
                sinal_y = p_signal[:, 1]
                sinal_z = p_signal[:, 2]
                
                # Criar o processador de ECG
                atributos = ECGProcessor()
                
                # Processar cada sinal
                amostra_x = atributos.process(sinal_x, sampling_rate=200, metodo=self.metodo)
                amostra_y = atributos.process(sinal_y, sampling_rate=200, metodo=self.metodo)
                amostra_z = atributos.process(sinal_z, sampling_rate=200, metodo=self.metodo)

                # Filtrar os sinais
                sinal_fil_x = amostra_x[self.inicio_amostra: self.inicio_amostra + self.duracao_amostras]
                sinal_fil_y = amostra_y[self.inicio_amostra: self.inicio_amostra + self.duracao_amostras]
                sinal_fil_z = amostra_z[self.inicio_amostra: self.inicio_amostra + self.duracao_amostras]

                # Identificação do paciente
                id_paciente = record.record_name  
                
                # # Criar um DataFrame com os dados do paciente

                df_ecg = pd.DataFrame({
                    "id_paciente": id_paciente,
                    "amostra": range(len(sinal_fil_x)),
                    "sinal_x": sinal_fil_x,
                    "sinal_y": sinal_fil_y,
                    "sinal_z": sinal_fil_z
                })


                # Inserir os dados no banco em lote
                if not df_ecg.empty:
                    print(f"📂 Inserindo dados do paciente {id_paciente} no banco...")
                    #conn.execute("INSERT INTO ecg_pacientes_limpo SELECT * FROM df_ecg")
                    conn.execute("BEGIN")
                    conn.execute("INSERT INTO ecg_pacientes_limpo SELECT * FROM df_ecg")
                    conn.execute("COMMIT")

                    print(f"✅ Dados do paciente {id_paciente} inseridos com sucesso!")

                # Liberar memória após cada lote
                del df_ecg
                #dados_geral.clear()

                # A cada 200 pacientes, limpar a memória e inserir novos dados
                if (idx + 1) % self.tamanho_lote == 0:
                    print(f"💾 Lote de {self.tamanho_lote} pacientes processado. Aguardando próximo lote...")

            except Exception as e:
                print(f"❌ Erro ao processar {endereco}: {e}")
                continue  # Continua para o próximo paciente

        

        # Finalizar contagem do tempo
        tempo_fim = time.time()
        tempo_total = tempo_fim - tempo_inicio

        # Exibir o tempo total de execução
        print(f"⏳ Tempo total de execução: {tempo_total:.2f} segundos")
        
        total_paciente = conn.execute("""SELECT COUNT(DISTINCT id_paciente) AS total_pacientes FROM ecg_pacientes_limpo  """).fetchdf()
        print(total_paciente)
        quant_atributo = conn.execute("""SELECT COUNT(sinal_x) AS total_atributos FROM ecg_pacientes_limpo WHERE id_paciente='P0001' """).fetchdf()
        print(f'Quantidade de atributo por paciente {quant_atributo['total_atributos'].values}')
        
        # Fechar conexão com o banco
        conn.close()



