import numpy as np
import pandas as pd
import neurokit2 as nk
import pywt  # Biblioteca para transformada wavelet


# ============================================================
# FUNÇÃO AUXILIAR: FILTRO WAVELET
# ============================================================
def wavelet_filter_preservando_qrs(signal, wavelet='db6', level=4, atenuacao=0.2):
    """
    Aplica filtro wavelet preservando as características do complexo QRS.
    
    Parâmetros:
    -----------
    signal : array
        Sinal ECG a ser filtrado
    wavelet : str
        Tipo de wavelet (padrão: 'db6')
    level : int
        Nível de decomposição (padrão: 4)
    atenuacao : float
        Fator de atenuação para detalhes de alto nível (padrão: 0.2)
        
    Retorna:
    --------
    array: Sinal filtrado
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # Mantém detalhes de nível 1 e 2 (alta frequência = QRS)
    # Atenua detalhes de nível mais alto (ruído)
    for i in range(3, len(coeffs)):
        coeffs[i] *= atenuacao  # mantém apenas uma fração da energia
    
    ecg_wavelet_filtered = pywt.waverec(coeffs, wavelet)
    
    # Ajusta o tamanho se necessário (waverec pode retornar tamanho ligeiramente diferente)
    if len(ecg_wavelet_filtered) > len(signal):
        ecg_wavelet_filtered = ecg_wavelet_filtered[:len(signal)]
    elif len(ecg_wavelet_filtered) < len(signal):
        # Padding com zeros se necessário
        ecg_wavelet_filtered = np.pad(ecg_wavelet_filtered, 
                                       (0, len(signal) - len(ecg_wavelet_filtered)), 
                                       mode='constant')
    
    return ecg_wavelet_filtered


# ============================================================
# CLASSE PRINCIPAL
# ============================================================
class ECGMetricCalculator:
    """
    Calcula métricas de HRV (domínio do tempo) e morfológicas (QRS/QT) 
    para validação contra o CSV clínico.
    
    Usa apenas nk.hrv_time() por ser mais robusto que nk.ecg_intervalrelated().
    """
    def __init__(self, cleaned_signal, sampling_rate=200, debug=False, metodo='elgendi2010'):
        """
        Inicializa o calculador de métricas ECG.
        
        Parâmetros:
        -----------
        cleaned_signal : array
            Sinal ECG já pré-processado
        sampling_rate : int
            Taxa de amostragem (padrão: 200 Hz)
        debug : bool
            Se True, imprime mensagens de debug
        metodo : str
            Método de limpeza do NeuroKit2 ('elgendi2010', 'pantompkins1985', etc.)
        """
        self.sampling_rate = sampling_rate
        self.debug = debug
        self.metodo = metodo
        
        if self.debug:
            print(f"📊 Sinal recebido: {len(cleaned_signal)} amostras")
        
        # Limpeza do sinal com NeuroKit2
        self.ecg_limpo = nk.ecg_clean(cleaned_signal, sampling_rate=sampling_rate, method=metodo)
        
        if self.debug:
            print(f"✅ Sinal limpo com método '{metodo}': {len(self.ecg_limpo)} amostras")
        
        # Cria épocas
        self.epochs = self._create_epochs()
        
        if self.debug:
            print(f"📦 Épocas criadas: {len(self.epochs)} épocas")

    def _create_epochs(self):
        """Cria épocas de 120 segundos."""
        epoch_duration_seconds = 120
        epoch_length_samples = epoch_duration_seconds * self.sampling_rate
        num_epochs = len(self.ecg_limpo) // epoch_length_samples
        
        if self.debug:
            print(f"   Duração da época: {epoch_duration_seconds}s ({epoch_length_samples} amostras)")
            print(f"   Número de épocas: {num_epochs}")
        
        events = [i * epoch_length_samples for i in range(num_epochs)]
        
        return nk.epochs_create(
            self.ecg_limpo, 
            events=events, 
            sampling_rate=self.sampling_rate, 
            epochs_start=0, 
            epochs_end=epoch_duration_seconds
        )

    def calculate_metrics_for_epochs(self):
        """Calcula métricas para todas as épocas."""
        all_metrics = []
        epochs_processadas = 0
        epochs_com_erro = 0
        
        if self.debug:
            print("\n" + "="*60)
            print("🔍 INICIANDO PROCESSAMENTO DAS ÉPOCAS")
            print("="*60)
        
        for epoch_name, epoch_df in self.epochs.items():
            if self.debug:
                print(f"\n📌 Processando época '{epoch_name}'...")
            
            signal = epoch_df["Signal"].values

            try:
                # ============================================================
                # DETECÇÃO DE PICOS R
                # ============================================================
                peaks, info = nk.ecg_peaks(signal, sampling_rate=self.sampling_rate)
                num_picos = len(info['ECG_R_Peaks'])
                
                if self.debug:
                    print(f"   ✓ Picos R encontrados: {num_picos}")
                
                # Verifica se há picos suficientes
                if num_picos < 5:
                    if self.debug:
                        print(f"   ⚠️  PULANDO: Poucos picos R ({num_picos} < 5)")
                    epochs_com_erro += 1
                    continue

                # ============================================================
                # MÉTRICAS DE HRV (Domínio do Tempo)
                # ============================================================
                hrv_metrics = nk.hrv_time(peaks, sampling_rate=self.sampling_rate, show=False)
                
                hrv_mean_nn = hrv_metrics['HRV_MeanNN'].values[0]
                hrv_sdnn = hrv_metrics['HRV_SDNN'].values[0]
                hrv_rmssd = hrv_metrics['HRV_RMSSD'].values[0]
                hrv_pnn50 = hrv_metrics['HRV_pNN50'].values[0]
                
                # Calcula Min, Max e Range manualmente (mais confiável)
                rr_intervals_ms = np.diff(info["ECG_R_Peaks"]) / self.sampling_rate * 1000
                hrv_min_nn = np.min(rr_intervals_ms)
                hrv_max_nn = np.max(rr_intervals_ms)
                rr_range = hrv_max_nn - hrv_min_nn
                
                # Calcula Bradycardia (FC < 60 bpm)
                heart_rate = 60000 / np.mean(rr_intervals_ms)
                bradycardia = int(heart_rate < 60)
                
                if self.debug:
                    print(f"   ✓ Métricas HRV calculadas (FC: {heart_rate:.1f} bpm)")
                
                # ============================================================
                # MÉTRICAS MORFOLÓGICAS (QRS/QT)
                # ============================================================
                try:
                    delineate, _ = nk.ecg_delineate(
                        signal, 
                        rpeaks=info["ECG_R_Peaks"], 
                        sampling_rate=self.sampling_rate, 
                        method="dwt"
                    )
                    
                    # QRS Duration
                    q_peaks = delineate[delineate["ECG_Q_Peaks"] == 1].index.to_numpy()
                    s_peaks = delineate[delineate["ECG_S_Peaks"] == 1].index.to_numpy()
                    
                    if len(q_peaks) > 0 and len(s_peaks) > 0:
                        min_len_qs = min(len(q_peaks), len(s_peaks))
                        pares_qs = [(q, s) for q, s in zip(q_peaks[:min_len_qs], s_peaks[:min_len_qs]) if s > q]
                        qrs_duration_ms = [(s - q) / self.sampling_rate * 1000 for q, s in pares_qs]
                        qrs_duration_mean = np.nanmean(qrs_duration_ms) if len(qrs_duration_ms) > 0 else np.nan
                    else:
                        qrs_duration_mean = np.nan
                        if self.debug:
                            print(f"   ⚠️  QRS não detectado (Q={len(q_peaks)}, S={len(s_peaks)})")

                    # QT Interval (corrigido por Bazett: QTc = QT / sqrt(RR))
                    t_offsets = delineate[delineate["ECG_T_Offsets"] == 1].index.to_numpy()
                    
                    if len(q_peaks) > 0 and len(t_offsets) > 0:
                        min_len_qt = min(len(q_peaks), len(t_offsets))
                        pares_qt = [(q, t) for q, t in zip(q_peaks[:min_len_qt], t_offsets[:min_len_qt]) if t > q]
                        qt_interval_ms = [(t - q) / self.sampling_rate * 1000 for q, t in pares_qt]
                        qt_interval_raw_mean = np.nanmean(qt_interval_ms) if len(qt_interval_ms) > 0 else np.nan
                        
                        # Correção de Bazett
                        rr_intervals_sec = np.diff(info["ECG_R_Peaks"]) / self.sampling_rate
                        rr_medio_sec = np.mean(rr_intervals_sec)
                        
                        if pd.notna(qt_interval_raw_mean) and rr_medio_sec > 0:
                            qt_interval_mean = qt_interval_raw_mean / np.sqrt(rr_medio_sec)
                        else:
                            qt_interval_mean = np.nan
                    else:
                        qt_interval_mean = np.nan
                        if self.debug:
                            print(f"   ⚠️  QT não detectado (Q={len(q_peaks)}, T_offset={len(t_offsets)})")
                    
                    if self.debug:
                        print(f"   ✓ Métricas morfológicas calculadas")
                        
                except Exception as e_morph:
                    if self.debug:
                        print(f"   ⚠️  Erro nas métricas morfológicas: {str(e_morph)[:50]}")
                    qrs_duration_mean = np.nan
                    qt_interval_mean = np.nan

                # ============================================================
                # DICIONÁRIO FINAL
                # ============================================================
                metrics_dict = {
                    'Epoca': epoch_name,
                    'HRV_MeanNN': hrv_mean_nn,
                    'HRV_MinNN': hrv_min_nn,
                    'HRV_MaxNN': hrv_max_nn,
                    'RR_Range': rr_range,
                    'Bradycardia': bradycardia,
                    'HRV_SDNN': hrv_sdnn,
                    'HRV_RMSSD': hrv_rmssd,
                    'HRV_pNN50': hrv_pnn50,
                    'QRS_Duration_Mean': qrs_duration_mean,
                    'QT_Interval_Mean': qt_interval_mean,
                    'Heart_Rate': heart_rate,
                    'Num_RR_Intervals': len(rr_intervals_ms),
                }
                all_metrics.append(metrics_dict)
                epochs_processadas += 1
                
                if self.debug:
                    print(f"   ✅ Época processada com sucesso!")
                
            except Exception as e:
                epochs_com_erro += 1
                if self.debug:
                    print(f"   ❌ ERRO: {str(e)[:80]}")
                continue
        
        if self.debug:
            print("\n" + "="*60)
            print(f"✅ Épocas processadas: {epochs_processadas}")
            print(f"❌ Épocas com erro: {epochs_com_erro}")
            print("="*60 + "\n")
                
        return pd.DataFrame(all_metrics)
    
    def get_summary_statistics(self, metrics_df):
        """
        Calcula estatísticas resumidas (média, std, min, max) das métricas por paciente.
        """
        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'Epoca']
        
        summary = {}
        for col in numeric_cols:
            summary[f'{col}_mean'] = metrics_df[col].mean()
            summary[f'{col}_std'] = metrics_df[col].std()
            summary[f'{col}_min'] = metrics_df[col].min()
            summary[f'{col}_max'] = metrics_df[col].max()
        
        return pd.DataFrame([summary])


# ============================================================
# FUNÇÃO DE AUTOMAÇÃO PARA MÚLTIPLOS PACIENTES
# ============================================================
def processar_multiplos_pacientes(pacientes_lista, canal='x', minutos_a_pular=60, 
                                   duracao_em_minutos=10, sampling_rate=200, 
                                   debug=False, metodo='elgendi2010',
                                   usar_wavelet=True, wavelet_level=4, wavelet_atenuacao=0.2):
    """
    Processa múltiplos pacientes e retorna um DataFrame consolidado.
    
    Parâmetros:
    -----------
    pacientes_lista : list
        Lista de IDs de pacientes (ex: ['P0002', 'P0003', ...])
    canal : str
        Canal do ECG ('x', 'y', ou 'z')
    minutos_a_pular : int
        Minutos iniciais a pular
    duracao_em_minutos : int
        Duração do sinal a processar em minutos
    sampling_rate : int
        Taxa de amostragem
    debug : bool
        Se True, mostra mensagens de debug
    metodo : str
        Método de limpeza do NeuroKit2
    usar_wavelet : bool
        Se True, aplica filtro wavelet antes da limpeza
    wavelet_level : int
        Nível de decomposição wavelet
    wavelet_atenuacao : float
        Fator de atenuação para detalhes de alto nível
    
    Retorna:
    --------
    tuple: (df_epocas, df_resumo)
        - df_epocas: DataFrame com métricas por época de cada paciente
        - df_resumo: DataFrame com estatísticas resumidas por paciente
    """
    from ecg_utils import load_ecg_segment
    
    all_epochs = []
    all_summaries = []
    
    print(f"\n{'='*70}")
    print(f"🚀 PROCESSANDO {len(pacientes_lista)} PACIENTES")
    print(f"   Método de limpeza: {metodo}")
    print(f"   Filtro Wavelet: {'SIM' if usar_wavelet else 'NÃO'}")
    if usar_wavelet:
        print(f"   Wavelet level: {wavelet_level}, atenuação: {wavelet_atenuacao}")
    print(f"{'='*70}\n")
    
    for i, paciente_id in enumerate(pacientes_lista, 1):
        try:
            print(f"[{i}/{len(pacientes_lista)}] Processando {paciente_id}...")
            
            # Carrega sinal
            sinal = load_ecg_segment(
                paciente_id, 
                canal=canal, 
                minutos_a_pular=minutos_a_pular, 
                duracao_em_minutos=duracao_em_minutos, 
                sampling_rate=sampling_rate
            )
            
            # Aplica filtro wavelet se solicitado
            if usar_wavelet:
                sinal = wavelet_filter_preservando_qrs(
                    sinal, 
                    wavelet='db6', 
                    level=wavelet_level, 
                    atenuacao=wavelet_atenuacao
                )
                if debug:
                    print(f"   ✓ Filtro wavelet aplicado")
            
            # Calcula métricas
            calculator = ECGMetricCalculator(sinal, sampling_rate=sampling_rate, 
                                            debug=debug, metodo=metodo)
            df_metricas = calculator.calculate_metrics_for_epochs()
            
            if len(df_metricas) == 0:
                print(f"   ⚠️  Nenhuma época processada para {paciente_id}")
                continue
            
            # Adiciona ID do paciente e configurações
            df_metricas['Paciente_ID'] = paciente_id
            all_epochs.append(df_metricas)
            
            # Calcula resumo
            df_resumo = calculator.get_summary_statistics(df_metricas)
            df_resumo['Paciente_ID'] = paciente_id
            df_resumo['metodo'] = metodo
            df_resumo['wavelet'] = 'sim' if usar_wavelet else 'nao'
            df_resumo['wavelet_level'] = wavelet_level if usar_wavelet else None
            df_resumo['wavelet_atenuacao'] = wavelet_atenuacao if usar_wavelet else None
            all_summaries.append(df_resumo)
            
            print(f"   ✅ {len(df_metricas)} épocas processadas\n")
            
        except Exception as e:
            print(f"   ❌ ERRO ao processar {paciente_id}: {str(e)[:80]}\n")
            continue
    
    print(f"{'='*70}")
    print(f"✅ PROCESSAMENTO CONCLUÍDO")
    print(f"   Total de pacientes processados: {len(all_summaries)}")
    print(f"   Total de épocas processadas: {sum(len(df) for df in all_epochs)}")
    print(f"{'='*70}\n")
    
    # Consolida resultados
    df_epocas_final = pd.concat(all_epochs, ignore_index=True) if all_epochs else pd.DataFrame()
    df_resumo_final = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    
    return df_epocas_final, df_resumo_final