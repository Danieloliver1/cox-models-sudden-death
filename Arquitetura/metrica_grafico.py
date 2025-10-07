# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy.integrate
from lifelines import KaplanMeierFitter

# Função auxiliar (sem alterações)
def get_censoring_probabilities(events, durations):
    kmf_cens = KaplanMeierFitter()
    kmf_cens.fit(durations, event_observed=(1 - events))
    return kmf_cens

def manual_integrated_brier_score(survival_prediction_df, durations, events, time_grid):
    """
    Calcula o Integrated Brier Score (IBS) manualmente.
    VERSÃO CORRIGIDA.
    """
    kmf_cens = get_censoring_probabilities(events, durations)

    brier_scores = []
    for t in time_grid:
        pos = survival_prediction_df.index.searchsorted(t, side='right')
        
        if pos == 0:
            preds_at_t = pd.Series(1.0, index=survival_prediction_df.columns)
        else:
            preds_at_t = survival_prediction_df.iloc[pos - 1]

        # A previsão agora é chamada sem o argumento extra
        g_t = kmf_cens.predict(t) # <-- MUDANÇA AQUI
        if g_t == 0: g_t = 1e-10

        weighted_errors = []
        for i in range(len(durations)):
            T_i, delta_i = durations[i], events[i]
            pred_i = preds_at_t.iloc[i]

            # A previsão agora é chamada sem o argumento extra
            g_ti = kmf_cens.predict(T_i) # <-- MUDANÇA AQUI
            if g_ti == 0: g_ti = 1e-10

            if delta_i == 1 and T_i <= t:
                error = (0 - pred_i)**2 / g_ti
                weighted_errors.append(error)
            elif T_i > t:
                error = (1 - pred_i)**2 / g_t
                weighted_errors.append(error)

        if len(weighted_errors) > 0:
            brier_scores.append(np.mean(weighted_errors))
        else:
            brier_scores.append(0)

    integral = scipy.integrate.simpson(brier_scores, time_grid)
    return integral / (time_grid[-1] - time_grid[0])

def manual_integrated_nbll(survival_prediction_df, durations, events, time_grid):
    """
    Calcula o Integrated Negative Binomial Log-Likelihood (INBLL) manualmente.
    VERSÃO CORRIGIDA.
    """
    kmf_cens = get_censoring_probabilities(events, durations)

    nbll_scores = []
    for t in time_grid:
        pos = survival_prediction_df.index.searchsorted(t, side='right')

        if pos == 0:
            preds_at_t = pd.Series(1.0 - 1e-10, index=survival_prediction_df.columns)
        else:
            preds_at_t = survival_prediction_df.iloc[pos - 1]

        # A previsão agora é chamada sem o argumento extra
        g_t = kmf_cens.predict(t) # <-- MUDANÇA AQUI
        if g_t == 0: g_t = 1e-10

        weighted_log_likelihoods = []
        for i in range(len(durations)):
            T_i, delta_i = durations[i], events[i]
            pred_i = max(min(preds_at_t.iloc[i], 1 - 1e-10), 1e-10)
            
            # A previsão agora é chamada sem o argumento extra
            g_ti = kmf_cens.predict(T_i) # <-- MUDANÇA AQUI
            if g_ti == 0: g_ti = 1e-10

            if delta_i == 1 and T_i <= t:
                ll = np.log(1 - pred_i) / g_ti
                weighted_log_likelihoods.append(ll)
            elif T_i > t:
                ll = np.log(pred_i) / g_t
                weighted_log_likelihoods.append(ll)

        if len(weighted_log_likelihoods) > 0:
            nbll_scores.append(-np.mean(weighted_log_likelihoods))
        else:
            nbll_scores.append(0)

    integral = scipy.integrate.simpson(nbll_scores, time_grid)
    return integral / (time_grid[-1] - time_grid[0])