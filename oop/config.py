import operator as op

def getcutstr(conf_cut):
    '''
    Examples: 
    "mtg_medva_==_4", "mtg_medva_==_7", "mtg_medva_==_3", "mtg_medva_==_5", "mtg_medva_==_6", "mtg_medva_>_0"
    '''

    l_conf_cut = conf_cut.split("_")
    l_conf_cut[0] = l_conf_cut[0].upper()
    l_conf_cut[1] = VAR_MAP[l_conf_cut[1]]
    if l_conf_cut[2] not in OP_MAP:
        raise ValueError(f"Invalid cutstr. Third element of string must correpond to an operator. Got {l_conf_cut[2]}.")
    cut_str = " ".join(l_conf_cut)
    return cut_str

# ...existing code...
OP_MAP = {'==': op.eq, '!=': op.ne, '>': op.gt, '<': op.lt, '>=': op.ge, '<=': op.le}
OP_MAP_STR = {'==':'eq','!=':'ne','>':'gt','<':'lt','>=':'ge','<=':'le'}
VAR_MAP = {'medva': 'Median_VA_Confidence'}

codes_to_ignore = ["noret"]
RETRIEVAL_CODE_LABELS = {
    "conf7_c1": "MTG Conf 7, MSG Conf 0 fails: C1 Threshold",
    "conf7_other": "MTG Conf 7, MSG Conf 0 fails: Other",
    "conf4_conmask": "MTG Conf 4, MSG Conf 0 fails: Conservative Mask",
    "conf4_c4": "MTG Conf 4, MSG Conf 0 fails: C4 Threshold",
    "conf4_c4_conmask": "MTG Conf 4, MSG Conf 0 fails: C4 & Con Mask",
    "conf4_other": "MTG Conf 4, MSG Conf 0 fails: Other",
    "conf3_btdcutoff_btd3_conmask": "MTG Conf 3, MSG Conf 0 fails: BTDcutoff, BTD3, Con Mask",
    "conf3_btdcutoff_btd3": "MTG Conf 3, MSG Conf 0 fails: BTDcutoff, BTD3",
    "conf3_btdcutoff_conmask": "MTG Conf 3, MSG Conf 0 fails: BTDcutoff, Con Mask",
    "conf3_c3_btd3_conmask": "MTG Conf 3, MSG Conf 0 fails: C3, BTD3, Con Mask",
    "conf3_c3_btd3": "MTG Conf 3, MSG Conf 0 fails: C3, BTD3",
    "conf3_c3_conmask": "MTG Conf 3, MSG Conf 0 fails: C3, Con Mask",
    "conf3_btd3_conmask": "MTG Conf 3, MSG Conf 0 fails: BTD3, Con Mask",
    "conf3_btd3": "MTG Conf 3, MSG Conf 0 fails: BTD3",
    "conf3_conmask": "MTG Conf 3, MSG Conf 0 fails: Con Mask",
    "conf3_btdcutoff": "MTG Conf 3, MSG Conf 0 fails: BTDcutoff",
    "conf3_c3": "MTG Conf 3, MSG Conf 0 fails: C3",
    "conf3_other": "MTG Conf 3, MSG Conf 0 fails: Other",
    "conf1_c3_libmask": "MTG Conf 1, MSG Conf 0 fails: C3, Liberal Mask",
    "conf1_c4_libmask": "MTG Conf 1, MSG Conf 0 fails: C4, Liberal Mask",
    "conf1_c4": "MTG Conf 1, MSG Conf 0 fails: C4 Threshold",
    "conf1_c3": "MTG Conf 1, MSG Conf 0 fails: C3 Threshold",
    "conf1_libmask": "MTG Conf 1, MSG Conf 0 fails: Liberal Mask",
    "conf1_other": "MTG Conf 1, MSG Conf 0 fails: Other",
    "conf7_c1_msgconf1": "MTG Conf 7, MSG Conf 1 fails: C1 Threshold",
    "conf7_other_msgconf1": "MTG Conf 7, MSG Conf 1 fails: Other",
    "conf4_conmask_msgconf1": "MTG Conf 4, MSG Conf 1 fails: Conservative Mask",
    "conf4_c4_msgconf1": "MTG Conf 4, MSG Conf 1 fails: C4 Threshold",
    "conf4_c4_conmask_msgconf1": "MTG Conf 4, MSG Conf 1 fails: C4 & Con Mask",
    "conf4_other_msgconf1": "MTG Conf 4, MSG Conf 1 fails: Other",
    "conf3_btdcutoff_btd3_conmask_msgconf1": "MTG Conf 3, MSG Conf 1 fails: BTDcutoff, BTD3, Con Mask",
    "conf3_btdcutoff_btd3_msgconf1": "MTG Conf 3, MSG Conf 1 fails: BTDcutoff, BTD3",
    "conf3_btdcutoff_conmask_msgconf1": "MTG Conf 3, MSG Conf 1 fails: BTDcutoff, Con Mask",
    "conf3_c3_btd3_conmask_msgconf1": "MTG Conf 3, MSG Conf 1 fails: C3, BTD3, Con Mask",
    "conf3_c3_btd3_msgconf1": "MTG Conf 3, MSG Conf 1 fails: C3, BTD3",
    "conf3_c3_conmask_msgconf1": "MTG Conf 3, MSG Conf 1 fails: C3, Con Mask",
    "conf3_btd3_conmask_msgconf1": "MTG Conf 3, MSG Conf 1 fails: BTD3, Con Mask",
    "conf3_btd3_msgconf1": "MTG Conf 3, MSG Conf 1 fails: BTD3",
    "conf3_conmask_msgconf1": "MTG Conf 3, MSG Conf 1 fails: Con Mask",
    "conf3_btdcutoff_msgconf1": "MTG Conf 3, MSG Conf 1 fails: BTDcutoff",
    "conf3_c3_msgconf1": "MTG Conf 3, MSG Conf 1 fails: C3",
    "conf3_other_msgconf1": "MTG Conf 3, MSG Conf 1 fails: Other",
    "conf1_msgconf1": "MTG Conf 1, MSG Conf 1",
    "conf1_msgconf4": "MTG Conf 1, MSG Conf 4",
    "conf4_msgconf4": "MTG Conf 4, MSG Conf 4",
    "conf7_msgconf7": "MTG Conf 7, MSG Conf 7",
    "conf3_msgconf3": "MTG Conf 3, MSG Conf 3",
    "conf3_msgconf4": "MTG Conf 3, MSG Conf 4",
    "conf6_msgconf3": "MTG Conf 6, MSG Conf 3",
    "conf2_msgconf4": "MSG Conf 4, MTG Conf 2",
    "conf4_c4_conmask_msgconf2": "MTG Conf 4, MSG Conf 2 fails: C4 & Con Mask",
    "other": "Other",
}

DICT_CUT_CONSTRAINT = {
    # ...existing code...
}

def select_cmap(n):
    if n < 9: return plt.get_cmap('Dark2', n)
    if n < 11: return plt.get_cmap('tab10', n)
    if n < 13: return plt.get_cmap('Paired', n)
    return plt.get_cmap('tab20', n)
