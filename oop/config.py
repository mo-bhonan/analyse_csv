
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
    cut_str = " ".join(conf_cut)
    return cut_str

# ...existing code...
OP_MAP = {'==': op.eq, '!=': op.ne, '>': op.gt, '<': op.lt, '>=': op.ge, '<=': op.le}
OP_MAP_STR = {'==':'eq','!=':'ne','>':'gt','<':'lt','>=':'ge','<=':'le'}
VAR_MAP = {'medva': 'Median_VA_Confidence'}
'''

class RetrievalCode(Enum):
    # ...existing code...
    pass

RETRIEVAL_CODE_LABELS = {
    # ...existing code...
}

DICT_CUT_CONSTRAINT = {
    # ...existing code...
}

def select_cmap(n):
    if n < 9: return plt.get_cmap('Dark2', n)
    if n < 11: return plt.get_cmap('tab10', n)
    if n < 13: return plt.get_cmap('Paired', n)
    return plt.get_cmap('tab20', n)
'''
