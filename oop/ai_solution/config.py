# ...existing code...
OP_MAP = {'==': op.eq, '!=': op.ne, '>': op.gt, '<': op.lt, '>=': op.ge, '<=': op.le}
OP_MAP_STR = {'==':'eq','!=':'ne','>':'gt','<':'lt','>=':'ge','<=':'le'}

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
