import numpy as np
from config import OP_MAP, OP_MAP_STR

def apply_constraints(df, constraints, output_perc=False):
    mask = np.ones(len(df), dtype=bool)
    for var, op_str, val in zip(constraints['variables'], constraints['operators'], constraints['values']):
        use_or = op_str.startswith("or_")
        if use_or:
            op_str = op_str[3:]
        fn = OP_MAP[op_str]
        v = df[val[3:]] if isinstance(val, str) and val.startswith('df_') else val
        a = fn(df[var], v)
        mask = (mask | fn(df[var], v)) if use_or else (mask & fn(df[var], v))
    if output_perc:
        perc_passed = (np.array(mask).sum()/len(mask))*100.
        return (df[mask], perc_passed)
    else:
        return df[mask]

def format_constraints_for_title(constraints):
    vals = [str(v)[3:] if isinstance(v,str) and v.startswith('df_') else str(v) for v in constraints['values']]
    vals = ['BTD_Cutoff' if v==str(-0.1) else v for v in vals]
    items = [f"{var} {op} {val}" for var, op, val in zip(constraints['variables'], constraints['operators'], vals)]
    out, buf = [], []
    for i, s in enumerate(items):
        buf.append(s)
        if i < len(items) - 1:
            buf.append("\n" if (i+1) % 2 == 0 else "; ")
    return "".join(buf)

def format_constraints_for_filename(constraints):
    vals = [str(v)[3:] if isinstance(v,str) and v.startswith('df_') else str(v) for v in constraints['values']]
    return "_".join([f"{var}_{OP_MAP_STR[op[3:] if op.startswith('or_') else op]}_{val}"
                     for var, op, val in zip(constraints['variables'], constraints['operators'], vals)])
