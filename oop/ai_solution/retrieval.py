from .config import RetrievalCode

def flags_from_pair(mtg, msg):
    return {
        'mtg_conf': mtg["PreFilter_VA_Confidence"],
        'msg_conf': msg["PreFilter_VA_Confidence"],
        'fail_c1': msg["BTD2_conf"] > msg["c1"],
        'fail_c3': msg["BTD2_conf"] <= msg["c3"],
        'fail_c4': msg["BTD2_conf"] > msg["c4"],
        'fail_btd3': msg["VolcanicAsh_BTD3"] > msg["BTD3thresh"],
        'fail_btdcutoff': msg["BTD2_conf"] > -0.1,
        'fail_conmask': msg["BMCon"] == 'F',
        'fail_libmask': msg["BMLib"] == 'F',
    }

def pick_code(flags):
    # Minimal example; expand to mirror your cases
    m, s = flags['mtg_conf'], flags['msg_conf']
    f = flags
    if m == 4 and s == 0:
        if f['fail_c4'] and f['fail_conmask']: return RetrievalCode.CONF4_C4_CONMASK
        if f['fail_c4']: return RetrievalCode.CONF4_C4
        if f['fail_conmask']: return RetrievalCode.CONF4_CONMASK
        return RetrievalCode.CONF4_OTHER
    if m == 3 and s == 0:
        if f['fail_btdcutoff'] and f['fail_btd3'] and f['fail_conmask']: return RetrievalCode.CONF3_BTDCUTOFF_BTD3_CONMASK
        # ...other combinations...
        return RetrievalCode.CONF3_OTHER
    # ...other mtg/msg conf combinations...
    if m == 0:
        return RetrievalCode.NORET
    return RetrievalCode.OTHER
