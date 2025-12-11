def flags_from_series(series):
    return {
        'conf': series["PreFilter_VA_Confidence"],
        'fail_c1': series["BTD2_conf"] > series["c1"],
        'fail_c3': series["BTD2_conf"] <= series["c3"],
        'fail_c4': series["BTD2_conf"] > series["c4"],
        'fail_btd3': series["VolcanicAsh_BTD3"] > series["BTD3thresh"],
        'fail_btdcutoff': series["BTD2_conf"] > -0.1,
        'fail_conmask': series["BMCon"] == 'F',
        'fail_libmask': series["BMLib"] == 'F',
    }

def pick_code(flags_msg, flags_mtg):

    cs, ct = flags_msg['conf'], flags_mtg['conf']
    fs, ft = flags_msg, flags_mtg

    # mtg_conf == 4 and msg_conf == 0
    if ct == 4 and cs == 0:
        if fs['fail_c4'] and fs['fail_conmask']: return "conf4_c4_conmask"
        if fs['fail_c4']: return "conf4_c4"
        if fs['fail_conmask']: return "conf4_conmask"
        return "conf4_other"

    # mtg_conf == 3 and msg_conf == 0
    if ct == 3 and cs == 0:
        if fs['fail_btdcutoff'] and fs['fail_btd3'] and fs['fail_conmask']: return "conf3_btdcutoff_btd3_conmask"
        if fs['fail_btdcutoff'] and fs['fail_btd3']: return "conf3_btdcutoff_btd3"
        if fs['fail_btdcutoff'] and fs['fail_conmask']: return "conf3_btdcutoff_conmask"
        if fs['fail_c3'] and fs['fail_btd3'] and fs['fail_conmask']: return "conf3_c3_btd3_conmask"
        if fs['fail_c3'] and fs['fail_btd3']: return "conf3_c3_btd3"
        if fs['fail_c3'] and fs['fail_conmask']: return "conf3_c3_conmask"
        if fs['fail_btd3'] and fs['fail_conmask']: return "conf3_btd3_conmask"
        if fs['fail_btd3']: return "conf3_btd3"
        if fs['fail_conmask']: return "conf3_conmask"
        if fs['fail_btdcutoff']: return "conf3_btdcutoff"
        if fs['fail_c3']: return "conf3_c3"
        return "conf3_other"

    # mtg_conf == 1 and msg_conf == 0
    if ct == 1 and cs == 0:
        if fs['fail_c3'] and fs['fail_libmask']: return "conf1_c3_libmask"
        if fs['fail_c4'] and fs['fail_libmask']: return "conf1_c4_libmask"
        if fs['fail_c4']: return "conf1_c4"
        if fs['fail_c3']: return "conf1_c3"
        if fs['fail_libmask']: return "conf1_libmask"
        return "conf1_other"

    # mtg_conf == 7 and msg_conf == 0
    if ct == 7 and cs == 0:
        if fs['fail_c1']: return "conf7_c1"
        return "conf7_other"

    # mtg_conf == 4 and msg_conf == 1
    if ct == 4 and cs == 1:
        if fs['fail_c4'] and fs['fail_conmask']: return "conf4_c4_conmask_msgconf1"
        if fs['fail_c4']: return "conf4_c4_msgconf1"
        if fs['fail_conmask']: return "conf4_conmask_msgconf1"
        return "conf4_other_msgconf1"

    # mtg_conf == 3 and msg_conf == 1
    if ct == 3 and cs == 1:
        if fs['fail_btdcutoff'] and fs['fail_btd3'] and fs['fail_conmask']: return "conf3_btdcutoff_btd3_conmask_msgconf1"
        if fs['fail_btdcutoff'] and fs['fail_btd3']: return "conf3_btdcutoff_btd3_msgconf1"
        if fs['fail_btdcutoff'] and fs['fail_conmask']: return "conf3_btdcutoff_conmask_msgconf1"
        if fs['fail_c3'] and fs['fail_btd3'] and fs['fail_conmask']: return "conf3_c3_btd3_conmask_msgconf1"
        if fs['fail_c3'] and fs['fail_btd3']: return "conf3_c3_btd3_msgconf1"
        if fs['fail_c3'] and fs['fail_conmask']: return "conf3_c3_conmask_msgconf1"
        if fs['fail_btd3'] and fs['fail_conmask']: return "conf3_btd3_conmask_msgconf1"
        if fs['fail_btd3']: return "conf3_btd3_msgconf1"
        if fs['fail_conmask']: return "conf3_conmask_msgconf1"
        if fs['fail_btdcutoff']: return "conf3_btdcutoff_msgconf1"
        if fs['fail_c3']: return "conf3_c3_msgconf1"
        return "conf3_other_msgconf1"

    # mtg_conf == 1 and msg_conf == 1
    if ct == 1 and cs == 1:
        return "conf1_msgconf1"

    # mtg_conf == 7 and msg_conf == 1
    if ct == 7 and cs == 1:
        if fs['fail_c1']: return "conf7_c1_msgconf1"
        return "conf7_other_msgconf1"

    # mtg_conf == 4 and msg_conf == 4
    if ct == 4 and cs == 4:
        return "conf4_msgconf4"

    # mtg_conf == 4 and msg_conf == 2
    if ct == 4 and cs == 2:
        if fs['fail_c4'] and fs['fail_conmask']: return "conf4_c4_conmask_msgconf2"
        if fs['fail_c4']: return "conf4_c4_msgconf2"
        if fs['fail_conmask']: return "conf4_conmask_msgconf2"
        return "conf4_other_msgconf2"

    # msg_conf == 4 and mtg_conf == 1  
    if ct == 1 and cs == 4:
        return "conf1_msgconf4"

    # mtg_conf == 7 and msg_conf == 7
    if ct == 7 and cs == 7:
        return "conf7_msgconf7"

    # mtg_conf == 6 and msg_conf == 3
    if ct == 6 and cs == 3:
        return "conf6_msgconf3"

    # mtg_conf == 3 and msg_conf == 4
    if ct == 3 and cs == 4:
        return "conf3_msgconf4"

    # mtg_conf == 3 and msg_conf == 3
    if ct == 3 and cs == 3:
        return "conf3_msgconf3"

    if ct == 2 and cs == 4:
        return "conf2_msgconf4"

    if ct == 0:
        return "noret"

    return "other"

