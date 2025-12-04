from .config import RetrievalCode

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

'''
for msg_match, mtg_match in search_matches:
    flags_msg = retrieval.flags_from_series(msg_match)
    flags_mtg = retrieval.flags_from_series(mtg_match)
    retrievalcode = pick_code(flags_msg, flags_mtg)
'''


def pick_code(flags_msg, flags_mtg):

    cs, ct = flags_msg['conf'], flags_mtg['conf']
    fs, ft = flags_msg, flags_mtg

    # From the commented logic: mtg_conf == 0 -> "noret"
    if ct == 0:
        return "noret"

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
        if fs['fail_btdcutoff'] and fs['fail_conmask']: return "conf3_btdcutoff_conmask"  # fix: not conf4
        if fs['fail_c3'] and fs['fail_btd3'] and fs['fail_conmask']: return "conf3_c3_btd3_conmask"  # fix: fail_conmask
        if fs['fail_c3'] and fs['fail_btd3']: return "conf3_c3_btd3"
        if fs['fail_c3'] and fs['fail_conmask']: return "conf3_c3_conmask"
        if fs['fail_btd3'] and fs['fail_conmask']: return "conf3_btd3_conmask"
        if fs['fail_btd3']: return "conf3_btd3"
        if fs['fail_conmask']: return "conf3_conmask"
        if fs['fail_btdcutoff']: return "conf3_btdcutoff"
        if fs['fail_c3']: return "conf3_c3"  # fix: fail_c3
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

    # msg_conf == 4 and mtg_conf == 1  (note: use meeting-side fails per the commented code)
    if cs == 4 and ct == 1:
        if ft['fail_c4'] and ft['fail_conmask']: return "conf1_c4_conmask_msgconf4"
        if ft['fail_c4']: return "conf1_c4_msgconf4"
        if ft['fail_conmask']: return "conf1_conmask_msgconf4"
        return "conf1_other_msgconf4"

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

    # default
    return


def pick_code(flags_msg, flags_mtg):
    cs, ct = flags_msg['conf'], flags_mtg['conf']
    fs, ft = flags_msg, flags_mtg
    if ct == 4 and cs == 0:
        if fs['fail_c4'] and fs['fail_conmask']: return "conf4_c4_conmask"
        if fs['fail_c4']: return "conf4_c4"
        if fs['fail_conmask']: return "conf4_conmask"
        return "conf4_other"
    if ct == 3 and cs == 0:
        if fs['fail_btdcutoff'] and fs['fail_btd3'] and fs['fail_conmask']: return "conf3_btdcutoff_btd3_conmask"
        if fs['fail_btdcutoff'] and fs['fail_btd3']: return "conf3_btdcutoff_btd3"
        if fs['fail_btdcutoff'] and fs['fail_conmask']: return "conf4_btdcutoff_conmask"
        if fs['fail_c3'] and fs['fail_btd3'] and fs['failconmask']: return "conf3_c3_btd3_conmask"
        if fs['fail_c3'] and fs['fail_btd3']: return "conf3_c3_btd3"
        if fs['fail_c3'] and fs['fail_conmask']: return "conf3_c3_conmask"
        if fs['fail_btd3'] and fs['fail_conmask']: return "conf3_btd3_conmask"
        if fs['fail_btd3']: return "conf3_btd3"
        if fs['fail_conmask']: return "conf3_conmask"
        if fs['fail_btdcutoff']: return "conf3_btdcutoff"
        if fs['failc3']: return "conf3_c3"
        return "conf3_other"
    if ct == 1 and cs == 0:
        if fs['fail_c3'] and fs['fail_libmask']: return "conf1_c3_libmask"
        if fs['fail_c4'] and fs['fail_libmask']: return "conf1_c4_libmask"
        if fs['fail_c4']: return "conf1_c4"
        if fs['fail_c3']: return "conf1_c3"
        if fs['fail_libmask']: return "conf1_libmask"
        return "conf1_other"
    if ct == 7 and cs == 0:
        if fs['fail_c1']: return "conf7_c1"
        return "conf7_other"
    if ct == 4 and cs == 1:
        if fs['fail_c4'] and fs['fail_conmask']: return "conf4_c4_conmask_msgconf1"
        if fs['fail_c4']: return "conf4_c4_msgconf1"
        if fs['fail_conmask']: return "conf4_conmask_msgconf1"
        return "conf4_other_msgconf1"
    if ct == 3 and cs == 1:
    if conf_msg == conf_msg == 0:
        return RetrievalCode.NORET
    return RetrievalCode.OTHER


'''
        elif mtg_conf == 3 and msg_conf == 1:
            if failbtdcutoff and failbtd3 and failconmask:
                retrievalcode = RetrievalCode("conf3_btdcutoff_btd3_conmask_msgconf1")
            elif failbtdcutoff and failbtd3:
                retrievalcode = RetrievalCode("conf3_btdcutoff_btd3_msgconf1")
            elif failbtdcutoff and failconmask:
                retrievalcode = RetrievalCode("conf3_btdcutoff_conmask_msgconf1")
            if failc3 and failbtd3 and failconmask:
                retrievalcode = RetrievalCode("conf3_c3_btd3_conmask_msgconf1")
            elif failc3 and failbtd3:
                retrievalcode = RetrievalCode("conf3_c3_btd3_msgconf1")
            elif failc3 and failconmask:
                retrievalcode = RetrievalCode("conf3_c3_conmask_msgconf1")
            elif failbtd3 and failconmask:
                retrievalcode = RetrievalCode("conf3_btd3_conmask_msgconf1")
            elif failbtd3:
                retrievalcode = RetrievalCode("conf3_btd3_msgconf1")
            elif failconmask:
                retrievalcode = RetrievalCode("conf3_conmask_msgconf1")
            elif failbtdcutoff:
                retrievalcode = RetrievalCode("conf3_btdcutoff_msgconf1")
            elif failc3:
                retrievalcode = RetrievalCode("conf3_c3_msgconf1")
            else:
                retrievalcode = RetrievalCode("conf3_other_msgconf1")
        elif mtg_conf == 1 and msg_conf == 1:
            retrievalcode = RetrievalCode("conf1_msgconf1")
        elif mtg_conf == 7 and msg_conf == 1:
            failc1 = msg_btd2 > msg_match["c1"]
            if failc1:
                retrievalcode = RetrievalCode("conf7_c1_msgconf1")
            else:
                retrievalcode = RetrievalCode("conf7_other_msgconf1")
        elif mtg_conf == 4 and msg_conf == 4:
            retrievalcode = RetrievalCode("conf4_msgconf4")
        elif mtg_conf == 4 and msg_conf == 2:
            failc4 = msg_btd2 > msg_match["c4"]
            failconmask = msg_conmask == 'F'
            if failc4 and failconmask:
                retrievalcode = RetrievalCode("conf4_c4_conmask_msgconf2")
            elif failc4:
                retrievalcode = RetrievalCode("conf4_c4_msgconf2")
            elif failconmask:
                retrievalcode = RetrievalCode("conf4_conmask_msgconf2")
            else:
                retrievalcode = RetrievalCode("conf4_other_msgconf2")
        elif msg_conf == 4 and mtg_conf == 1:
            failc4 = mtg_btd2 > mtg_match["c4"]
            failconmask = mtg_conmask == 'F'
            if failc4 and failconmask:
                retrievalcode = RetrievalCode("conf1_c4_conmask_msgconf4")
            elif failc4:
                retrievalcode = RetrievalCode("conf1_c4_msgconf4")
            elif failconmask:
                retrievalcode = RetrievalCode("conf1_conmask_msgconf4")
            else:
                retrievalcode = RetrievalCode("conf1_other_msgconf4")
        elif mtg_conf == 7 and msg_conf == 7:
            retrievalcode = RetrievalCode("conf7_msgconf7")
        elif mtg_conf == 6 and msg_conf == 3:
            retrievalcode = RetrievalCode("conf6_msgconf3")
        elif mtg_conf == 3 and msg_conf == 4:
            retrievalcode = RetrievalCode("conf3_msgconf4")
        elif mtg_conf == 3 and msg_conf == 3:
            retrievalcode = RetrievalCode("conf3_msgconf3")
        elif mtg_conf == 0:
            retrievalcode = RetrievalCode("noret")
        else:
            retrievalcode = RetrievalCode("other")

'''
